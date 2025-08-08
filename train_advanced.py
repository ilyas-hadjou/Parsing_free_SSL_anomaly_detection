"""
Advanced training script for LogGraph-SSL with all optimizations.
Includes hard negative mining, contrastive learning, and improved SSL tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from log_graph_builder import LogGraphBuilder
from ssl_tasks import MaskedNodePrediction, EdgePrediction, NodeClassification, CombinedSSLTasks
from gnn_model import LogGraphSSL
from utils import load_log_data, set_seed


class AdvancedSSLTrainer:
    """
    Advanced trainer with all optimizations for LogGraph-SSL.
    """
    
    def __init__(self,
                 model: LogGraphSSL,
                 device: torch.device = torch.device('cpu'),
                 learning_rate: float = 0.0001,
                 weight_decay: float = 1e-3):
        """
        Initialize advanced SSL trainer.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer with different learning rates for different components
        encoder_params = list(self.model.encoder.parameters())
        ssl_head_params = (list(self.model.masked_node_head.parameters()) + 
                          list(self.model.edge_pred_head.parameters()) + 
                          list(self.model.node_class_head.parameters()))
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': learning_rate},
            {'params': ssl_head_params, 'lr': learning_rate * 2}  # Higher LR for heads
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'ssl_losses': [],
            'contrastive_losses': [],
            'diversity_losses': [],
            'edge_aucs': [],
            'node_accs': []
        }
    
    def train_epoch(self, train_graph: Data, epoch: int) -> dict:
        """Train for one epoch with all optimizations."""
        self.model.train()
        train_graph = train_graph.to(str(self.device))
        
        # Ensure we have valid data
        if train_graph.x is None or train_graph.edge_index is None:
            raise ValueError("Graph data is incomplete")
        
        total_loss = 0
        ssl_loss = 0
        contrastive_loss = 0
        diversity_loss = 0
        
        self.optimizer.zero_grad()
        
        # 1. Standard forward pass
        embeddings = self.model(train_graph.x, train_graph.edge_index)
        
        # 2. SSL Tasks with improved architecture
        
        # Masked Node Prediction
        num_nodes = train_graph.x.size(0)
        mask_rate = 0.15
        num_mask = int(num_nodes * mask_rate)
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]
        
        masked_embeddings = embeddings[mask_indices]
        target_features = train_graph.x[mask_indices]
        reconstructed = self.model.masked_node_head(masked_embeddings)
        mask_loss = nn.MSELoss()(reconstructed, target_features)
        
        # Enhanced Edge Prediction with Hard Negatives
        pos_edge_index = train_graph.edge_index
        neg_edge_index = negative_sampling(
            pos_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        
        pos_logits, neg_logits = self.model.forward_edge_prediction_with_hard_negatives(
            train_graph.x, train_graph.edge_index, pos_edge_index, neg_edge_index
        )
        
        # Hard negative mining - select hardest negatives
        with torch.no_grad():
            neg_scores = torch.sigmoid(neg_logits)
            hard_neg_mask = neg_scores > 0.3  # Select harder negatives
            
        if hard_neg_mask.sum() > 0:
            hard_neg_logits = neg_logits[hard_neg_mask]
            easy_neg_logits = neg_logits[~hard_neg_mask]
            
            # Weight hard negatives more
            edge_loss = (nn.BCEWithLogitsLoss()(pos_logits, torch.ones_like(pos_logits)) +
                        2.0 * nn.BCEWithLogitsLoss()(hard_neg_logits, torch.zeros_like(hard_neg_logits)) +
                        nn.BCEWithLogitsLoss()(easy_neg_logits, torch.zeros_like(easy_neg_logits))) / 3
        else:
            edge_loss = (nn.BCEWithLogitsLoss()(pos_logits, torch.ones_like(pos_logits)) +
                        nn.BCEWithLogitsLoss()(neg_logits, torch.zeros_like(neg_logits))) / 2
        
        # Node Classification with pseudo-labels
        node_logits = self.model.forward_node_classification(train_graph.x, train_graph.edge_index)
        
        # Generate pseudo-labels based on graph structure
        degrees = torch.bincount(train_graph.edge_index[0], minlength=num_nodes).float()
        pseudo_labels = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        pseudo_labels[degrees > degrees.quantile(0.8)] = 2  # High degree nodes
        pseudo_labels[(degrees > degrees.quantile(0.2)) & (degrees <= degrees.quantile(0.8))] = 1  # Medium
        # Low degree nodes remain 0
        
        node_loss = nn.CrossEntropyLoss()(node_logits, pseudo_labels)
        
        # 3. Contrastive Learning
        # Create two augmented views
        aug_x1, aug_edge_index1 = self.model.graph_augment(train_graph.x, train_graph.edge_index, 'dropout')
        aug_x2, aug_edge_index2 = self.model.graph_augment(train_graph.x, train_graph.edge_index, 'mask')
        
        # Get embeddings for augmented graphs
        emb1 = self.model(aug_x1, aug_edge_index1)
        emb2 = self.model(aug_x2, aug_edge_index2)
        
        # Graph-level pooling for contrastive learning
        graph_emb1 = torch.mean(emb1, dim=0, keepdim=True)
        graph_emb2 = torch.mean(emb2, dim=0, keepdim=True)
        
        contrastive_loss_val = self.model.contrastive_loss(graph_emb1, graph_emb2)
        
        # 4. Diversity Loss to prevent collapse
        diversity_loss_val = self.model.diversity_loss(embeddings)
        variance_loss_val = self.model.embedding_variance_loss(embeddings)
        
        # 5. Combine all losses
        ssl_loss_val = mask_loss + edge_loss + node_loss
        total_loss_val = (ssl_loss_val + 
                         0.1 * contrastive_loss_val + 
                         0.05 * diversity_loss_val + 
                         0.05 * variance_loss_val)
        
        # Backward pass
        total_loss_val.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Edge prediction AUC
            all_logits = torch.cat([pos_logits, neg_logits])
            all_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            edge_auc = self._calculate_auc(all_logits, all_labels)
            
            # Node classification accuracy
            node_acc = (node_logits.argmax(dim=1) == pseudo_labels).float().mean()
        
        return {
            'total_loss': total_loss_val.item(),
            'ssl_loss': ssl_loss_val.item(),
            'contrastive_loss': contrastive_loss_val.item(),
            'diversity_loss': diversity_loss_val.item(),
            'mask_loss': mask_loss.item(),
            'edge_loss': edge_loss.item(),
            'node_loss': node_loss.item(),
            'edge_auc': edge_auc,
            'node_acc': node_acc.item()
        }
    
    def _calculate_auc(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate AUC score."""
        try:
            from sklearn.metrics import roc_auc_score
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            return float(roc_auc_score(labels_np, probs))
        except:
            return 0.5
    
    def validate_epoch(self, val_graph: Data) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        val_graph = val_graph.to(str(self.device))
        
        if val_graph.x is None or val_graph.edge_index is None:
            raise ValueError("Validation graph data is incomplete")
        
        with torch.no_grad():
            embeddings = self.model(val_graph.x, val_graph.edge_index)
            
            # Validation metrics
            embedding_variance = torch.var(embeddings, dim=0).mean().item()
            cosine_sim = torch.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1).mean().item()
            
            return {
                'embedding_variance': embedding_variance,
                'cosine_similarity': cosine_sim,
                'embedding_norm': torch.norm(embeddings, dim=1).mean().item()
            }
    
    def train(self, train_graph: Data, val_graph: Data, num_epochs: int = 50) -> dict:
        """Full training loop with all optimizations."""
        print(f"Starting advanced training for {num_epochs} epochs...")
        
        best_edge_auc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_graph, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_graph)
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['ssl_losses'].append(train_metrics['ssl_loss'])
            self.history['contrastive_losses'].append(train_metrics['contrastive_loss'])
            self.history['diversity_losses'].append(train_metrics['diversity_loss'])
            self.history['edge_aucs'].append(train_metrics['edge_auc'])
            self.history['node_accs'].append(train_metrics['node_acc'])
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{num_epochs}")
                print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
                print(f"  Edge AUC: {train_metrics['edge_auc']:.4f}")
                print(f"  Node Acc: {train_metrics['node_acc']:.4f}")
                print(f"  Embedding Var: {val_metrics['embedding_variance']:.6f}")
                print(f"  Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
            
            # Early stopping based on edge AUC improvement
            if train_metrics['edge_auc'] > best_edge_auc:
                best_edge_auc = train_metrics['edge_auc']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience and epoch > 20:
                print(f"Early stopping at epoch {epoch} with best Edge AUC: {best_edge_auc:.4f}")
                break
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Advanced LogGraph-SSL Training')
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load and process data
    print("Loading data...")
    messages = load_log_data(args.data_path)
    print(f"Loaded {len(messages)} messages")
    
    # Build graph
    graph_builder = LogGraphBuilder()
    train_graph = graph_builder.build_graph_from_logs(messages[:int(0.9*len(messages))])
    val_graph = graph_builder.build_graph_from_logs(messages[int(0.9*len(messages)):])
    
    print(f"Train graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Val graph: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    
    # Ensure graphs have valid data
    if train_graph.x is None or train_graph.edge_index is None:
        raise ValueError("Training graph is incomplete")
    if val_graph.x is None or val_graph.edge_index is None:
        raise ValueError("Validation graph is incomplete")
    
    # Initialize model
    model = LogGraphSSL(
        input_dim=train_graph.x.shape[1],
        hidden_dims=[128, 64],
        output_dim=64,
        encoder_type='gat',
        num_heads=8,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AdvancedSSLTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-3
    )
    
    # Train
    history = trainer.train(train_graph, val_graph, args.epochs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"advanced_loggraph_ssl_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'advanced_model.pth'))
    torch.save(graph_builder, os.path.join(output_dir, 'graph_builder.pth'))
    
    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 3, 2)
    plt.plot(history['edge_aucs'])
    plt.title('Edge Prediction AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.subplot(2, 3, 3)
    plt.plot(history['node_accs'])
    plt.title('Node Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 3, 4)
    plt.plot(history['ssl_losses'], label='SSL')
    plt.plot(history['contrastive_losses'], label='Contrastive')
    plt.plot(history['diversity_losses'], label='Diversity')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_training_curves.png'), dpi=300)
    
    print(f"Advanced training completed! Results saved to {output_dir}")
    print(f"Final Edge AUC: {history['edge_aucs'][-1]:.4f}")
    print(f"Final Node Accuracy: {history['node_accs'][-1]:.4f}")


if __name__ == "__main__":
    main()
