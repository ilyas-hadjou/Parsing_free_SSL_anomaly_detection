"""
Memory-efficient training script for LogGraph-SSL.
Optimized for systems with limited GPU memory.
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
import gc

from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from utils import load_log_data, set_seed


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer for LogGraph-SSL.
    """
    
    def __init__(self,
                 model: LogGraphSSL,
                 device: torch.device = torch.device('cpu'),
                 learning_rate: float = 0.0001,
                 weight_decay: float = 1e-3):
        """
        Initialize memory-efficient trainer.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Use standard optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'ssl_losses': [],
            'edge_aucs': [],
            'node_accs': [],
            'embedding_vars': []
        }
    
    def train_epoch(self, train_graph: Data, epoch: int) -> dict:
        """Train for one epoch with memory optimization."""
        self.model.train()
        train_graph = train_graph.to(str(self.device))
        
        if train_graph.x is None or train_graph.edge_index is None:
            raise ValueError("Graph data is incomplete")
        
        self.optimizer.zero_grad()
        
        # Forward pass
        embeddings = self.model(train_graph.x, train_graph.edge_index)
        
        # 1. Masked Node Prediction (reduced mask rate to save memory)
        num_nodes = train_graph.x.size(0)
        mask_rate = 0.05  # Reduced from 0.15
        num_mask = max(1, int(num_nodes * mask_rate))
        mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]
        
        masked_embeddings = embeddings[mask_indices]
        target_features = train_graph.x[mask_indices]
        reconstructed = self.model.masked_node_head(masked_embeddings)
        mask_loss = nn.MSELoss()(reconstructed, target_features)
        
        # 2. Edge Prediction (reduced number of edges)
        pos_edge_index = train_graph.edge_index
        num_edges = min(5000, pos_edge_index.size(1))  # Limit edges to save memory
        edge_indices = torch.randperm(pos_edge_index.size(1), device=self.device)[:num_edges]
        
        sampled_pos_edges = pos_edge_index[:, edge_indices]
        neg_edge_index = negative_sampling(
            sampled_pos_edges, num_nodes=num_nodes,
            num_neg_samples=num_edges
        )
        
        # Simple edge prediction without hard negatives to save memory
        pos_logits, neg_logits = self.model.forward_edge_prediction_with_hard_negatives(
            train_graph.x, train_graph.edge_index, sampled_pos_edges, neg_edge_index
        )
        
        edge_loss = (nn.BCEWithLogitsLoss()(pos_logits, torch.ones_like(pos_logits)) +
                    nn.BCEWithLogitsLoss()(neg_logits, torch.zeros_like(neg_logits))) / 2
        
        # 3. Node Classification (simplified pseudo-labels)
        node_logits = self.model.forward_node_classification(train_graph.x, train_graph.edge_index)
        
        # Simple pseudo-labels based on node degrees
        degrees = torch.bincount(train_graph.edge_index[0], minlength=num_nodes).float()
        degree_threshold = degrees.median()
        pseudo_labels = (degrees > degree_threshold).long()
        
        # Use only binary classification to save memory
        node_logits_binary = node_logits[:, :2]  # Only use first 2 classes
        node_loss = nn.CrossEntropyLoss()(node_logits_binary, pseudo_labels)
        
        # 4. Diversity loss (simplified)
        diversity_loss_val = self.model.diversity_loss(embeddings)
        
        # Combine losses
        total_loss = mask_loss + edge_loss + node_loss + 0.01 * diversity_loss_val
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Edge prediction AUC
            all_logits = torch.cat([pos_logits, neg_logits])
            all_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            edge_auc = self._calculate_auc(all_logits, all_labels)
            
            # Node classification accuracy
            node_acc = (node_logits_binary.argmax(dim=1) == pseudo_labels).float().mean()
            
            # Embedding variance
            embedding_var = torch.var(embeddings, dim=0).mean()
        
        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return {
            'total_loss': total_loss.item(),
            'mask_loss': mask_loss.item(),
            'edge_loss': edge_loss.item(),
            'node_loss': node_loss.item(),
            'diversity_loss': diversity_loss_val.item(),
            'edge_auc': edge_auc,
            'node_acc': node_acc.item(),
            'embedding_var': embedding_var.item()
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
            # Use smaller batch for validation to save memory
            embeddings = self.model(val_graph.x, val_graph.edge_index)
            
            # Validation metrics
            embedding_variance = torch.var(embeddings, dim=0).mean().item()
            cosine_sim = torch.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1).mean().item()
            
        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return {
            'embedding_variance': embedding_variance,
            'cosine_similarity': cosine_sim,
            'embedding_norm': torch.norm(embeddings, dim=1).mean().item()
        }
    
    def train(self, train_graph: Data, val_graph: Data, num_epochs: int = 30) -> dict:
        """Full training loop."""
        print(f"Starting memory-efficient training for {num_epochs} epochs...")
        
        best_edge_auc = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_graph, epoch)
            
            # Validation every 5 epochs to save time
            if epoch % 5 == 0:
                val_metrics = self.validate_epoch(val_graph)
            else:
                val_metrics = {'embedding_variance': 0, 'cosine_similarity': 0}
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['ssl_losses'].append(
                train_metrics['mask_loss'] + train_metrics['edge_loss'] + train_metrics['node_loss']
            )
            self.history['edge_aucs'].append(train_metrics['edge_auc'])
            self.history['node_accs'].append(train_metrics['node_acc'])
            self.history['embedding_vars'].append(train_metrics['embedding_var'])
            
            # Print progress
            if epoch % 3 == 0:
                print(f"Epoch {epoch:3d}/{num_epochs}")
                print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
                print(f"  Edge AUC: {train_metrics['edge_auc']:.4f}")
                print(f"  Node Acc: {train_metrics['node_acc']:.4f}")
                print(f"  Embedding Var: {train_metrics['embedding_var']:.6f}")
                if epoch % 5 == 0:
                    print(f"  Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
            
            if train_metrics['edge_auc'] > best_edge_auc:
                best_edge_auc = train_metrics['edge_auc']
        
        print(f"Training completed with best Edge AUC: {best_edge_auc:.4f}")
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Memory-Efficient LogGraph-SSL Training')
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--sample_size', type=int, default=20000, help='Sample size for training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load and sample data to reduce memory usage
    print("Loading data...")
    messages = load_log_data(args.data_path)
    
    # Sample to reduce memory usage
    if len(messages) > args.sample_size:
        indices = np.random.choice(len(messages), args.sample_size, replace=False)
        messages = [messages[i] for i in sorted(indices)]
    
    print(f"Using {len(messages)} messages for training")
    
    # Build graph
    graph_builder = LogGraphBuilder()
    
    # Use smaller graphs to fit in memory
    train_size = int(0.8 * len(messages))
    train_graph = graph_builder.build_graph_from_logs(messages[:train_size])
    val_graph = graph_builder.build_graph_from_logs(messages[train_size:])
    
    print(f"Train graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Val graph: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    
    # Ensure graphs have valid data
    if train_graph.x is None or train_graph.edge_index is None:
        raise ValueError("Training graph is incomplete")
    if val_graph.x is None or val_graph.edge_index is None:
        raise ValueError("Validation graph is incomplete")
    
    # Initialize smaller model to fit in memory
    model = LogGraphSSL(
        input_dim=train_graph.x.shape[1],
        hidden_dims=[64, 32],  # Smaller dimensions
        output_dim=32,         # Smaller output
        encoder_type='gat',
        num_heads=4,           # Fewer heads
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-3
    )
    
    # Train
    history = trainer.train(train_graph, val_graph, args.epochs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"memory_efficient_ssl_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'memory_efficient_model.pth'))
    torch.save(graph_builder, os.path.join(output_dir, 'graph_builder.pth'))
    
    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training completed! Results saved to {output_dir}")
    print(f"Final Edge AUC: {history['edge_aucs'][-1]:.4f}")
    print(f"Final Node Accuracy: {history['node_accs'][-1]:.4f}")
    print(f"Final Embedding Variance: {history['embedding_vars'][-1]:.6f}")


if __name__ == "__main__":
    main()
