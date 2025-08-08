"""
Full dataset training script for LogGraph-SSL with comprehensive optimizations.
Uses the complete HDFS dataset for maximum performance evaluation.
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
import gc

from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from utils import load_log_data, set_seed


class FullDatasetTrainer:
    """
    Full dataset trainer for LogGraph-SSL with all optimizations.
    """
    
    def __init__(self,
                 model: LogGraphSSL,
                 device: torch.device = torch.device('cpu'),
                 learning_rate: float = 0.0005,
                 weight_decay: float = 1e-3,
                 use_gradient_accumulation: bool = True):
        """
        Initialize full dataset trainer.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_gradient_accumulation = use_gradient_accumulation
        
        # Use AdamW optimizer with different learning rates for different components
        encoder_params = list(self.model.encoder.parameters())
        ssl_head_params = (list(self.model.masked_node_head.parameters()) + 
                          list(self.model.edge_pred_head.parameters()) + 
                          list(self.model.node_class_head.parameters()))
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': learning_rate},
            {'params': ssl_head_params, 'lr': learning_rate * 1.5}  # Slightly higher for heads
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=learning_rate * 2,
            total_steps=50,  # Will be updated based on actual epochs
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'ssl_losses': [],
            'contrastive_losses': [],
            'diversity_losses': [],
            'edge_aucs': [],
            'node_accs': [],
            'embedding_vars': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_graph: Data, epoch: int, accumulation_steps: int = 2) -> dict:
        """Train for one epoch with gradient accumulation for large graphs."""
        self.model.train()
        train_graph = train_graph.to(str(self.device))
        
        if train_graph.x is None or train_graph.edge_index is None:
            raise ValueError("Graph data is incomplete")
        
        # Use gradient accumulation for large graphs
        effective_batch_size = accumulation_steps
        
        total_loss = 0
        ssl_loss_total = 0
        contrastive_loss_total = 0
        diversity_loss_total = 0
        
        self.optimizer.zero_grad()
        
        for step in range(accumulation_steps):
            # Sample different parts of the graph for each accumulation step
            num_nodes = train_graph.x.size(0)
            
            # 1. Forward pass on full graph
            embeddings = self.model(train_graph.x, train_graph.edge_index)
            
            # 2. Masked Node Prediction (higher mask rate for full dataset)
            mask_rate = 0.2  # Increased for better learning
            num_mask = max(100, int(num_nodes * mask_rate))
            mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]
            
            masked_embeddings = embeddings[mask_indices]
            target_features = train_graph.x[mask_indices]
            reconstructed = self.model.masked_node_head(masked_embeddings)
            mask_loss = nn.MSELoss()(reconstructed, target_features)
            
            # 3. Enhanced Edge Prediction with larger sampling
            pos_edge_index = train_graph.edge_index
            num_edges = min(10000, pos_edge_index.size(1))  # Larger sample for full dataset
            edge_indices = torch.randperm(pos_edge_index.size(1), device=self.device)[:num_edges]
            
            sampled_pos_edges = pos_edge_index[:, edge_indices]
            neg_edge_index = negative_sampling(
                sampled_pos_edges, num_nodes=num_nodes,
                num_neg_samples=num_edges
            )
            
            # Get edge predictions with hard negatives
            pos_logits, neg_logits = self.model.forward_edge_prediction_with_hard_negatives(
                train_graph.x, train_graph.edge_index, sampled_pos_edges, neg_edge_index
            )
            
            # Advanced edge loss with hard negative mining
            with torch.no_grad():
                neg_scores = torch.sigmoid(neg_logits)
                hard_neg_mask = neg_scores > 0.4  # More challenging threshold
                
            if hard_neg_mask.sum() > 0:
                hard_neg_logits = neg_logits[hard_neg_mask]
                easy_neg_logits = neg_logits[~hard_neg_mask]
                
                # Weight hard negatives more heavily
                edge_loss = (nn.BCEWithLogitsLoss()(pos_logits, torch.ones_like(pos_logits)) +
                            3.0 * nn.BCEWithLogitsLoss()(hard_neg_logits, torch.zeros_like(hard_neg_logits)) +
                            nn.BCEWithLogitsLoss()(easy_neg_logits, torch.zeros_like(easy_neg_logits))) / 3
            else:
                edge_loss = (nn.BCEWithLogitsLoss()(pos_logits, torch.ones_like(pos_logits)) +
                            nn.BCEWithLogitsLoss()(neg_logits, torch.zeros_like(neg_logits))) / 2
            
            # 4. Node Classification with improved pseudo-labels
            node_logits = self.model.forward_node_classification(train_graph.x, train_graph.edge_index)
            
            # More sophisticated pseudo-labeling based on multiple graph properties
            degrees = torch.bincount(train_graph.edge_index[0], minlength=num_nodes).float()
            clustering_coeff = self._compute_clustering_coefficient(train_graph.edge_index, num_nodes)
            
            # Combine degree and clustering for better pseudo-labels
            degree_percentiles = torch.quantile(degrees, torch.tensor([0.33, 0.67], device=self.device))
            clustering_percentiles = torch.quantile(clustering_coeff, torch.tensor([0.33, 0.67], device=self.device))
            
            pseudo_labels = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
            
            # High degree + high clustering = central nodes (label 2)
            central_mask = (degrees > degree_percentiles[1]) & (clustering_coeff > clustering_percentiles[1])
            pseudo_labels[central_mask] = 2
            
            # Medium degree or medium clustering = intermediate nodes (label 1)
            intermediate_mask = ((degrees > degree_percentiles[0]) & (degrees <= degree_percentiles[1])) | \
                              ((clustering_coeff > clustering_percentiles[0]) & (clustering_coeff <= clustering_percentiles[1]))
            pseudo_labels[intermediate_mask] = 1
            
            # Low degree + low clustering = peripheral nodes (label 0)
            node_loss = nn.CrossEntropyLoss()(node_logits, pseudo_labels)
            
            # 5. Contrastive Learning with multiple augmentations
            aug_x1, aug_edge_index1 = self.model.graph_augment(train_graph.x, train_graph.edge_index, 'dropout')
            aug_x2, aug_edge_index2 = self.model.graph_augment(train_graph.x, train_graph.edge_index, 'noise')
            
            # Get embeddings for augmented graphs
            emb1 = self.model(aug_x1, aug_edge_index1)
            emb2 = self.model(aug_x2, aug_edge_index2)
            
            # Node-level contrastive learning (sample nodes)
            node_sample_size = min(1000, num_nodes)
            node_indices = torch.randperm(num_nodes, device=self.device)[:node_sample_size]
            
            sampled_emb1 = emb1[node_indices]
            sampled_emb2 = emb2[node_indices]
            
            contrastive_loss_val = self.model.contrastive_loss(sampled_emb1, sampled_emb2)
            
            # 6. Regularization losses
            diversity_loss_val = self.model.diversity_loss(embeddings)
            variance_loss_val = self.model.embedding_variance_loss(embeddings)
            
            # 7. Combine all losses with adaptive weighting
            ssl_loss_val = mask_loss + edge_loss + node_loss
            regularization_loss = 0.1 * diversity_loss_val + 0.05 * variance_loss_val
            contrastive_weight = 0.2 * (1 + 0.1 * epoch)  # Increase contrastive weight over time
            
            step_loss = (ssl_loss_val + 
                        contrastive_weight * contrastive_loss_val + 
                        regularization_loss) / accumulation_steps
            
            # Backward pass
            step_loss.backward()
            
            # Accumulate losses
            total_loss += step_loss.item()
            ssl_loss_total += ssl_loss_val.item() / accumulation_steps
            contrastive_loss_total += contrastive_loss_val.item() / accumulation_steps
            diversity_loss_total += diversity_loss_val.item() / accumulation_steps
        
        # Gradient clipping and optimization step
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
            
            # Embedding variance
            embedding_var = torch.var(embeddings, dim=0).mean()
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
        
        # Memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return {
            'total_loss': total_loss,
            'ssl_loss': ssl_loss_total,
            'contrastive_loss': contrastive_loss_total,
            'diversity_loss': diversity_loss_total,
            'mask_loss': mask_loss.item(),
            'edge_loss': edge_loss.item(),
            'node_loss': node_loss.item(),
            'edge_auc': edge_auc,
            'node_acc': node_acc.item(),
            'embedding_var': embedding_var.item(),
            'learning_rate': current_lr
        }
    
    def _compute_clustering_coefficient(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute local clustering coefficient for each node."""
        clustering_coeffs = torch.zeros(num_nodes, device=edge_index.device)
        
        # Convert to adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src, dst = int(edge_index[0, i].item()), int(edge_index[1, i].item())
            if src != dst:  # Avoid self-loops
                adj_list[src].append(dst)
                adj_list[dst].append(src)
        
        # Compute clustering coefficient for each node
        for node in range(num_nodes):
            neighbors = list(set(adj_list[node]))  # Remove duplicates
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs[node] = 0.0
            else:
                # Count triangles
                triangles = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if neighbors[j] in adj_list[neighbors[i]]:
                            triangles += 1
                
                # Clustering coefficient = 2 * triangles / (k * (k - 1))
                clustering_coeffs[node] = 2.0 * triangles / (k * (k - 1))
        
        return clustering_coeffs
    
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
            
            # Comprehensive validation metrics
            embedding_variance = torch.var(embeddings, dim=0).mean().item()
            cosine_sim = torch.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1).mean().item()
            embedding_norm = torch.norm(embeddings, dim=1).mean().item()
            
            # Check for representation collapse
            pairwise_distances = torch.pdist(embeddings)
            mean_distance = pairwise_distances.mean().item()
            
        # Memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return {
            'embedding_variance': embedding_variance,
            'cosine_similarity': cosine_sim,
            'embedding_norm': embedding_norm,
            'mean_pairwise_distance': mean_distance
        }
    
    def train(self, train_graph: Data, val_graph: Data, num_epochs: int = 30) -> dict:
        """Full training loop with comprehensive monitoring."""
        print(f"Starting full dataset training for {num_epochs} epochs...")
        print(f"Train graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Update scheduler total steps
        self.scheduler.total_steps = num_epochs
        
        best_edge_auc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_graph, epoch)
            
            # Validation every 3 epochs to save time
            if epoch % 3 == 0:
                val_metrics = self.validate_epoch(val_graph)
            else:
                val_metrics = {'embedding_variance': 0, 'cosine_similarity': 0, 'embedding_norm': 0, 'mean_pairwise_distance': 0}
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['ssl_losses'].append(train_metrics['ssl_loss'])
            self.history['contrastive_losses'].append(train_metrics['contrastive_loss'])
            self.history['diversity_losses'].append(train_metrics['diversity_loss'])
            self.history['edge_aucs'].append(train_metrics['edge_auc'])
            self.history['node_accs'].append(train_metrics['node_acc'])
            self.history['embedding_vars'].append(train_metrics['embedding_var'])
            self.history['learning_rates'].append(train_metrics['learning_rate'])
            
            # Print detailed progress
            if epoch % 2 == 0:
                print(f"\nEpoch {epoch:3d}/{num_epochs}")
                print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
                print(f"  SSL Loss: {train_metrics['ssl_loss']:.4f}")
                print(f"  Edge AUC: {train_metrics['edge_auc']:.4f}")
                print(f"  Node Acc: {train_metrics['node_acc']:.4f}")
                print(f"  Embedding Var: {train_metrics['embedding_var']:.6f}")
                print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")
                if epoch % 3 == 0:
                    print(f"  Val Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
                    print(f"  Val Pairwise Dist: {val_metrics['mean_pairwise_distance']:.4f}")
            
            # Save best model
            if train_metrics['edge_auc'] > best_edge_auc:
                best_edge_auc = train_metrics['edge_auc']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  ✅ New best Edge AUC: {best_edge_auc:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience and epoch > 15:
                print(f"\nEarly stopping at epoch {epoch} with best Edge AUC: {best_edge_auc:.4f}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with Edge AUC: {best_edge_auc:.4f}")
        
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Full Dataset LogGraph-SSL Training')
    parser.add_argument('--train_data', type=str, default='hdfs_full_train.txt', help='Training data path')
    parser.add_argument('--test_data', type=str, default='hdfs_full_test.txt', help='Test data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 64], help='Hidden dimensions')
    parser.add_argument('--output_dim', type=int, default=64, help='Output embedding dimension')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "CPU mode")
    
    # Set seeds for reproducibility
    set_seed(42)
    
    # Load data
    print("Loading training data...")
    train_messages = load_log_data(args.train_data)
    print(f"Loaded {len(train_messages)} training messages")
    
    # For validation, use a subset of test data
    print("Loading test data for validation...")
    test_messages = load_log_data(args.test_data)
    val_messages = test_messages[:5000]  # Use first 5k for validation
    print(f"Using {len(val_messages)} messages for validation")
    
    # Build graphs
    print("Building training graph...")
    graph_builder = LogGraphBuilder()
    train_graph = graph_builder.build_graph_from_logs(train_messages)
    
    print("Building validation graph...")
    val_graph = graph_builder.build_graph_from_logs(val_messages)
    
    print(f"Train graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Val graph: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    
    # Ensure graphs have valid data
    if train_graph.x is None or train_graph.edge_index is None:
        raise ValueError("Training graph is incomplete")
    if val_graph.x is None or val_graph.edge_index is None:
        raise ValueError("Validation graph is incomplete")
    
    # Initialize model with larger capacity for full dataset
    model = LogGraphSSL(
        input_dim=train_graph.x.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        encoder_type='gat',
        num_heads=8,
        dropout=0.4
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = FullDatasetTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1e-3,
        use_gradient_accumulation=True
    )
    
    # Train
    history = trainer.train(train_graph, val_graph, args.epochs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"full_dataset_ssl_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and components
    torch.save(model.state_dict(), os.path.join(output_dir, 'full_dataset_model.pth'))
    torch.save(graph_builder, os.path.join(output_dir, 'graph_builder.pth'))
    
    # Save model configuration
    model_config = {
        'input_dim': train_graph.x.shape[1],
        'hidden_dims': args.hidden_dims,
        'output_dim': args.output_dim,
        'encoder_type': 'gat',
        'num_heads': 8,
        'dropout': 0.4
    }
    
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot comprehensive results
    plt.figure(figsize=(20, 12))
    
    # Training curves
    plt.subplot(3, 4, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(3, 4, 2)
    plt.plot(history['edge_aucs'])
    plt.title('Edge Prediction AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.subplot(3, 4, 3)
    plt.plot(history['node_accs'])
    plt.title('Node Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(3, 4, 4)
    plt.plot(history['embedding_vars'])
    plt.title('Embedding Variance')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
    
    # Loss components
    plt.subplot(3, 4, 5)
    plt.plot(history['ssl_losses'], label='SSL')
    plt.plot(history['contrastive_losses'], label='Contrastive')
    plt.plot(history['diversity_losses'], label='Diversity')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(3, 4, 6)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_training_curves.png'), dpi=300, bbox_inches='tight')
    
    print(f"\n🎉 Full dataset training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Final metrics:")
    print(f"  - Edge Prediction AUC: {history['edge_aucs'][-1]:.4f}")
    print(f"  - Node Classification Accuracy: {history['node_accs'][-1]:.4f}")
    print(f"  - Embedding Variance: {history['embedding_vars'][-1]:.6f}")
    print(f"  - Best Edge AUC achieved: {max(history['edge_aucs']):.4f}")


if __name__ == "__main__":
    main()
