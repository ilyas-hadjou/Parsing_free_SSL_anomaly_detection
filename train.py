"""
Training script for LogGraph-SSL framework.
Pre-trains the model using self-supervised learning tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from log_graph_builder import LogGraphBuilder
from ssl_tasks import MaskedNodePrediction, EdgePrediction, GraphContrastiveLearning, NodeClassification, CombinedSSLTasks
from gnn_model import LogGraphSSL
from utils import load_log_data, set_seed, save_checkpoint, load_checkpoint


class SSLTrainer:
    """
    Trainer for self-supervised learning of LogGraph-SSL model.
    """
    
    def __init__(self,
                 model: LogGraphSSL,
                 ssl_tasks: CombinedSSLTasks,
                 device: torch.device = torch.device('cpu'),
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize SSL trainer.
        
        Args:
            model: LogGraph-SSL model to train
            ssl_tasks: Combined SSL tasks
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.ssl_tasks = ssl_tasks
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'task_losses': {task.__class__.__name__: [] for task in ssl_tasks.tasks}
        }
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        task_losses = {task.__class__.__name__: 0.0 for task in self.ssl_tasks.tasks}
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Training")):
            batch_data = batch_data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Create SSL tasks for the batch
            batch_task_losses = []
            
            for i, task in enumerate(self.ssl_tasks.tasks):
                task_name = task.__class__.__name__
                
                try:
                    # Create task-specific data
                    modified_data, targets = task.create_task(batch_data)
                    modified_data = modified_data.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass based on task type
                    if isinstance(task, MaskedNodePrediction):
                        predictions = self.model.forward_masked_nodes(
                            modified_data.x, modified_data.edge_index, modified_data.mask_indices
                        )
                    elif isinstance(task, EdgePrediction):
                        predictions = self.model.forward_edge_prediction(
                            modified_data.x, modified_data.edge_index, modified_data.edge_label_index
                        )
                    elif isinstance(task, GraphContrastiveLearning):
                        # Handle contrastive learning differently
                        embeddings = self.model.forward_contrastive(
                            modified_data.x, modified_data.edge_index, modified_data.batch
                        )
                        # Split embeddings for two views
                        mid_point = embeddings.size(0) // 2
                        z1, z2 = embeddings[:mid_point], embeddings[mid_point:]
                        task_loss = task.compute_loss(z1, z2)
                        batch_task_losses.append(task_loss)
                        task_losses[task_name] += task_loss.item()
                        continue
                    elif isinstance(task, NodeClassification):
                        predictions = self.model.forward_node_classification(
                            modified_data.x, modified_data.edge_index
                        )
                    else:
                        # Generic forward pass
                        predictions = self.model(modified_data.x, modified_data.edge_index)
                    
                    # Compute task loss
                    task_loss = task.compute_loss(predictions, targets)
                    batch_task_losses.append(task_loss)
                    task_losses[task_name] += task_loss.item()
                    
                except Exception as e:
                    print(f"Error in task {task_name}: {str(e)}")
                    # Create dummy loss to prevent training from stopping
                    dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    batch_task_losses.append(dummy_loss)
            
            # Combine task losses
            if batch_task_losses:
                combined_loss = self.ssl_tasks.compute_combined_loss(batch_task_losses)
                combined_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += combined_loss.item()
            
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_task_losses = {name: loss / max(num_batches, 1) for name, loss in task_losses.items()}
        
        return {'total_loss': avg_loss, **avg_task_losses}
    
    def validate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        task_losses = {task.__class__.__name__: 0.0 for task in self.ssl_tasks.tasks}
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Validation"):
                batch_data = batch_data.to(self.device)
                
                batch_task_losses = []
                
                for i, task in enumerate(self.ssl_tasks.tasks):
                    task_name = task.__class__.__name__
                    
                    try:
                        # Create task-specific data
                        modified_data, targets = task.create_task(batch_data)
                        modified_data = modified_data.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Forward pass based on task type
                        if isinstance(task, MaskedNodePrediction):
                            predictions = self.model.forward_masked_nodes(
                                modified_data.x, modified_data.edge_index, modified_data.mask_indices
                            )
                        elif isinstance(task, EdgePrediction):
                            predictions = self.model.forward_edge_prediction(
                                modified_data.x, modified_data.edge_index, modified_data.edge_label_index
                            )
                        elif isinstance(task, GraphContrastiveLearning):
                            embeddings = self.model.forward_contrastive(
                                modified_data.x, modified_data.edge_index, modified_data.batch
                            )
                            mid_point = embeddings.size(0) // 2
                            z1, z2 = embeddings[:mid_point], embeddings[mid_point:]
                            task_loss = task.compute_loss(z1, z2)
                            batch_task_losses.append(task_loss)
                            task_losses[task_name] += task_loss.item()
                            continue
                        elif isinstance(task, NodeClassification):
                            predictions = self.model.forward_node_classification(
                                modified_data.x, modified_data.edge_index
                            )
                        else:
                            predictions = self.model(modified_data.x, modified_data.edge_index)
                        
                        # Compute task loss
                        task_loss = task.compute_loss(predictions, targets)
                        batch_task_losses.append(task_loss)
                        task_losses[task_name] += task_loss.item()
                        
                    except Exception as e:
                        print(f"Error in validation task {task_name}: {str(e)}")
                        dummy_loss = torch.tensor(0.0, device=self.device)
                        batch_task_losses.append(dummy_loss)
                
                # Combine task losses
                if batch_task_losses:
                    combined_loss = self.ssl_tasks.compute_combined_loss(batch_task_losses)
                    total_loss += combined_loss.item()
                
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_task_losses = {name: loss / max(num_batches, 1) for name, loss in task_losses.items()}
        
        return {'total_loss': avg_loss, **avg_task_losses}
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              save_dir: str = './checkpoints',
              save_every: int = 10) -> None:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['total_loss'])
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            for task_name, loss in train_metrics.items():
                if task_name != 'total_loss':
                    print(f"  {task_name}: {loss:.4f}")
                    self.history['task_losses'][task_name].append(loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_metrics['total_loss'])
                
                print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['total_loss'])
                
                # Early stopping and checkpointing
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                    save_checkpoint(self.model, self.optimizer, epoch, val_metrics['total_loss'], checkpoint_path)
                    print(f"New best model saved with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {max_patience} epochs without improvement")
                    break
            
            # Regular checkpointing
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                current_loss = val_metrics['total_loss'] if val_loader else train_metrics['total_loss']
                save_checkpoint(self.model, self.optimizer, epoch, current_loss, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch + 1}")
        
        print("Training completed!")
        
        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pth')
        final_loss = val_metrics['total_loss'] if val_loader else train_metrics['total_loss']
        save_checkpoint(self.model, self.optimizer, num_epochs - 1, final_loss, final_path)
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Task losses
        for i, (task_name, losses) in enumerate(self.history['task_losses'].items()):
            row = (i + 1) // 2
            col = (i + 1) % 2
            if row < 2 and col < 2:
                axes[row, col].plot(losses, label=task_name)
                axes[row, col].set_title(f'{task_name} Loss')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LogGraph-SSL model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to log data')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 64], help='Hidden dimensions')
    parser.add_argument('--output_dim', type=int, default=32, help='Output embedding dimension')
    parser.add_argument('--encoder_type', type=str, default='gcn', choices=['gcn', 'gat', 'sage'], help='Encoder type')
    parser.add_argument('--window_size', type=int, default=5, help='Token co-occurrence window size')
    parser.add_argument('--min_token_freq', type=int, default=2, help='Minimum token frequency')
    parser.add_argument('--max_vocab_size', type=int, default=10000, help='Maximum vocabulary size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Training device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"loggraph_ssl_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Loading and processing data...")
    
    # Load log data
    log_messages = load_log_data(args.data_path)
    print(f"Loaded {len(log_messages)} log messages")
    
    # Build graph
    graph_builder = LogGraphBuilder(
        window_size=args.window_size,
        min_token_freq=args.min_token_freq,
        max_vocab_size=args.max_vocab_size
    )
    
    # Split data
    split_idx = int(0.8 * len(log_messages))
    train_messages = log_messages[:split_idx]
    val_messages = log_messages[split_idx:]
    
    # Build training graph
    train_graph = graph_builder.build_graph_from_logs(train_messages)
    print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    
    # Build validation graph
    val_graph = graph_builder.build_graph_from_logs(val_messages)
    print(f"Validation graph: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    
    # Create data loaders
    train_loader = DataLoader([train_graph], batch_size=1, shuffle=True)
    val_loader = DataLoader([val_graph], batch_size=1, shuffle=False)
    
    # Initialize model
    input_dim = train_graph.x.size(1)
    model = LogGraphSSL(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        encoder_type=args.encoder_type
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize SSL tasks
    ssl_tasks = CombinedSSLTasks([
        MaskedNodePrediction(mask_rate=0.15),
        EdgePrediction(neg_sampling_ratio=1.0),
        NodeClassification(num_classes=3),
        # GraphContrastiveLearning(temperature=0.07)  # Uncomment if needed
    ], task_weights=[1.0, 0.5, 0.3])
    
    # Initialize trainer
    trainer = SSLTrainer(
        model=model,
        ssl_tasks=ssl_tasks,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=output_dir
    )
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.pth')
    graph_builder.save_vocabulary(vocab_path)
    
    print(f"Training completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
