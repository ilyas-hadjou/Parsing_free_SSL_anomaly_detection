"""
Fixed training script for LogGraph-SSL with anti-collapse mechanisms.
Addresses representation collapse and overfitting issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import numpy as np
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from ssl_tasks import CombinedSSLTasks
from utils import load_log_data, set_seed


def compute_anti_collapse_loss(model, embeddings, alpha=0.1, beta=0.1):
    """
    Compute anti-collapse regularization loss.
    
    Args:
        model: LogGraphSSL model
        embeddings: Node embeddings
        alpha: Weight for diversity loss
        beta: Weight for variance loss
    
    Returns:
        Anti-collapse loss
    """
    diversity_loss = model.diversity_loss(embeddings)
    variance_loss = model.embedding_variance_loss(embeddings)
    
    total_loss = alpha * diversity_loss + beta * variance_loss
    return total_loss, diversity_loss, variance_loss


def train_epoch(model, ssl_manager, optimizer, scheduler, device, 
                train_graph, config, epoch):
    """
    Train for one epoch with anti-collapse mechanisms.
    """
    model.train()
    total_loss = 0
    total_ssl_loss = 0
    total_diversity_loss = 0
    total_variance_loss = 0
    
    # Move graph to device
    train_graph = train_graph.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    embeddings = model(train_graph.x, train_graph.edge_index)
    
    # Compute SSL losses
    ssl_losses = ssl_manager.compute_all_losses(
        model, train_graph, embeddings, device
    )
    
    # Main SSL loss
    ssl_loss = (
        config['ssl_weights']['masked_nodes'] * ssl_losses['masked_nodes'] +
        config['ssl_weights']['edge_prediction'] * ssl_losses['edge_prediction'] +
        config['ssl_weights']['node_classification'] * ssl_losses['node_classification']
    )
    
    # Anti-collapse regularization
    anti_collapse_loss, diversity_loss, variance_loss = compute_anti_collapse_loss(
        model, embeddings, 
        alpha=config.get('diversity_weight', 0.1),
        beta=config.get('variance_weight', 0.1)
    )
    
    # Total loss
    total_loss_value = ssl_loss + anti_collapse_loss
    
    # Backward pass
    total_loss_value.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if scheduler:
        scheduler.step()
    
    return {
        'total_loss': total_loss_value.item(),
        'ssl_loss': ssl_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'variance_loss': variance_loss.item(),
        'individual_ssl_losses': {k: v.item() for k, v in ssl_losses.items()}
    }


def validate_epoch(model, ssl_manager, device, val_graph, config):
    """
    Validate for one epoch.
    """
    model.eval()
    
    with torch.no_grad():
        # Move graph to device
        val_graph = val_graph.to(device)
        
        # Forward pass
        embeddings = model(val_graph.x, val_graph.edge_index)
        
        # Compute SSL losses
        ssl_losses = ssl_manager.compute_all_losses(
            model, val_graph, embeddings, device
        )
        
        # Main SSL loss
        ssl_loss = (
            config['ssl_weights']['masked_nodes'] * ssl_losses['masked_nodes'] +
            config['ssl_weights']['edge_prediction'] * ssl_losses['edge_prediction'] +
            config['ssl_weights']['node_classification'] * ssl_losses['node_classification']
        )
        
        # Anti-collapse regularization
        anti_collapse_loss, diversity_loss, variance_loss = compute_anti_collapse_loss(
            model, embeddings,
            alpha=config.get('diversity_weight', 0.1),
            beta=config.get('variance_weight', 0.1)
        )
        
        # Total loss
        total_loss_value = ssl_loss + anti_collapse_loss
        
        # Compute embedding statistics for monitoring
        embedding_stats = {
            'mean_norm': torch.norm(embeddings, dim=1).mean().item(),
            'variance': torch.var(embeddings, dim=0).mean().item(),
            'cosine_similarity_mean': torch.cosine_similarity(
                embeddings[:-1], embeddings[1:], dim=1
            ).mean().item()
        }
    
    return {
        'total_loss': total_loss_value.item(),
        'ssl_loss': ssl_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'variance_loss': variance_loss.item(),
        'individual_ssl_losses': {k: v.item() for k, v in ssl_losses.items()},
        'embedding_stats': embedding_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Fixed LogGraph-SSL Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--train_data', type=str, required=True, help='Training data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    config['model']['dropout'] = args.dropout
    config['training']['learning_rate'] = args.learning_rate
    config['training']['weight_decay'] = args.weight_decay
    config['training']['epochs'] = args.epochs
    config['training']['early_stopping_patience'] = args.early_stopping_patience
    
    # Add anti-collapse regularization weights
    config['diversity_weight'] = 0.1
    config['variance_weight'] = 0.1
    
    print("Loading and processing data...")
    
    # Load training data
    messages = load_log_data(args.train_data)
    print(f"Loaded {len(messages)} training messages")
    
    # Create graph builder
    graph_builder = LogGraphBuilder(
        vocab_size=config['data']['vocab_size'],
        min_count=config['data']['min_count'],
        window_size=config['data']['window_size'],
        directed=config['data']['directed']
    )
    
    # Build graph from training data
    print("Building graph from training data...")
    train_graph = graph_builder.build_graph_from_logs(messages)
    
    # Split for validation (use last 10% as validation)
    split_idx = int(0.9 * len(messages))
    val_messages = messages[split_idx:]
    val_graph = graph_builder.build_graph_from_logs(val_messages)
    
    print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Validation graph: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    
    # Initialize model with stronger regularization
    model = LogGraphSSL(
        input_dim=train_graph.x.shape[1],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim'],
        encoder_type=config['model']['encoder_type'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],  # Will be 0.5 from args
        activation=config['model']['activation']
    ).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize SSL task manager
    ssl_manager = SSLTaskManager(
        mask_rate=config['ssl']['mask_rate'],
        neg_sampling_ratio=config['ssl']['neg_sampling_ratio']
    )
    
    # Initialize optimizer with strong weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"loggraph_ssl_fixed_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_ssl_loss': [],
        'val_ssl_loss': [],
        'train_diversity_loss': [],
        'val_diversity_loss': [],
        'train_variance_loss': [],
        'val_variance_loss': [],
        'embedding_stats': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print("Starting training...")
    
    for epoch in range(config['training']['epochs']):
        # Training
        train_metrics = train_epoch(
            model, ssl_manager, optimizer, scheduler, device,
            train_graph, config, epoch
        )
        
        # Validation
        val_metrics = validate_epoch(
            model, ssl_manager, device, val_graph, config
        )
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_ssl_loss'].append(train_metrics['ssl_loss'])
        history['val_ssl_loss'].append(val_metrics['ssl_loss'])
        history['train_diversity_loss'].append(train_metrics['diversity_loss'])
        history['val_diversity_loss'].append(val_metrics['diversity_loss'])
        history['train_variance_loss'].append(train_metrics['variance_loss'])
        history['val_variance_loss'].append(val_metrics['variance_loss'])
        history['embedding_stats'].append(val_metrics['embedding_stats'])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
              f"(SSL: {train_metrics['ssl_loss']:.4f}, "
              f"Div: {train_metrics['diversity_loss']:.4f}, "
              f"Var: {train_metrics['variance_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['total_loss']:.4f} "
              f"(SSL: {val_metrics['ssl_loss']:.4f}, "
              f"Div: {val_metrics['diversity_loss']:.4f}, "
              f"Var: {val_metrics['variance_loss']:.4f})")
        print(f"  Embedding Stats: Norm={val_metrics['embedding_stats']['mean_norm']:.4f}, "
              f"Var={val_metrics['embedding_stats']['variance']:.6f}, "
              f"CosSim={val_metrics['embedding_stats']['cosine_similarity_mean']:.4f}")
        
        # Early stopping check
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                os.path.join(output_dir, 'best_model.pth')
            )
            
            # Save vocabulary
            torch.save(graph_builder.vocab, os.path.join(output_dir, 'vocabulary.pth'))
            
        else:
            patience_counter += 1
            
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Save final training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['train_ssl_loss'], label='Train SSL')
    plt.plot(history['val_ssl_loss'], label='Val SSL')
    plt.title('SSL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['train_diversity_loss'], label='Train Diversity')
    plt.plot(history['val_diversity_loss'], label='Val Diversity')
    plt.title('Diversity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(history['train_variance_loss'], label='Train Variance')
    plt.plot(history['val_variance_loss'], label='Val Variance')
    plt.title('Variance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    embedding_norms = [stats['mean_norm'] for stats in history['embedding_stats']]
    plt.plot(embedding_norms)
    plt.title('Embedding Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Norm')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    embedding_vars = [stats['variance'] for stats in history['embedding_stats']]
    plt.plot(embedding_vars)
    plt.title('Embedding Variance')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
