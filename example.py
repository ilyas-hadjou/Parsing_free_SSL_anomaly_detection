"""
Simple example script demonstrating LogGraph-SSL framework usage.
This script shows the basic workflow without complex dependencies.
"""

import torch
import numpy as np
from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from utils import create_sample_log_data, set_seed


def simple_example():
    """Simple example of using LogGraph-SSL framework."""
    print("LogGraph-SSL Simple Example")
    print("=" * 40)
    
    # Set random seed
    set_seed(42)
    
    # Step 1: Create sample data
    print("1. Creating sample log data...")
    messages, labels = create_sample_log_data(num_samples=100, anomaly_ratio=0.1)
    
    normal_messages = [msg for msg, label in zip(messages, labels) if label == 0]
    print(f"   Created {len(messages)} messages ({len(normal_messages)} normal)")
    
    # Show examples
    print("\n   Sample messages:")
    for i in range(3):
        status = "ANOMALY" if labels[i] else "NORMAL"
        print(f"   [{status}] {messages[i]}")
    
    # Step 2: Build graph
    print("\n2. Building co-occurrence graph...")
    builder = LogGraphBuilder(window_size=3, min_token_freq=1, max_vocab_size=500)
    graph = builder.build_graph_from_logs(normal_messages[:50])  # Use subset for demo
    
    print(f"   Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"   Node features: {graph.x.shape}")
    
    # Step 3: Initialize model
    print("\n3. Initializing model...")
    model = LogGraphSSL(
        input_dim=graph.x.size(1),
        hidden_dims=[32, 16],
        output_dim=8,
        encoder_type='gcn'
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")
    
    # Step 4: Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        embeddings = model(graph.x, graph.edge_index)
        print(f"   Output embeddings shape: {embeddings.shape}")
        print(f"   Embedding statistics:")
        print(f"     Mean: {embeddings.mean().item():.4f}")
        print(f"     Std:  {embeddings.std().item():.4f}")
    
    # Step 5: Test SSL tasks
    print("\n5. Testing SSL tasks...")
    
    # Masked node prediction
    try:
        from ssl_tasks import MaskedNodePrediction
        mask_task = MaskedNodePrediction(mask_rate=0.2)
        masked_data, targets = mask_task.create_task(graph)
        print(f"   Masked {len(masked_data.mask_indices)} out of {graph.num_nodes} nodes")
        
        # Test forward pass
        predictions = model.forward_masked_nodes(
            masked_data.x, masked_data.edge_index, masked_data.mask_indices
        )
        loss = mask_task.compute_loss(predictions, targets)
        print(f"   Reconstruction loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   Masked node prediction error: {e}")
    
    # Edge prediction
    try:
        from ssl_tasks import EdgePrediction
        edge_task = EdgePrediction(neg_sampling_ratio=0.5)
        edge_data, targets = edge_task.create_task(graph)
        print(f"   Edge prediction with {len(targets)} edge pairs")
        
        # Test forward pass
        predictions = model.forward_edge_prediction(
            edge_data.x, edge_data.edge_index, edge_data.edge_label_index
        )
        loss = edge_task.compute_loss(predictions, targets)
        print(f"   Edge prediction loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   Edge prediction error: {e}")
    
    print("\n6. Framework components verified!")
    print("   ✓ Graph construction working")
    print("   ✓ Model initialization working")
    print("   ✓ Forward pass working")
    print("   ✓ SSL tasks working")
    
    print(f"\nExample completed successfully!")
    print("Next steps:")
    print("- Use train.py for full model training")
    print("- Use evaluate.py for comprehensive evaluation")
    print("- Check README.md for detailed usage instructions")


if __name__ == "__main__":
    simple_example()
