"""
Demo script for LogGraph-SSL framework.
Demonstrates the complete workflow from data creation to anomaly detection.
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import framework components
from log_graph_builder import LogGraphBuilder
from ssl_tasks import MaskedNodePrediction, EdgePrediction, NodeClassification, CombinedSSLTasks
from gnn_model import LogGraphSSL
from anomaly_detector import AnomalyDetector
from utils import create_sample_log_data, set_seed, create_log_dataset_splits


def run_demo():
    """Run a complete demo of the LogGraph-SSL framework."""
    print("=" * 60)
    print("LogGraph-SSL Framework Demo")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Step 1: Create sample data
    print("\n1. Creating sample log data...")
    log_messages, labels = create_sample_log_data(
        num_samples=500,
        anomaly_ratio=0.15,
        random_seed=42
    )
    
    print(f"Created {len(log_messages)} log messages")
    print(f"Normal messages: {labels.count(0)}")
    print(f"Anomalous messages: {labels.count(1)}")
    
    # Show some examples
    print("\nSample log messages:")
    for i in range(3):
        label_str = "ANOMALY" if labels[i] == 1 else "NORMAL"
        print(f"  [{label_str}] {log_messages[i]}")
    
    # Step 2: Split data
    print("\n2. Splitting data into train/test sets...")
    splits = create_log_dataset_splits(
        log_messages, labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    train_messages, train_labels = splits['train']
    val_messages, val_labels = splits['val']
    test_messages, test_labels = splits['test']
    
    print(f"Training set: {len(train_messages)} messages")
    print(f"Validation set: {len(val_messages)} messages")
    print(f"Test set: {len(test_messages)} messages")
    
    # Step 3: Build graphs
    print("\n3. Building token co-occurrence graphs...")
    graph_builder = LogGraphBuilder(
        window_size=5,
        min_token_freq=2,
        max_vocab_size=1000
    )
    
    # Build training graph
    train_graph = graph_builder.build_graph_from_logs(train_messages)
    print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Node feature dimension: {train_graph.x.shape[1]}")
    
    # Step 4: Initialize model
    print("\n4. Initializing LogGraph-SSL model...")
    model = LogGraphSSL(
        input_dim=train_graph.x.size(1),
        hidden_dims=[64, 32],
        output_dim=16,
        encoder_type='gcn',
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    
    # Step 5: Setup SSL tasks
    print("\n5. Setting up self-supervised learning tasks...")
    ssl_tasks = CombinedSSLTasks([
        MaskedNodePrediction(mask_rate=0.15),
        EdgePrediction(neg_sampling_ratio=1.0),
        NodeClassification(num_classes=3)
    ], task_weights=[1.0, 0.5, 0.3])
    
    print(f"Configured {len(ssl_tasks.tasks)} SSL tasks")
    
    # Step 6: Simulate training (simplified)
    print("\n6. Simulating model training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    train_graph = train_graph.to(device)
    
    # Simple training simulation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print("Training for 10 epochs (demo)...")
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(train_graph.x, train_graph.edge_index)
        
        # Simulate SSL loss (simplified)
        loss = torch.mean(embeddings ** 2) * 0.1  # Dummy loss for demo
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 3 == 0:
            print(f"  Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    
    # Step 7: Extract embeddings
    print("\n7. Extracting node embeddings...")
    model.eval()
    with torch.no_grad():
        node_embeddings = model.get_embeddings(train_graph.x, train_graph.edge_index)
        node_embeddings = node_embeddings.cpu()
    
    print(f"Extracted embeddings shape: {node_embeddings.shape}")
    
    # Step 8: Anomaly detection
    print("\n8. Setting up anomaly detection...")
    
    # Prepare normal data (use only normal messages from training)
    normal_indices = [i for i, label in enumerate(train_labels) if label == 0]
    normal_messages = [train_messages[i] for i in normal_indices[:50]]  # Limit for demo
    
    # Create test sequences
    test_sequences = []
    test_seq_labels = []
    
    sequence_length = 5
    for i in range(0, len(test_messages) - sequence_length + 1, sequence_length):
        seq = test_messages[i:i + sequence_length]
        seq_labels = test_labels[i:i + sequence_length]
        
        # Sequence is anomalous if any message is anomalous
        seq_label = 1 if any(seq_labels) else 0
        
        test_sequences.append(seq)
        test_seq_labels.append(seq_label)
    
    print(f"Created {len(test_sequences)} test sequences")
    print(f"Normal sequences: {test_seq_labels.count(0)}")
    print(f"Anomalous sequences: {test_seq_labels.count(1)}")
    
    # Build graphs for sequences
    normal_graphs = []
    for seq in normal_messages[:10]:  # Limit for demo
        graph = graph_builder.build_sequence_graph([seq])
        normal_graphs.append(graph)
    
    test_graphs = []
    for seq in test_sequences[:20]:  # Limit for demo
        graph = graph_builder.build_sequence_graph(seq)
        test_graphs.append(graph)
    
    # Test different detection methods
    detection_methods = ['neural', 'statistical']  # Simplified for demo
    results = {}
    
    for method in detection_methods:
        print(f"\n  Testing {method} detection...")
        try:
            detector = AnomalyDetector(
                pretrained_model=model,
                detection_method=method,
                contamination=0.2
            )
            
            # Fit on normal data
            detector.fit(normal_graphs)
            
            # Predict on test data
            predictions, scores = detector.predict(test_graphs)
            
            # Calculate accuracy
            true_labels = test_seq_labels[:len(predictions)]
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
            
            results[method] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'scores': scores
            }
            
            print(f"    Accuracy: {accuracy:.3f}")
            print(f"    Detected {sum(predictions)} anomalies out of {len(predictions)} sequences")
            
        except Exception as e:
            print(f"    Error with {method}: {str(e)}")
            results[method] = {'error': str(e)}
    
    # Step 9: Visualization
    print("\n9. Creating visualizations...")
    create_demo_visualizations(results, node_embeddings, test_seq_labels)
    
    # Step 10: Summary
    print("\n" + "=" * 60)
    print("Demo Summary")
    print("=" * 60)
    print(f"✓ Created {len(log_messages)} synthetic log messages")
    print(f"✓ Built graph with {train_graph.num_nodes} nodes and {train_graph.num_edges} edges")
    print(f"✓ Trained GNN model with {total_params:,} parameters")
    print(f"✓ Extracted {node_embeddings.shape[0]} node embeddings")
    print(f"✓ Tested {len(detection_methods)} anomaly detection methods")
    
    print("\nDetection Results:")
    for method, result in results.items():
        if 'error' not in result:
            print(f"  {method}: {result['accuracy']:.3f} accuracy")
        else:
            print(f"  {method}: Error occurred")
    
    print("\nDemo completed successfully!")
    print("Check 'demo_results.png' for visualizations.")


def create_demo_visualizations(results, embeddings, labels):
    """Create visualization plots for the demo."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Embedding distribution
        axes[0, 0].hist(embeddings.flatten().numpy(), bins=50, alpha=0.7)
        axes[0, 0].set_title('Node Embedding Distribution')
        axes[0, 0].set_xlabel('Embedding Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Embedding variance per dimension
        embedding_var = torch.var(embeddings, dim=0).numpy()
        axes[0, 1].bar(range(len(embedding_var)), embedding_var)
        axes[0, 1].set_title('Embedding Variance per Dimension')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Variance')
        
        # Plot 3: Detection method comparison
        methods = []
        accuracies = []
        for method, result in results.items():
            if 'error' not in result:
                methods.append(method)
                accuracies.append(result['accuracy'])
        
        if methods:
            axes[1, 0].bar(methods, accuracies)
            axes[1, 0].set_title('Detection Method Accuracy')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Label distribution
        normal_count = labels.count(0) if isinstance(labels, list) else sum(labels == 0)
        anomaly_count = labels.count(1) if isinstance(labels, list) else sum(labels == 1)
        
        axes[1, 1].pie([normal_count, anomaly_count], 
                      labels=['Normal', 'Anomaly'], 
                      autopct='%1.1f%%',
                      colors=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('Test Data Distribution')
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to 'demo_results.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")


if __name__ == "__main__":
    run_demo()
