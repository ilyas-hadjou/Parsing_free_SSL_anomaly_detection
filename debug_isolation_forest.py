#!/usr/bin/env python3
"""
Debug script to understand why Isolation Forest has F1=0
"""

import torch
import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL

def debug_isolation_forest():
    print("🔍 Debugging Isolation Forest predictions...")
    
    # Load vocabulary and model
    builder = LogGraphBuilder()
    builder.load_vocabulary('outputs/loggraph_ssl_20250804_151845/vocabulary.pth')
    
    # Load test data - sample from different parts to get anomalies
    with open('hdfs_test.txt', 'r') as f:
        all_test_messages = [line.strip() for line in f.readlines()]
    
    with open('hdfs_test_labels.txt', 'r') as f:
        all_test_labels = [int(line.strip()) for line in f.readlines()]
    
    # Sample 200 messages with some anomalies
    total_messages = len(all_test_messages)
    sample_indices = list(range(0, 100)) + list(range(total_messages-100, total_messages))  # First 100 + last 100
    
    test_messages = [all_test_messages[i] for i in sample_indices]
    test_labels = [all_test_labels[i] for i in sample_indices]
    
    print(f"Loaded {len(test_messages)} test messages")
    print(f"Anomaly rate: {np.mean(test_labels):.3f}")
    print(f"Normal messages: {test_labels.count(0)}")
    print(f"Anomalous messages: {test_labels.count(1)}")
    
    # Create sequence graphs (simplified version)
    sequence_length = 10
    test_graphs = []
    sequence_labels = []
    
    for i in range(0, len(test_messages) - sequence_length + 1, sequence_length):
        sequence = test_messages[i:i + sequence_length]
        seq_labels = test_labels[i:i + sequence_length]
        
        graph = builder.build_sequence_graph(sequence)
        sequence_label = 1 if any(seq_labels) else 0
        
        test_graphs.append(graph)
        sequence_labels.append(sequence_label)
    
    print(f"Created {len(test_graphs)} test sequences")
    print(f"Sequence anomaly rate: {np.mean(sequence_labels):.3f}")
    
    # Load model and extract embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model config and determine input_dim from vocabulary
    with open('outputs/loggraph_ssl_20250804_151845/config.json', 'r') as f:
        config = json.load(f)
    
    # Input dim is vocab_size + 1 (from log_graph_builder.py)
    input_dim = builder.vocab_size + 1
    
    model = LogGraphSSL(
        input_dim=input_dim,
        output_dim=config['output_dim'],
        encoder_type=config['encoder_type']
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load('outputs/loggraph_ssl_20250804_151845/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            embedding = model.get_embeddings(graph.x, graph.edge_index)
            # Graph-level pooling
            graph_embedding = torch.mean(embedding, dim=0).cpu().numpy()
            embeddings.append(graph_embedding)
    
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    
    # Scale embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    print(f"Scaled embedding stats: mean={embeddings_scaled.mean():.4f}, std={embeddings_scaled.std():.4f}")
    
    # Test different contamination values
    contamination_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for contamination in contamination_values:
        print(f"\n🧪 Testing contamination={contamination}")
        
        # Create Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_forest.fit(embeddings_scaled)
        
        # Get predictions
        predictions = iso_forest.predict(embeddings_scaled)
        scores = iso_forest.score_samples(embeddings_scaled)
        
        # Convert predictions
        binary_predictions = (predictions == -1).astype(int)
        
        print(f"Raw predictions: {np.unique(predictions, return_counts=True)}")
        print(f"Binary predictions: {np.unique(binary_predictions, return_counts=True)}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Score stats: mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if np.sum(binary_predictions) > 0:  # If any anomalies detected
            accuracy = accuracy_score(sequence_labels, binary_predictions)
            precision = precision_score(sequence_labels, binary_predictions, zero_division=0)
            recall = recall_score(sequence_labels, binary_predictions, zero_division=0)
            f1 = f1_score(sequence_labels, binary_predictions, zero_division=0)
            
            print(f"Metrics: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        else:
            print("❌ No anomalies detected!")
    
    # Analyze embedding similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n📊 Embedding Analysis:")
    similarity_matrix = cosine_similarity(embeddings_scaled)
    
    # Get similarities between normal and anomalous sequences
    normal_indices = [i for i, label in enumerate(sequence_labels) if label == 0]
    anomaly_indices = [i for i, label in enumerate(sequence_labels) if label == 1]
    
    if len(normal_indices) > 0 and len(anomaly_indices) > 0:
        # Normal-normal similarities
        normal_similarities = []
        for i in range(len(normal_indices)):
            for j in range(i+1, len(normal_indices)):
                normal_similarities.append(similarity_matrix[normal_indices[i], normal_indices[j]])
        
        # Anomaly-anomaly similarities
        anomaly_similarities = []
        for i in range(len(anomaly_indices)):
            for j in range(i+1, len(anomaly_indices)):
                anomaly_similarities.append(similarity_matrix[anomaly_indices[i], anomaly_indices[j]])
        
        # Normal-anomaly similarities
        cross_similarities = []
        for i in normal_indices:
            for j in anomaly_indices:
                cross_similarities.append(similarity_matrix[i, j])
        
        print(f"Normal-Normal similarity: {np.mean(normal_similarities):.4f} ± {np.std(normal_similarities):.4f}")
        print(f"Anomaly-Anomaly similarity: {np.mean(anomaly_similarities):.4f} ± {np.std(anomaly_similarities):.4f}")
        print(f"Normal-Anomaly similarity: {np.mean(cross_similarities):.4f} ± {np.std(cross_similarities):.4f}")
        
        # Check if anomalies are distinguishable
        normal_mean = np.mean(normal_similarities)
        cross_mean = np.mean(cross_similarities)
        
        if abs(normal_mean - cross_mean) < 0.01:
            print("⚠️  WARNING: Embeddings are too similar! Normal and anomalous sequences have nearly identical representations.")
            print("   This explains why Isolation Forest can't distinguish them.")

if __name__ == "__main__":
    debug_isolation_forest()
