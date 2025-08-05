"""
LogGraph-SSL: A parsing-free anomaly detection framework for distributed system logs
using Graph Neural Networks and Self-Supervised Learning.

This framework builds token co-occurrence graphs from raw log messages and uses
self-supervised learning tasks to train GNN models for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm


class LogGraphBuilder:
    """
    Builds token co-occurrence graphs from raw log messages without parsing.
    Uses sliding window approach to capture token relationships.
    """
    
    def __init__(self, 
                 window_size: int = 5,
                 min_token_freq: int = 2,
                 max_vocab_size: int = 10000,
                 token_pattern: str = r'\b\w+\b'):
        """
        Initialize the log graph builder.
        
        Args:
            window_size: Size of sliding window for token co-occurrence
            min_token_freq: Minimum frequency for tokens to be included
            max_vocab_size: Maximum vocabulary size
            token_pattern: Regular expression pattern for tokenization
        """
        self.window_size = window_size
        self.min_token_freq = min_token_freq
        self.max_vocab_size = max_vocab_size
        self.token_pattern = token_pattern
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_freq = Counter()
        self.vocab_size = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a log message using regex pattern."""
        tokens = re.findall(self.token_pattern, text.lower())
        return tokens
    
    def build_vocabulary(self, log_messages: List[str]) -> None:
        """Build vocabulary from log messages."""
        print("Building vocabulary...")
        
        # Count token frequencies
        for message in tqdm(log_messages, desc="Counting tokens"):
            tokens = self.tokenize(message)
            self.token_freq.update(tokens)
        
        # Filter tokens by frequency and limit vocabulary size
        filtered_tokens = {token: freq for token, freq in self.token_freq.items() 
                          if freq >= self.min_token_freq}
        
        # Sort by frequency and take top tokens
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:self.max_vocab_size]
        
        # Create token mappings
        self.token_to_id = {token: idx for idx, (token, _) in enumerate(top_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def build_cooccurrence_matrix(self, log_messages: List[str]) -> np.ndarray:
        """Build token co-occurrence matrix from log messages."""
        print("Building co-occurrence matrix...")
        
        cooccurrence = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        
        for message in tqdm(log_messages, desc="Processing messages"):
            tokens = self.tokenize(message)
            token_ids = [self.token_to_id[token] for token in tokens 
                        if token in self.token_to_id]
            
            # Sliding window for co-occurrence
            for i, center_token in enumerate(token_ids):
                start = max(0, i - self.window_size // 2)
                end = min(len(token_ids), i + self.window_size // 2 + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_token = token_ids[j]
                        cooccurrence[center_token, context_token] += 1.0
        
        return cooccurrence
    
    def build_graph_from_logs(self, log_messages: List[str], 
                             edge_threshold: float = 1.0) -> Data:
        """
        Build PyTorch Geometric graph from log messages.
        
        Args:
            log_messages: List of raw log messages
            edge_threshold: Minimum co-occurrence count for edges
            
        Returns:
            PyTorch Geometric Data object
        """
        # Build vocabulary if not already built
        if not self.token_to_id:
            self.build_vocabulary(log_messages)
        
        # Build co-occurrence matrix
        cooccurrence = self.build_cooccurrence_matrix(log_messages)
        
        # Create edges from co-occurrence matrix
        edge_indices = []
        edge_weights = []
        
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if cooccurrence[i, j] >= edge_threshold:
                    edge_indices.append([i, j])
                    edge_weights.append(cooccurrence[i, j])
        
        if not edge_indices:
            # Create self-loops if no edges found
            edge_indices = [[i, i] for i in range(self.vocab_size)]
            edge_weights = [1.0] * self.vocab_size
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Create node features (can be enhanced with embeddings)
        # For now, use one-hot encoding or frequency-based features
        node_features = torch.eye(self.vocab_size, dtype=torch.float)
        
        # Add frequency-based features
        freq_features = torch.zeros(self.vocab_size, 1)
        for token, token_id in self.token_to_id.items():
            freq_features[token_id, 0] = np.log(self.token_freq[token] + 1)
        
        x = torch.cat([node_features, freq_features], dim=1)
        
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        print(f"Graph created with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
        
        return graph_data
    
    def build_sequence_graph(self, log_sequence: List[str], 
                           sequence_labels: Optional[List[int]] = None) -> Data:
        """
        Build graph for a sequence of log messages (e.g., for anomaly detection).
        Each message becomes a node, with edges based on temporal proximity and token similarity.
        
        Args:
            log_sequence: Sequential log messages
            sequence_labels: Optional labels for each message (0=normal, 1=anomaly)
            
        Returns:
            PyTorch Geometric Data object
        """
        if not self.token_to_id:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        num_messages = len(log_sequence)
        
        # Create message embeddings by averaging token embeddings
        message_embeddings = []
        
        for message in log_sequence:
            tokens = self.tokenize(message)
            token_ids = [self.token_to_id[token] for token in tokens 
                        if token in self.token_to_id]
            
            if token_ids:
                # Simple bag-of-words representation with vocab_size + 1 dimensions (matching build_graph_from_logs)
                embedding = torch.zeros(self.vocab_size + 1)
                for token_id in token_ids:
                    embedding[token_id] += 1.0
                # Normalize
                embedding[:self.vocab_size] = embedding[:self.vocab_size] / len(token_ids)
                # Add frequency feature (mean log frequency of tokens in message)
                avg_freq = sum(np.log(self.token_freq[self.id_to_token[tid]] + 1) for tid in token_ids) / len(token_ids)
                embedding[self.vocab_size] = avg_freq
            else:
                embedding = torch.zeros(self.vocab_size + 1)
            
            message_embeddings.append(embedding)
        
        x = torch.stack(message_embeddings)
        
        # Create edges based on temporal proximity
        edge_indices = []
        temporal_window = 3  # Connect messages within window
        
        for i in range(num_messages):
            for j in range(max(0, i - temporal_window), 
                          min(num_messages, i + temporal_window + 1)):
                if i != j:
                    edge_indices.append([i, j])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index)
        
        if sequence_labels is not None:
            graph_data.y = torch.tensor(sequence_labels, dtype=torch.long)
        
        return graph_data
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'token_freq': dict(self.token_freq),
            'vocab_size': self.vocab_size
        }
        torch.save(vocab_data, filepath)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary from file."""
        vocab_data = torch.load(filepath)
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = vocab_data['id_to_token']
        self.token_freq = Counter(vocab_data['token_freq'])
        self.vocab_size = vocab_data['vocab_size']
        print(f"Vocabulary loaded from {filepath}")


def create_sample_logs() -> List[str]:
    """Create sample log messages for testing."""
    sample_logs = [
        "2024-01-01 10:00:01 INFO [main] Starting application server on port 8080",
        "2024-01-01 10:00:02 INFO [main] Database connection established successfully",
        "2024-01-01 10:00:03 INFO [worker-1] Processing user request id=12345",
        "2024-01-01 10:00:04 INFO [worker-2] Processing user request id=12346",
        "2024-01-01 10:00:05 WARN [worker-1] Slow query detected, execution time 2.5s",
        "2024-01-01 10:00:06 ERROR [worker-3] Database connection timeout after 30s",
        "2024-01-01 10:00:07 INFO [main] Health check passed, all services running",
        "2024-01-01 10:00:08 ERROR [worker-2] Out of memory error in request processing",
        "2024-01-01 10:00:09 INFO [worker-1] Request completed successfully id=12345",
        "2024-01-01 10:00:10 FATAL [main] Critical system failure, shutting down",
    ]
    return sample_logs


if __name__ == "__main__":
    # Example usage
    builder = LogGraphBuilder(window_size=5, min_token_freq=1, max_vocab_size=1000)
    
    # Create sample logs
    sample_logs = create_sample_logs()
    
    # Build graph from logs
    graph = builder.build_graph_from_logs(sample_logs)
    print(f"Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
    print(f"Node feature dimension: {graph.x.shape[1]}")
    
    # Build sequence graph
    sequence_graph = builder.build_sequence_graph(sample_logs)
    print(f"Created sequence graph with {sequence_graph.num_nodes} nodes and {sequence_graph.num_edges} edges")
