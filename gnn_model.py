"""
Graph Neural Network models for LogGraph-SSL framework.
Implements GCN, GAT, and GraphSAGE architectures with SSL support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import math
from typing import Optional, List, Union


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize GCN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(GCNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(GCNConv(dims[i], dims[i + 1]))
        
        self.convs = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        return x


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize GAT encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
        """
        super(GATEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Hidden layers with multiple heads
        for i in range(len(dims) - 1):
            layers.append(GATConv(dims[i], dims[i + 1], heads=num_heads, dropout=dropout))
        
        # Output layer with single head
        layers.append(GATConv(dims[-1] * num_heads, output_dim, heads=1, dropout=dropout))
        
        self.convs = nn.ModuleList(layers)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply activation (except for last layer)
            if i < len(self.convs) - 1:
                x = self.activation(x)
        
        return x


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize GraphSAGE encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super(GraphSAGEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(SAGEConv(dims[i], dims[i + 1]))
        
        self.convs = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GraphSAGE layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        return x


class LogGraphSSL(nn.Module):
    """
    Main LogGraph-SSL model that combines GNN encoder with SSL task heads.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 output_dim: int = 32,
                 encoder_type: str = 'gcn',
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize LogGraph-SSL model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            encoder_type: Type of GNN encoder ('gcn', 'gat', 'sage')
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            activation: Activation function
        """
        super(LogGraphSSL, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        
        # Initialize encoder
        if encoder_type == 'gcn':
            self.encoder = GCNEncoder(input_dim, hidden_dims, output_dim, dropout, activation)
        elif encoder_type == 'gat':
            self.encoder = GATEncoder(input_dim, hidden_dims, output_dim, num_heads, dropout, activation)
        elif encoder_type == 'sage':
            self.encoder = GraphSAGEEncoder(input_dim, hidden_dims, output_dim, dropout, activation)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # SSL task heads
        self.masked_node_head = nn.Linear(output_dim, input_dim)  # Reconstruct original features
        self.edge_pred_head = nn.Linear(output_dim * 2, 1)  # Binary edge prediction
        self.node_class_head = nn.Linear(output_dim, 3)  # Node classification (3 classes)
        
        # Graph-level pooling for contrastive learning
        self.graph_pooling = global_mean_pool
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices for graph-level tasks
            
        Returns:
            Node embeddings
        """
        # Get node embeddings from encoder
        node_embeddings = self.encoder(x, edge_index)
        return node_embeddings
    
    def forward_masked_nodes(self, x: torch.Tensor, edge_index: torch.Tensor,
                           mask_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass for masked node prediction."""
        node_embeddings = self.forward(x, edge_index)
        masked_embeddings = node_embeddings[mask_indices]
        reconstructed = self.masked_node_head(masked_embeddings)
        return reconstructed
    
    def forward_edge_prediction(self, x: torch.Tensor, edge_index: torch.Tensor,
                              edge_label_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for edge prediction."""
        node_embeddings = self.forward(x, edge_index)
        
        # Get embeddings for edge endpoints
        src_embeddings = node_embeddings[edge_label_index[0]]
        dst_embeddings = node_embeddings[edge_label_index[1]]
        
        # Concatenate embeddings
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Predict edge existence
        edge_logits = self.edge_pred_head(edge_embeddings)
        return edge_logits.squeeze(-1)
    
    def forward_node_classification(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for node classification."""
        node_embeddings = self.forward(x, edge_index)
        node_logits = self.node_class_head(node_embeddings)
        return node_logits
    
    def forward_contrastive(self, x: torch.Tensor, edge_index: torch.Tensor,
                          batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for contrastive learning."""
        node_embeddings = self.forward(x, edge_index, batch)
        
        # Pool node embeddings to graph-level
        graph_embeddings = self.graph_pooling(node_embeddings, batch)
        
        return graph_embeddings
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings for downstream tasks."""
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings


class AnomalyDetectionHead(nn.Module):
    """
    Anomaly detection head for downstream tasks.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 dropout: float = 0.1):
        """
        Initialize anomaly detection head.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout rate
        """
        super(AnomalyDetectionHead, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through anomaly detection head."""
        return self.layers(x)


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data
    num_nodes = 100
    input_dim = 64
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    print("Testing GNN Models:")
    
    # Test GCN
    model_gcn = LogGraphSSL(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=32,
        encoder_type='gcn'
    )
    
    embeddings = model_gcn(x, edge_index)
    print(f"GCN embeddings shape: {embeddings.shape}")
    
    # Test GAT
    model_gat = LogGraphSSL(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=32,
        encoder_type='gat',
        num_heads=4
    )
    
    embeddings_gat = model_gat(x, edge_index)
    print(f"GAT embeddings shape: {embeddings_gat.shape}")
    
    # Test GraphSAGE
    model_sage = LogGraphSSL(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=32,
        encoder_type='sage'
    )
    
    embeddings_sage = model_sage(x, edge_index)
    print(f"GraphSAGE embeddings shape: {embeddings_sage.shape}")
    
    # Test SSL task heads
    mask_indices = torch.randint(0, num_nodes, (10,))
    reconstructed = model_gcn.forward_masked_nodes(x, edge_index, mask_indices)
    print(f"Reconstructed features shape: {reconstructed.shape}")
    
    edge_label_index = torch.randint(0, num_nodes, (2, 20))
    edge_logits = model_gcn.forward_edge_prediction(x, edge_index, edge_label_index)
    print(f"Edge prediction logits shape: {edge_logits.shape}")
    
    node_logits = model_gcn.forward_node_classification(x, edge_index)
    print(f"Node classification logits shape: {node_logits.shape}")
    
    # Test anomaly detection head
    anomaly_head = AnomalyDetectionHead(input_dim=32, hidden_dim=64)
    anomaly_scores = anomaly_head(embeddings)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    
    print("All models initialized and tested successfully!")
