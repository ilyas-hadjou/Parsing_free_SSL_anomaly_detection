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
from typing import Optional, List, Union, Tuple


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
    Graph Attention Network encoder with anti-collapse mechanisms.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.3,
                 activation: str = 'relu'):
        """
        Initialize GAT encoder with regularization to prevent collapse.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate (increased default)
            activation: Activation function
        """
        super(GATEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Build layers with batch normalization and residual connections
        layers = []
        batch_norms = []
        dims = [input_dim] + hidden_dims
        
        # Hidden layers with multiple heads - divide dims by num_heads to compensate
        for i in range(len(dims) - 1):
            # Ensure the actual output dimension matches what we expect
            head_dim = max(1, dims[i + 1] // num_heads)
            layers.append(GATConv(dims[i], head_dim, heads=num_heads, dropout=dropout, concat=True))
            # Add batch normalization to prevent collapse
            batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
        
        # Output layer with single head to get exact output_dim
        if len(dims) > 1:
            final_input_dim = dims[-1]  # This will be head_dim * num_heads from previous layer
        else:
            final_input_dim = input_dim
        layers.append(GATConv(final_input_dim, output_dim, heads=1, dropout=dropout, concat=False))
        
        self.convs = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(layers))])
        
        # Layer normalization for output
        self.output_norm = nn.LayerNorm(output_dim)
        
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
        """Forward pass through GAT layers with residual connections and normalization."""
        residual = x
        
        for i, conv in enumerate(self.convs[:-1]):  # All hidden layers
            out = conv(x, edge_index)
            
            # Apply batch normalization
            if len(out.shape) == 2 and out.shape[0] > 1:  # Only if batch size > 1
                out = self.batch_norms[i](out)
            
            # Apply activation and dropout
            out = self.activation(out)
            out = self.dropout_layers[i](out)
            
            # Add residual connection if dimensions match
            if x.shape[-1] == out.shape[-1]:
                out = out + x
            
            x = out
        
        # Final layer without residual
        x = self.convs[-1](x, edge_index)
        
        # Apply output normalization to prevent collapse
        x = self.output_norm(x)
        
        # Add noise during training to prevent collapse
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
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
    Main LogGraph-SSL model with anti-collapse mechanisms.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 output_dim: int = 32,
                 encoder_type: str = 'gcn',
                 num_heads: int = 8,
                 dropout: float = 0.3,
                 activation: str = 'relu'):
        """
        Initialize LogGraph-SSL model with regularization.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            encoder_type: Type of GNN encoder ('gcn', 'gat', 'sage')
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout rate (increased default)
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
        
        # SSL task heads with better architecture
        self.masked_node_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, input_dim)
        )
        
        self.edge_pred_head = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim * 2),  # Increased input size for enhanced features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1)
        )
        
        self.node_class_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 3)
        )
        
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
        
        # Debug print for dimension tracking
        # print(f"Encoder output shape: {node_embeddings.shape}, expected output_dim: {self.output_dim}")
        
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
        """Forward pass for edge prediction with improved architecture."""
        node_embeddings = self.forward(x, edge_index)
        
        # Get embeddings for edge endpoints
        src_embeddings = node_embeddings[edge_label_index[0]]
        dst_embeddings = node_embeddings[edge_label_index[1]]
        
        # Enhanced edge representation with multiple interaction types
        edge_embeddings = torch.cat([
            src_embeddings, 
            dst_embeddings,
            src_embeddings * dst_embeddings,  # Element-wise product
            torch.abs(src_embeddings - dst_embeddings)  # Absolute difference
        ], dim=1)
        
        # Predict edge existence
        edge_logits = self.edge_pred_head(edge_embeddings)
        return edge_logits.squeeze(-1)
    
    def forward_edge_prediction_with_hard_negatives(self, x: torch.Tensor, edge_index: torch.Tensor,
                                                  pos_edge_index: torch.Tensor, 
                                                  neg_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for edge prediction with explicit positive and negative samples."""
        node_embeddings = self.forward(x, edge_index)
        
        # Positive edges
        pos_src = node_embeddings[pos_edge_index[0]]
        pos_dst = node_embeddings[pos_edge_index[1]]
        pos_edge_embeddings = torch.cat([
            pos_src, pos_dst,
            pos_src * pos_dst,
            torch.abs(pos_src - pos_dst)
        ], dim=1)
        pos_logits = self.edge_pred_head(pos_edge_embeddings).squeeze(-1)
        
        # Negative edges
        neg_src = node_embeddings[neg_edge_index[0]]
        neg_dst = node_embeddings[neg_edge_index[1]]
        neg_edge_embeddings = torch.cat([
            neg_src, neg_dst,
            neg_src * neg_dst,
            torch.abs(neg_src - neg_dst)
        ], dim=1)
        neg_logits = self.edge_pred_head(neg_edge_embeddings).squeeze(-1)
        
        return pos_logits, neg_logits
    
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
    
    def contrastive_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                        temperature: float = 0.07) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [batch_size, embed_dim]
            embeddings2: Second set of embeddings [batch_size, embed_dim] 
            temperature: Temperature parameter for softmax
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings1, embeddings2.t()) / temperature
        
        # Create labels (positive pairs are on the diagonal)
        batch_size = embeddings1.size(0)
        labels = torch.arange(batch_size, device=embeddings1.device)
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def graph_augment(self, x: torch.Tensor, edge_index: torch.Tensor, 
                     aug_type: str = 'dropout') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply graph augmentation for contrastive learning.
        
        Args:
            x: Node features
            edge_index: Edge indices
            aug_type: Type of augmentation ('dropout', 'mask', 'noise')
            
        Returns:
            Augmented node features and edge indices
        """
        if aug_type == 'dropout':
            # Random edge dropout
            num_edges = edge_index.size(1)
            keep_prob = 0.8
            mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
            aug_edge_index = edge_index[:, mask]
            aug_x = x
            
        elif aug_type == 'mask':
            # Random feature masking
            mask_prob = 0.2
            mask = torch.rand_like(x) > mask_prob
            aug_x = x * mask.float()
            aug_edge_index = edge_index
            
        elif aug_type == 'noise':
            # Add Gaussian noise to features
            noise_std = 0.1
            noise = torch.randn_like(x) * noise_std
            aug_x = x + noise
            aug_edge_index = edge_index
            
        else:
            aug_x, aug_edge_index = x, edge_index
            
        return aug_x, aug_edge_index
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings for downstream tasks."""
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings
    
    def diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss to prevent representation collapse.
        Encourages embeddings to be different from each other.
        """
        # Normalize embeddings
        normalized = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        # Remove diagonal (self-similarity)
        n = similarity_matrix.size(0)
        mask = torch.eye(n, device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)
        
        # Penalize high similarities (encourage diversity)
        diversity_loss = similarity_matrix.abs().mean()
        
        return diversity_loss
    
    def embedding_variance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding variance loss to prevent collapse.
        Encourages high variance in embedding dimensions.
        """
        # Compute variance across batch for each dimension
        variances = torch.var(embeddings, dim=0)
        
        # Penalize low variance (encourage high variance)
        variance_loss = -variances.mean()
        
        return variance_loss


class AnomalyDetectionHead(nn.Module):
    """
    Enhanced anomaly detection head with better architecture and threshold learning.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Initialize enhanced anomaly detection head.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(AnomalyDetectionHead, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Second layer
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Third layer
        layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 4))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim // 4, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced anomaly detection head."""
        return self.layers(x)
    
    def predict_with_threshold(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomalies using learnable threshold.
        
        Returns:
            scores: Anomaly scores
            predictions: Binary predictions
        """
        scores = torch.sigmoid(self.forward(x))
        predictions = (scores > self.threshold).float()
        return scores, predictions


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
