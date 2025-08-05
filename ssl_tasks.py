"""
Self-Supervised Learning tasks for LogGraph-SSL framework.
Implements masked node prediction, edge prediction, and graph contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling, add_self_loops, remove_self_loops
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod


class SSLTask(ABC):
    """Abstract base class for self-supervised learning tasks."""
    
    @abstractmethod
    def create_task(self, data: Data) -> Tuple[Data, torch.Tensor]:
        """
        Create SSL task from input data.
        
        Args:
            data: Input graph data
            
        Returns:
            Tuple of (modified_data, targets)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss."""
        pass


class MaskedNodePrediction(SSLTask):
    """
    Masked node prediction task - predict masked node features.
    Similar to BERT's masked language modeling but for graph nodes.
    """
    
    def __init__(self, mask_rate: float = 0.15, mask_token_id: int = -1):
        """
        Initialize masked node prediction task.
        
        Args:
            mask_rate: Proportion of nodes to mask
            mask_token_id: ID to use for masked tokens
        """
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id
    
    def create_task(self, data: Data) -> Tuple[Data, torch.Tensor]:
        """Create masked node prediction task."""
        num_nodes = data.x.size(0)
        num_mask = int(num_nodes * self.mask_rate)
        
        # Randomly select nodes to mask
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        
        # Store original features as targets
        targets = data.x[mask_indices].clone()
        
        # Create modified data with masked nodes
        masked_data = data.clone()
        
        # Mask selected nodes - zero out features or use special mask token
        masked_data.x[mask_indices] = 0  # or use mask_token_id
        
        # Store mask information
        masked_data.mask_indices = mask_indices
        
        return masked_data, targets
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss for node feature prediction."""
        return F.mse_loss(predictions, targets)


class EdgePrediction(SSLTask):
    """
    Edge prediction task - predict whether edges exist between node pairs.
    Uses positive edges from the graph and negative sampling for negative edges.
    """
    
    def __init__(self, neg_sampling_ratio: float = 1.0):
        """
        Initialize edge prediction task.
        
        Args:
            neg_sampling_ratio: Ratio of negative to positive samples
        """
        self.neg_sampling_ratio = neg_sampling_ratio
    
    def create_task(self, data: Data) -> Tuple[Data, torch.Tensor]:
        """Create edge prediction task."""
        edge_index = data.edge_index
        num_nodes = data.x.size(0)
        
        # Remove some edges for prediction (train/test split)
        num_edges = edge_index.size(1)
        num_test_edges = max(1, num_edges // 10)  # Use 10% for testing
        
        perm = torch.randperm(num_edges)
        train_edges = edge_index[:, perm[num_test_edges:]]
        test_pos_edges = edge_index[:, perm[:num_test_edges]]
        
        # Generate negative edges
        neg_edge_index = negative_sampling(
            edge_index=train_edges,
            num_nodes=num_nodes,
            num_neg_samples=int(test_pos_edges.size(1) * self.neg_sampling_ratio)
        )
        
        # Create targets (1 for positive edges, 0 for negative edges)
        pos_targets = torch.ones(test_pos_edges.size(1))
        neg_targets = torch.zeros(neg_edge_index.size(1))
        targets = torch.cat([pos_targets, neg_targets])
        
        # Create modified data
        modified_data = data.clone()
        modified_data.edge_index = train_edges
        modified_data.test_pos_edges = test_pos_edges
        modified_data.test_neg_edges = neg_edge_index
        modified_data.edge_label_index = torch.cat([test_pos_edges, neg_edge_index], dim=1)
        
        return modified_data, targets
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss for edge prediction."""
        return F.binary_cross_entropy_with_logits(predictions, targets)


class GraphContrastiveLearning(SSLTask):
    """
    Graph contrastive learning task - learn representations by contrasting
    augmented versions of the same graph.
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 augmentation_prob: float = 0.2):
        """
        Initialize graph contrastive learning.
        
        Args:
            temperature: Temperature parameter for contrastive loss
            augmentation_prob: Probability for graph augmentations
        """
        self.temperature = temperature
        self.augmentation_prob = augmentation_prob
    
    def augment_graph(self, data: Data) -> Data:
        """Apply random augmentations to the graph."""
        augmented_data = data.clone()
        
        # Edge dropping
        if random.random() < self.augmentation_prob:
            edge_index = augmented_data.edge_index
            num_edges = edge_index.size(1)
            keep_edges = torch.rand(num_edges) > 0.1  # Drop 10% of edges
            augmented_data.edge_index = edge_index[:, keep_edges]
        
        # Node feature noise
        if random.random() < self.augmentation_prob:
            noise = torch.randn_like(augmented_data.x) * 0.1
            augmented_data.x = augmented_data.x + noise
        
        # Attribute masking
        if random.random() < self.augmentation_prob:
            mask = torch.rand_like(augmented_data.x) > 0.1
            augmented_data.x = augmented_data.x * mask
        
        return augmented_data
    
    def create_task(self, data: Data) -> Tuple[Data, torch.Tensor]:
        """Create contrastive learning task with two augmented views."""
        # Create two augmented views
        view1 = self.augment_graph(data)
        view2 = self.augment_graph(data)
        
        # Batch the two views
        batch_data = Batch.from_data_list([view1, view2])
        
        # Targets are not used in contrastive learning directly
        # But we return batch indices to identify corresponding pairs
        targets = torch.arange(2)  # Two views
        
        return batch_data, targets
    
    def compute_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            z1: Embeddings from first view
            z2: Embeddings from second view
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Create labels for contrastive learning
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class NodeClassification(SSLTask):
    """
    Node classification as a pretext task.
    Predicts node types based on local graph structure.
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize node classification task.
        
        Args:
            num_classes: Number of node classes to predict
        """
        self.num_classes = num_classes
    
    def create_task(self, data: Data) -> Tuple[Data, torch.Tensor]:
        """Create node classification task based on node degree."""
        # Use node degree as a simple classification target
        edge_index = data.edge_index
        num_nodes = data.x.size(0)
        
        # Compute node degrees
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index.size(1)):
            degrees[edge_index[0, i]] += 1
        
        # Create degree-based classes
        # Low degree (0-2), Medium degree (3-5), High degree (6+)
        targets = torch.zeros(num_nodes, dtype=torch.long)
        targets[degrees <= 2] = 0
        targets[(degrees > 2) & (degrees <= 5)] = 1
        targets[degrees > 5] = 2
        
        return data, targets
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for node classification."""
        return F.cross_entropy(predictions, targets)


class CombinedSSLTasks:
    """
    Combines multiple SSL tasks for joint training.
    """
    
    def __init__(self, 
                 tasks: List[SSLTask],
                 task_weights: Optional[List[float]] = None):
        """
        Initialize combined SSL tasks.
        
        Args:
            tasks: List of SSL tasks
            task_weights: Weights for combining task losses
        """
        self.tasks = tasks
        self.task_weights = task_weights or [1.0] * len(tasks)
        
        if len(self.task_weights) != len(self.tasks):
            raise ValueError("Number of task weights must match number of tasks")
    
    def create_tasks(self, data: Data) -> Dict[str, Tuple[Data, torch.Tensor]]:
        """Create all SSL tasks."""
        task_data = {}
        
        for i, task in enumerate(self.tasks):
            task_name = task.__class__.__name__
            modified_data, targets = task.create_task(data)
            task_data[f"{task_name}_{i}"] = (modified_data, targets)
        
        return task_data
    
    def compute_combined_loss(self, task_losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute weighted combination of task losses."""
        if len(task_losses) != len(self.task_weights):
            raise ValueError("Number of losses must match number of task weights")
        
        weighted_losses = [weight * loss for weight, loss in zip(self.task_weights, task_losses)]
        return sum(weighted_losses)


# Example usage and testing
if __name__ == "__main__":
    # Create dummy graph data
    num_nodes = 100
    num_features = 32
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    data = Data(x=x, edge_index=edge_index)
    
    print("Testing SSL Tasks:")
    
    # Test Masked Node Prediction
    masked_task = MaskedNodePrediction(mask_rate=0.15)
    masked_data, targets = masked_task.create_task(data)
    print(f"Masked {len(masked_data.mask_indices)} nodes out of {num_nodes}")
    
    # Test Edge Prediction
    edge_task = EdgePrediction(neg_sampling_ratio=1.0)
    edge_data, edge_targets = edge_task.create_task(data)
    print(f"Edge prediction with {len(edge_targets)} edge pairs")
    
    # Test Graph Contrastive Learning
    contrastive_task = GraphContrastiveLearning(temperature=0.07)
    contrast_data, contrast_targets = contrastive_task.create_task(data)
    print(f"Contrastive learning with batch size {contrast_data.batch.max() + 1}")
    
    # Test Node Classification
    node_class_task = NodeClassification(num_classes=3)
    class_data, class_targets = node_class_task.create_task(data)
    print(f"Node classification with classes: {torch.unique(class_targets)}")
    
    # Test Combined Tasks
    combined_tasks = CombinedSSLTasks([
        masked_task,
        edge_task,
        node_class_task
    ], task_weights=[1.0, 0.5, 0.3])
    
    all_task_data = combined_tasks.create_tasks(data)
    print(f"Combined {len(all_task_data)} SSL tasks")
