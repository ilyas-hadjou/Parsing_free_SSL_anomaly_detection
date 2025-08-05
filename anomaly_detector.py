"""
Anomaly detector for LogGraph-SSL framework.
Performs downstream anomaly detection using pre-trained graph embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pickle
from typing import List, Tuple, Dict, Optional, Union
import os

from gnn_model import LogGraphSSL, AnomalyDetectionHead
from log_graph_builder import LogGraphBuilder


class AnomalyDetector:
    """
    Anomaly detector using pre-trained LogGraph-SSL embeddings.
    Supports multiple detection methods including neural networks and traditional ML.
    """
    
    def __init__(self, 
                 pretrained_model: LogGraphSSL,
                 detection_method: str = 'neural',
                 threshold: Optional[float] = None,
                 contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Args:
            pretrained_model: Pre-trained LogGraph-SSL model
            detection_method: Detection method ('neural', 'isolation_forest', 'one_class_svm', 'statistical')
            threshold: Anomaly threshold (computed automatically if None)
            contamination: Expected proportion of anomalies (for unsupervised methods)
        """
        self.pretrained_model = pretrained_model
        self.detection_method = detection_method
        self.threshold = threshold
        self.contamination = contamination
        
        # Initialize detection model
        if detection_method == 'neural':
            self.detector = AnomalyDetectionHead(
                input_dim=pretrained_model.output_dim,
                hidden_dim=64,
                output_dim=1
            )
        elif detection_method == 'isolation_forest':
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        elif detection_method == 'one_class_svm':
            self.detector = OneClassSVM(gamma='scale', nu=contamination)
        elif detection_method == 'statistical':
            self.detector = None  # Statistical methods don't need a model
        else:
            raise ValueError(f"Unsupported detection method: {detection_method}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_embeddings(self, data: Data) -> torch.Tensor:
        """Extract embeddings from graph data using pre-trained model."""
        self.pretrained_model.eval()
        with torch.no_grad():
            embeddings = self.pretrained_model.get_embeddings(data.x, data.edge_index)
            return embeddings
    
    def fit(self, normal_data: List[Data], validation_data: Optional[List[Data]] = None) -> None:
        """
        Fit the anomaly detector on normal data.
        
        Args:
            normal_data: List of normal graph data
            validation_data: Optional validation data for threshold tuning
        """
        print(f"Fitting anomaly detector using {self.detection_method} method...")
        
        # Extract embeddings from normal data
        normal_embeddings = []
        for data in normal_data:
            embeddings = self.extract_embeddings(data)
            # Use graph-level aggregation (mean pooling)
            graph_embedding = torch.mean(embeddings, dim=0).cpu().numpy()
            normal_embeddings.append(graph_embedding)
        
        normal_embeddings = np.array(normal_embeddings)
        
        # Standardize features
        normal_embeddings_scaled = self.scaler.fit_transform(normal_embeddings)
        
        if self.detection_method == 'neural':
            self._fit_neural_detector(normal_embeddings_scaled, validation_data)
        elif self.detection_method in ['isolation_forest', 'one_class_svm']:
            self.detector.fit(normal_embeddings_scaled)
        elif self.detection_method == 'statistical':
            self._fit_statistical_detector(normal_embeddings_scaled)
        
        self.is_fitted = True
        print("Anomaly detector fitted successfully!")
    
    def _fit_neural_detector(self, normal_embeddings: np.ndarray, 
                           validation_data: Optional[List[Data]] = None) -> None:
        """Fit neural anomaly detector."""
        # Create training data (normal samples have label 0)
        X_train = torch.tensor(normal_embeddings, dtype=torch.float32)
        y_train = torch.zeros(len(normal_embeddings))  # All normal
        
        # Add some synthetic anomalies for training
        # Generate synthetic anomalies by adding noise
        num_synthetic_anomalies = int(len(normal_embeddings) * self.contamination)
        noise_std = np.std(normal_embeddings, axis=0) * 2
        synthetic_anomalies = []
        
        for _ in range(num_synthetic_anomalies):
            # Pick a random normal sample and add noise
            base_sample = normal_embeddings[np.random.randint(len(normal_embeddings))]
            noise = np.random.normal(0, noise_std)
            synthetic_anomaly = base_sample + noise
            synthetic_anomalies.append(synthetic_anomaly)
        
        if synthetic_anomalies:
            X_synthetic = torch.tensor(np.array(synthetic_anomalies), dtype=torch.float32)
            y_synthetic = torch.ones(len(synthetic_anomalies))  # Anomalies
            
            X_train = torch.cat([X_train, X_synthetic])
            y_train = torch.cat([y_train, y_synthetic])
        
        # Train neural detector
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        self.detector.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.detector(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Set threshold based on validation data or training data
        self.detector.eval()
        with torch.no_grad():
            normal_scores = torch.sigmoid(self.detector(torch.tensor(normal_embeddings, dtype=torch.float32))).numpy()
            self.threshold = np.percentile(normal_scores, 95)  # 95th percentile as threshold
    
    def _fit_statistical_detector(self, normal_embeddings: np.ndarray) -> None:
        """Fit statistical anomaly detector using mean and covariance."""
        self.mean = np.mean(normal_embeddings, axis=0)
        self.cov = np.cov(normal_embeddings.T)
        self.cov_inv = np.linalg.pinv(self.cov)  # Pseudo-inverse for stability
        
        # Compute threshold based on Mahalanobis distances of normal data
        distances = []
        for embedding in normal_embeddings:
            dist = self._mahalanobis_distance(embedding)
            distances.append(dist)
        
        self.threshold = np.percentile(distances, 95)  # 95th percentile as threshold
    
    def _mahalanobis_distance(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance."""
        diff = x - self.mean
        distance = np.sqrt(diff.T @ self.cov_inv @ diff)
        return distance
    
    def predict(self, test_data: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in test data.
        
        Args:
            test_data: List of test graph data
            
        Returns:
            Tuple of (predictions, scores)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Extract embeddings from test data
        test_embeddings = []
        for data in test_data:
            embeddings = self.extract_embeddings(data)
            # Use graph-level aggregation (mean pooling)
            graph_embedding = torch.mean(embeddings, dim=0).cpu().numpy()
            test_embeddings.append(graph_embedding)
        
        test_embeddings = np.array(test_embeddings)
        test_embeddings_scaled = self.scaler.transform(test_embeddings)
        
        if self.detection_method == 'neural':
            return self._predict_neural(test_embeddings_scaled)
        elif self.detection_method == 'isolation_forest':
            predictions = self.detector.predict(test_embeddings_scaled)
            scores = self.detector.score_samples(test_embeddings_scaled)
            # Convert to binary (1 for anomaly, 0 for normal)
            predictions = (predictions == -1).astype(int)
            scores = -scores  # Higher scores = more anomalous
            return predictions, scores
        elif self.detection_method == 'one_class_svm':
            predictions = self.detector.predict(test_embeddings_scaled)
            scores = self.detector.decision_function(test_embeddings_scaled)
            # Convert to binary (1 for anomaly, 0 for normal)
            predictions = (predictions == -1).astype(int)
            scores = -scores  # Higher scores = more anomalous
            return predictions, scores
        elif self.detection_method == 'statistical':
            return self._predict_statistical(test_embeddings_scaled)
    
    def _predict_neural(self, test_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using neural detector."""
        self.detector.eval()
        with torch.no_grad():
            X_test = torch.tensor(test_embeddings, dtype=torch.float32)
            scores = torch.sigmoid(self.detector(X_test)).squeeze().numpy()
            predictions = (scores > self.threshold).astype(int)
        return predictions, scores
    
    def _predict_statistical(self, test_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using statistical method."""
        scores = []
        for embedding in test_embeddings:
            score = self._mahalanobis_distance(embedding)
            scores.append(score)
        
        scores = np.array(scores)
        predictions = (scores > self.threshold).astype(int)
        return predictions, scores
    
    def evaluate(self, test_data: List[Data], test_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate anomaly detector performance.
        
        Args:
            test_data: List of test graph data
            test_labels: True labels (0=normal, 1=anomaly)
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions, scores = self.predict(test_data)
        
        # Compute metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary', zero_division=0
        )
        
        try:
            auc_score = roc_auc_score(test_labels, scores)
        except ValueError:
            auc_score = 0.0  # Handle case where all labels are the same
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'threshold': self.threshold if self.threshold is not None else 0.0
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """Save the trained anomaly detector."""
        save_dict = {
            'detection_method': self.detection_method,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        
        if self.detection_method == 'neural':
            save_dict['detector_state_dict'] = self.detector.state_dict()
        elif self.detection_method in ['isolation_forest', 'one_class_svm']:
            save_dict['detector'] = self.detector
        elif self.detection_method == 'statistical':
            save_dict['mean'] = self.mean
            save_dict['cov'] = self.cov
            save_dict['cov_inv'] = self.cov_inv
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Anomaly detector saved to {filepath}")
    
    def load(self, filepath: str, pretrained_model: LogGraphSSL) -> None:
        """Load a trained anomaly detector."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.pretrained_model = pretrained_model
        self.detection_method = save_dict['detection_method']
        self.threshold = save_dict['threshold']
        self.contamination = save_dict['contamination']
        self.scaler = save_dict['scaler']
        self.is_fitted = save_dict['is_fitted']
        
        if self.detection_method == 'neural':
            self.detector = AnomalyDetectionHead(
                input_dim=pretrained_model.output_dim,
                hidden_dim=64,
                output_dim=1
            )
            self.detector.load_state_dict(save_dict['detector_state_dict'])
        elif self.detection_method in ['isolation_forest', 'one_class_svm']:
            self.detector = save_dict['detector']
        elif self.detection_method == 'statistical':
            self.mean = save_dict['mean']
            self.cov = save_dict['cov']
            self.cov_inv = save_dict['cov_inv']
        
        print(f"Anomaly detector loaded from {filepath}")


class LogSequenceAnomalyDetector(AnomalyDetector):
    """
    Specialized anomaly detector for log sequences.
    Detects anomalies in sequences of log messages.
    """
    
    def __init__(self, 
                 pretrained_model: LogGraphSSL,
                 graph_builder: LogGraphBuilder,
                 sequence_length: int = 10,
                 detection_method: str = 'neural',
                 threshold: Optional[float] = None,
                 contamination: float = 0.1):
        """
        Initialize log sequence anomaly detector.
        
        Args:
            pretrained_model: Pre-trained LogGraph-SSL model
            graph_builder: LogGraphBuilder for processing sequences
            sequence_length: Length of log sequences to analyze
            detection_method: Detection method
            threshold: Anomaly threshold
            contamination: Expected proportion of anomalies
        """
        super().__init__(pretrained_model, detection_method, threshold, contamination)
        self.graph_builder = graph_builder
        self.sequence_length = sequence_length
    
    def create_sequences(self, log_messages: List[str], 
                        labels: Optional[List[int]] = None) -> List[Tuple[Data, int]]:
        """
        Create graph sequences from log messages.
        
        Args:
            log_messages: List of log messages
            labels: Optional labels for each message
            
        Returns:
            List of (graph_data, sequence_label) tuples
        """
        sequences = []
        
        for i in range(0, len(log_messages) - self.sequence_length + 1, self.sequence_length):
            sequence = log_messages[i:i + self.sequence_length]
            
            # Determine sequence label (1 if any message in sequence is anomalous)
            if labels is not None:
                sequence_labels = labels[i:i + self.sequence_length]
                sequence_label = 1 if any(sequence_labels) else 0
            else:
                sequence_label = 0  # Default to normal
            
            # Build graph for sequence
            graph_data = self.graph_builder.build_sequence_graph(sequence)
            sequences.append((graph_data, sequence_label))
        
        return sequences
    
    def fit_on_sequences(self, log_messages: List[str], 
                        labels: Optional[List[int]] = None) -> None:
        """Fit detector on log message sequences."""
        sequences = self.create_sequences(log_messages, labels)
        
        # Separate normal and anomalous sequences
        normal_data = [data for data, label in sequences if label == 0]
        
        self.fit(normal_data)
    
    def predict_sequences(self, log_messages: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in log message sequences."""
        sequences = self.create_sequences(log_messages)
        test_data = [data for data, _ in sequences]
        
        return self.predict(test_data)


# Example usage and testing
if __name__ == "__main__":
    # This would typically be run after training the LogGraph-SSL model
    print("Anomaly Detector Framework initialized!")
    print("Available detection methods:")
    print("- neural: Neural network-based detection")
    print("- isolation_forest: Isolation Forest")
    print("- one_class_svm: One-Class SVM")
    print("- statistical: Statistical method using Mahalanobis distance")
    
    # Example of how to use (requires trained model)
    # model = LogGraphSSL(input_dim=64, output_dim=32)
    # detector = AnomalyDetector(model, detection_method='neural')
    # detector.fit(normal_data)
    # predictions, scores = detector.predict(test_data)
    # metrics = detector.evaluate(test_data, test_labels)
