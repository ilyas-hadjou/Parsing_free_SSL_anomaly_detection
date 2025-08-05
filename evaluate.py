"""
Evaluation script for LogGraph-SSL framework.
Evaluates the model on benchmark datasets (HDFS, BGL, etc.).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, accuracy_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional

from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from anomaly_detector import AnomalyDetector, LogSequenceAnomalyDetector
from utils import load_log_data, load_checkpoint, set_seed


class LogGraphSSLEvaluator:
    """
    Evaluator for LogGraph-SSL model on benchmark datasets.
    """
    
    def __init__(self,
                 model: LogGraphSSL,
                 graph_builder: LogGraphBuilder,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize evaluator.
        
        Args:
            model: Pre-trained LogGraph-SSL model
            graph_builder: LogGraphBuilder with fitted vocabulary
            device: Evaluation device
        """
        self.model = model.to(device)
        self.graph_builder = graph_builder
        self.device = device
        self.model.eval()
    
    def evaluate_anomaly_detection(self,
                                 test_messages: List[str],
                                 test_labels: List[int],
                                 normal_messages: List[str],
                                 detection_methods: List[str] = None,
                                 sequence_length: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            test_messages: Test log messages
            test_labels: Test labels (0=normal, 1=anomaly)
            normal_messages: Normal messages for training detectors
            detection_methods: List of detection methods to evaluate
            sequence_length: Length of log sequences
            
        Returns:
            Dictionary of results for each detection method
        """
        if detection_methods is None:
            detection_methods = ['neural', 'isolation_forest', 'one_class_svm', 'statistical']
        
        results = {}
        
        # Prepare normal data for training detectors
        print("Building graphs from normal data...")
        normal_graph = self.graph_builder.build_graph_from_logs(normal_messages)
        normal_graph = normal_graph.to(self.device)
        normal_data = [normal_graph]
        
        # Prepare test data
        print("Building graphs from test data...")
        test_graphs = []
        
        # Create sequences for evaluation
        for i in range(0, len(test_messages) - sequence_length + 1, sequence_length):
            sequence = test_messages[i:i + sequence_length]
            sequence_labels = test_labels[i:i + sequence_length]
            
            # Create graph for sequence
            graph_data = self.graph_builder.build_sequence_graph(sequence)
            graph_data = graph_data.to(self.device)  # Move to device
            
            # Sequence is anomalous if any message in it is anomalous
            sequence_label = 1 if any(sequence_labels) else 0
            
            test_graphs.append((graph_data, sequence_label))
        
        test_data = [data for data, _ in test_graphs]
        sequence_labels = [label for _, label in test_graphs]
        
        print(f"Created {len(test_data)} test sequences")
        print(f"Normal sequences: {sequence_labels.count(0)}")
        print(f"Anomalous sequences: {sequence_labels.count(1)}")
        
        # Evaluate each detection method
        for method in detection_methods:
            print(f"\nEvaluating {method} detection method...")
            
            try:
                # Calculate actual contamination rate from sequence labels
                actual_contamination = np.mean(sequence_labels)
                # Use the actual contamination rate but clamp it to reasonable bounds
                contamination = max(0.05, min(0.5, actual_contamination))
                
                print(f"Actual anomaly rate: {actual_contamination:.3f}, Using contamination: {contamination:.3f} for {method}")
                
                # Initialize detector
                detector = AnomalyDetector(
                    pretrained_model=self.model,
                    detection_method=method,
                    contamination=contamination
                )
                
                # Fit detector on normal data
                detector.fit(normal_data)
                
                # Predict on test data
                predictions, scores = detector.predict(test_data)
                
                # Evaluate performance
                metrics = detector.evaluate(test_data, sequence_labels)
                results[method] = metrics
                
                print(f"{method} Results:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {method}: {str(e)}")
                results[method] = {'error': str(e)}
        
        return results
    
    def evaluate_representation_quality(self,
                                      test_messages: List[str],
                                      test_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the quality of learned representations.
        
        Args:
            test_messages: Test log messages
            test_labels: Test labels
            
        Returns:
            Dictionary of representation quality metrics
        """
        print("Evaluating representation quality...")
        
        # Build test graph
        test_graph = self.graph_builder.build_graph_from_logs(test_messages)
        test_graph = test_graph.to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model.get_embeddings(test_graph.x, test_graph.edge_index)
            embeddings = embeddings.cpu().numpy()
        
        # Compute representation quality metrics
        metrics = {}
        
        # 1. Embedding variance (higher is better for diversity)
        embedding_var = np.var(embeddings, axis=0).mean()
        metrics['embedding_variance'] = float(embedding_var)
        
        # 2. Embedding norm (indicates activation level)
        embedding_norm = np.linalg.norm(embeddings, axis=1).mean()
        metrics['embedding_norm'] = float(embedding_norm)
        
        # 3. Pairwise cosine similarity distribution
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Sample embeddings for efficiency
        sample_size = min(1000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Remove diagonal elements
        similarity_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        metrics['mean_cosine_similarity'] = float(np.mean(similarity_values))
        metrics['std_cosine_similarity'] = float(np.std(similarity_values))
        
        print("Representation Quality Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_ssl_tasks(self, test_graph: Data) -> Dict[str, float]:
        """
        Evaluate SSL task performance.
        
        Args:
            test_graph: Test graph data
            
        Returns:
            Dictionary of SSL task metrics
        """
        print("Evaluating SSL task performance...")
        
        metrics = {}
        
        # Masked Node Prediction
        try:
            from ssl_tasks import MaskedNodePrediction
            
            mask_task = MaskedNodePrediction(mask_rate=0.15)
            masked_data, targets = mask_task.create_task(test_graph)
            
            # Move data to device
            masked_data = masked_data.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                predictions = self.model.forward_masked_nodes(
                    masked_data.x, masked_data.edge_index, masked_data.mask_indices
                )
                mask_loss = mask_task.compute_loss(predictions, targets)
                metrics['masked_node_prediction_loss'] = float(mask_loss.item())
        
        except Exception as e:
            print(f"Error in masked node prediction evaluation: {str(e)}")
            metrics['masked_node_prediction_loss'] = float('inf')
        
        # Edge Prediction
        try:
            from ssl_tasks import EdgePrediction
            
            edge_task = EdgePrediction(neg_sampling_ratio=1.0)
            edge_data, targets = edge_task.create_task(test_graph)
            
            # Move data to device
            edge_data = edge_data.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                predictions = self.model.forward_edge_prediction(
                    edge_data.x, edge_data.edge_index, edge_data.edge_label_index
                )
                edge_loss = edge_task.compute_loss(predictions, targets)
                metrics['edge_prediction_loss'] = float(edge_loss.item())
                
                # Compute AUC for edge prediction
                predictions_prob = torch.sigmoid(predictions).cpu().numpy()
                targets_np = targets.cpu().numpy()
                if len(np.unique(targets_np)) > 1:  # Check if both classes present
                    edge_auc = roc_auc_score(targets_np, predictions_prob)
                    metrics['edge_prediction_auc'] = float(edge_auc)
        
        except Exception as e:
            print(f"Error in edge prediction evaluation: {str(e)}")
            metrics['edge_prediction_loss'] = float('inf')
            metrics['edge_prediction_auc'] = 0.0
        
        # Node Classification
        try:
            from ssl_tasks import NodeClassification
            
            node_task = NodeClassification(num_classes=3)
            class_data, targets = node_task.create_task(test_graph)
            
            # Move data to device
            class_data = class_data.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                predictions = self.model.forward_node_classification(
                    class_data.x, class_data.edge_index
                )
                class_loss = node_task.compute_loss(predictions, targets)
                metrics['node_classification_loss'] = float(class_loss.item())
                
                # Compute accuracy
                pred_classes = torch.argmax(predictions, dim=1)
                accuracy = (pred_classes == targets).float().mean()
                metrics['node_classification_accuracy'] = float(accuracy.item())
        
        except Exception as e:
            print(f"Error in node classification evaluation: {str(e)}")
            metrics['node_classification_loss'] = float('inf')
            metrics['node_classification_accuracy'] = 0.0
        
        print("SSL Task Performance:")
        for metric, value in metrics.items():
            if not np.isinf(value):
                print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def generate_evaluation_report(self,
                                 results: Dict,
                                 output_dir: str,
                                 dataset_name: str = "benchmark") -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_dir: Output directory
            dataset_name: Dataset name
        """
        print(f"Generating evaluation report for {dataset_name}...")
        
        # Create report
        report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'encoder_type': self.model.encoder_type
            },
            'results': results
        }
        
        # Save report as JSON
        report_path = os.path.join(output_dir, f'{dataset_name}_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self._plot_results(results, output_dir, dataset_name)
        
        # Generate summary table
        self._generate_summary_table(results, output_dir, dataset_name)
        
        print(f"Evaluation report saved to {output_dir}")
    
    def _plot_results(self, results: Dict, output_dir: str, dataset_name: str) -> None:
        """Generate plots for evaluation results."""
        # Anomaly detection results
        if 'anomaly_detection' in results:
            anomaly_results = results['anomaly_detection']
            
            # Extract metrics for plotting
            methods = []
            accuracies = []
            f1_scores = []
            auc_scores = []
            
            for method, metrics in anomaly_results.items():
                if 'error' not in metrics:
                    methods.append(method)
                    accuracies.append(metrics.get('accuracy', 0))
                    f1_scores.append(metrics.get('f1_score', 0))
                    auc_scores.append(metrics.get('auc_score', 0))
            
            if methods:
                # Create comparison plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Accuracy
                axes[0].bar(methods, accuracies)
                axes[0].set_title('Accuracy by Detection Method')
                axes[0].set_ylabel('Accuracy')
                axes[0].tick_params(axis='x', rotation=45)
                
                # F1 Score
                axes[1].bar(methods, f1_scores)
                axes[1].set_title('F1 Score by Detection Method')
                axes[1].set_ylabel('F1 Score')
                axes[1].tick_params(axis='x', rotation=45)
                
                # AUC Score
                axes[2].bar(methods, auc_scores)
                axes[2].set_title('AUC Score by Detection Method')
                axes[2].set_ylabel('AUC Score')
                axes[2].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{dataset_name}_anomaly_detection_comparison.png'))
                plt.close()
        
        # SSL task results
        if 'ssl_tasks' in results:
            ssl_results = results['ssl_tasks']
            
            # Plot SSL losses
            loss_metrics = {k: v for k, v in ssl_results.items() if 'loss' in k and not np.isinf(v)}
            
            if loss_metrics:
                plt.figure(figsize=(10, 6))
                plt.bar(loss_metrics.keys(), loss_metrics.values())
                plt.title('SSL Task Losses')
                plt.ylabel('Loss')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{dataset_name}_ssl_losses.png'))
                plt.close()
    
    def _generate_summary_table(self, results: Dict, output_dir: str, dataset_name: str) -> None:
        """Generate summary table of results."""
        summary_data = []
        
        # Anomaly detection results
        if 'anomaly_detection' in results:
            for method, metrics in results['anomaly_detection'].items():
                if 'error' not in metrics:
                    summary_data.append({
                        'Category': 'Anomaly Detection',
                        'Method': method,
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}",
                        'F1 Score': f"{metrics.get('f1_score', 0):.4f}",
                        'AUC Score': f"{metrics.get('auc_score', 0):.4f}"
                    })
        
        # Create summary DataFrame
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, f'{dataset_name}_summary.csv')
            df.to_csv(summary_path, index=False)
            print(f"Summary table saved to {summary_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate LogGraph-SSL model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--test_labels_path', type=str, help='Path to test labels (optional)')
    parser.add_argument('--normal_data_path', type=str, help='Path to normal training data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--dataset_name', type=str, default='benchmark', help='Dataset name')
    parser.add_argument('--sequence_length', type=int, default=10, help='Log sequence length')
    parser.add_argument('--detection_methods', nargs='+', default=['neural', 'isolation_forest'], 
                       help='Detection methods to evaluate')
    parser.add_argument('--device', type=str, default='auto', help='Evaluation device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    output_dir = os.path.join(args.output_dir, f"{args.dataset_name}_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and data...")
    
    # Load vocabulary
    graph_builder = LogGraphBuilder()
    graph_builder.load_vocabulary(args.vocab_path)
    
    # Load model
    checkpoint = load_checkpoint(args.model_path, device)
    
    # Load config if available
    config_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Get input dimensions from vocabulary size
    # The input dimension should be vocabulary_size + 1 (for TF-IDF features)
    vocab_size = graph_builder.vocab_size
    input_dim = vocab_size + 1
    
    # Initialize model with correct dimensions
    model = LogGraphSSL(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [128, 64]),
        output_dim=config.get('output_dim', 32),
        encoder_type=config.get('encoder_type', 'gcn')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize evaluator
    evaluator = LogGraphSSLEvaluator(model, graph_builder, device)
    
    # Load test data
    test_messages = load_log_data(args.test_data_path)
    print(f"Loaded {len(test_messages)} test messages")
    
    # Load test labels if available
    test_labels = None
    if args.test_labels_path and os.path.exists(args.test_labels_path):
        with open(args.test_labels_path, 'r') as f:
            test_labels = [int(line.strip()) for line in f]
        print(f"Loaded {len(test_labels)} test labels")
    
    # Load normal data if available
    normal_messages = []
    if args.normal_data_path and os.path.exists(args.normal_data_path):
        normal_messages = load_log_data(args.normal_data_path)
        print(f"Loaded {len(normal_messages)} normal messages")
    else:
        # Use first portion of test data as normal (if no labels available)
        normal_messages = test_messages[:len(test_messages)//2]
        test_messages = test_messages[len(test_messages)//2:]
        if test_labels:
            test_labels = test_labels[len(test_labels)//2:]
    
    results = {}
    
    # Evaluate representation quality
    if test_labels:
        repr_metrics = evaluator.evaluate_representation_quality(test_messages, test_labels)
        results['representation_quality'] = repr_metrics
    
    # Evaluate SSL tasks
    test_graph = graph_builder.build_graph_from_logs(test_messages[:1000])  # Sample for efficiency
    test_graph = test_graph.to(device)
    ssl_metrics = evaluator.evaluate_ssl_tasks(test_graph)
    results['ssl_tasks'] = ssl_metrics
    
    # Evaluate anomaly detection
    if test_labels and normal_messages:
        anomaly_results = evaluator.evaluate_anomaly_detection(
            test_messages=test_messages,
            test_labels=test_labels,
            normal_messages=normal_messages,
            detection_methods=args.detection_methods,
            sequence_length=args.sequence_length
        )
        results['anomaly_detection'] = anomaly_results
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(results, output_dir, args.dataset_name)
    
    print(f"Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
