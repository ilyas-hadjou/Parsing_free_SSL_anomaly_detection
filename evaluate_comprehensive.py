"""
Comprehensive evaluation script for the optimized LogGraph-SSL model.
Compares performance across different configurations and datasets.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from anomaly_detector import AnomalyDetector
from utils import load_log_data, load_log_labels


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for LogGraph-SSL models.
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.results = {}
    
    def load_model(self, model_path: str, graph_builder_path: str, model_config: dict) -> Tuple[LogGraphSSL, LogGraphBuilder]:
        """Load trained model and graph builder."""
        # Initialize model
        model = LogGraphSSL(**model_config)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        
        # Load graph builder
        graph_builder = torch.load(graph_builder_path, map_location=self.device, weights_only=False)
        
        return model, graph_builder
    
    def evaluate_ssl_tasks(self, model: LogGraphSSL, graph, num_samples: int = 1000) -> Dict:
        """Evaluate SSL tasks performance."""
        model.eval()
        graph = graph.to(str(self.device))
        
        if graph.x is None or graph.edge_index is None:
            return {'error': 'Graph data incomplete'}
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model(graph.x, graph.edge_index)
            
            # 1. Representation Quality
            embedding_variance = torch.var(embeddings, dim=0).mean().item()
            cosine_sim_mean = torch.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1).mean().item()
            embedding_norm = torch.norm(embeddings, dim=1).mean().item()
            
            # 2. Edge Prediction
            from torch_geometric.utils import negative_sampling
            
            # Sample edges for evaluation
            num_edges = min(num_samples, graph.edge_index.size(1))
            edge_indices = torch.randperm(graph.edge_index.size(1), device=self.device)[:num_edges]
            pos_edge_index = graph.edge_index[:, edge_indices]
            
            neg_edge_index = negative_sampling(
                pos_edge_index, num_nodes=graph.x.size(0),
                num_neg_samples=num_edges
            )
            
            # Get edge predictions
            pos_logits, neg_logits = model.forward_edge_prediction_with_hard_negatives(
                graph.x, graph.edge_index, pos_edge_index, neg_edge_index
            )
            
            # Calculate edge prediction metrics
            all_logits = torch.cat([pos_logits, neg_logits])
            all_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            
            edge_auc = roc_auc_score(
                all_labels.cpu().numpy(),
                torch.sigmoid(all_logits).cpu().numpy()
            )
            
            # 3. Node Classification
            node_logits = model.forward_node_classification(graph.x, graph.edge_index)
            
            # Generate pseudo-labels for evaluation
            degrees = torch.bincount(graph.edge_index[0], minlength=graph.x.size(0)).float()
            degree_threshold = degrees.median()
            pseudo_labels = (degrees > degree_threshold).long()
            
            node_predictions = node_logits[:, :2].argmax(dim=1)  # Binary classification
            node_accuracy = (node_predictions == pseudo_labels).float().mean().item()
            
            return {
                'embedding_variance': embedding_variance,
                'cosine_similarity': cosine_sim_mean,
                'embedding_norm': embedding_norm,
                'edge_prediction_auc': edge_auc,
                'node_classification_accuracy': node_accuracy,
                'num_nodes': graph.x.size(0),
                'num_edges': graph.edge_index.size(1)
            }
    
    def evaluate_anomaly_detection(self, model: LogGraphSSL, graph_builder: LogGraphBuilder,
                                 test_messages: List[str], test_labels: List[int]) -> Dict:
        """Evaluate anomaly detection performance."""
        
        # Build test graph
        test_graph = graph_builder.build_graph_from_logs(test_messages)
        test_graph = test_graph.to(str(self.device))
        
        if test_graph.x is None or test_graph.edge_index is None:
            return {'error': 'Test graph incomplete'}
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(test_graph.x, test_graph.edge_index)
        
        embeddings_np = embeddings.cpu().numpy()
        results = {}
        
        # 1. Isolation Forest
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest_preds = iso_forest.fit_predict(embeddings_np)
            iso_forest_scores = iso_forest.score_samples(embeddings_np)
            iso_forest_labels = (iso_forest_preds == -1).astype(int)
            
            results['isolation_forest'] = {
                'accuracy': accuracy_score(test_labels, iso_forest_labels),
                'auc': roc_auc_score(test_labels, -iso_forest_scores) if len(np.unique(test_labels)) > 1 else 0.5
            }
        except Exception as e:
            results['isolation_forest'] = {'error': str(e)}
        
        # 2. One-Class SVM
        try:
            from sklearn.svm import OneClassSVM
            svm = OneClassSVM(nu=0.1, gamma='scale')
            svm_preds = svm.fit_predict(embeddings_np)
            svm_scores = svm.score_samples(embeddings_np)
            svm_labels = (svm_preds == -1).astype(int)
            
            results['one_class_svm'] = {
                'accuracy': accuracy_score(test_labels, svm_labels),
                'auc': roc_auc_score(test_labels, -svm_scores) if len(np.unique(test_labels)) > 1 else 0.5
            }
        except Exception as e:
            results['one_class_svm'] = {'error': str(e)}
        
        # 3. Reconstruction Error (using masked node prediction head)
        try:
            with torch.no_grad():
                reconstructed = model.masked_node_head(embeddings)
                reconstruction_errors = torch.mean((test_graph.x - reconstructed) ** 2, dim=1)
                
            # Threshold-based anomaly detection
            threshold = torch.quantile(reconstruction_errors, 0.9)
            recon_labels = (reconstruction_errors > threshold).int().cpu().numpy()
            
            results['reconstruction_error'] = {
                'accuracy': accuracy_score(test_labels, recon_labels),
                'auc': roc_auc_score(test_labels, reconstruction_errors.cpu().numpy()) if len(np.unique(test_labels)) > 1 else 0.5
            }
        except Exception as e:
            results['reconstruction_error'] = {'error': str(e)}
        
        # 4. Statistical method using embedding distances
        try:
            # Use distance to embedding centroid as anomaly score
            centroid = np.mean(embeddings_np, axis=0)
            distances = np.linalg.norm(embeddings_np - centroid, axis=1)
            
            threshold = np.quantile(distances, 0.9)
            stat_labels = (distances > threshold).astype(int)
            
            results['statistical_distance'] = {
                'accuracy': accuracy_score(test_labels, stat_labels),
                'auc': roc_auc_score(test_labels, distances) if len(np.unique(test_labels)) > 1 else 0.5
            }
        except Exception as e:
            results['statistical_distance'] = {'error': str(e)}
        
        return results
    
    def run_comprehensive_evaluation(self, model_dir: str, test_data_path: str, 
                                   test_labels_path: str, model_config: dict) -> Dict:
        """Run comprehensive evaluation."""
        print("Starting comprehensive evaluation...")
        
        # Load model
        model_path = os.path.join(model_dir, 'memory_efficient_model.pth')
        graph_builder_path = os.path.join(model_dir, 'graph_builder.pth')
        
        if not os.path.exists(model_path) or not os.path.exists(graph_builder_path):
            return {'error': 'Model files not found'}
        
        model, graph_builder = self.load_model(model_path, graph_builder_path, model_config)
        
        # Load test data
        test_messages = load_log_data(test_data_path)
        test_labels = load_log_labels(test_labels_path)
        
        # Ensure we have matching data
        min_len = min(len(test_messages), len(test_labels))
        test_messages = test_messages[:min_len]
        test_labels = test_labels[:min_len]
        
        print(f"Evaluating on {len(test_messages)} test samples")
        
        # Build test graph
        test_graph = graph_builder.build_graph_from_logs(test_messages)
        
        # 1. SSL Tasks Evaluation
        print("Evaluating SSL tasks...")
        ssl_results = self.evaluate_ssl_tasks(model, test_graph)
        
        # 2. Anomaly Detection Evaluation
        print("Evaluating anomaly detection...")
        anomaly_results = self.evaluate_anomaly_detection(model, graph_builder, test_messages, test_labels)
        
        # Combine results
        results = {
            'model_config': model_config,
            'test_data_info': {
                'num_messages': len(test_messages),
                'num_anomalies': sum(test_labels),
                'anomaly_rate': sum(test_labels) / len(test_labels)
            },
            'ssl_tasks': ssl_results,
            'anomaly_detection': anomaly_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive LogGraph-SSL Evaluation')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--test_data', type=str, required=True, help='Test data path')
    parser.add_argument('--test_labels', type=str, required=True, help='Test labels path')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model configuration from saved file or use default
    config_path = os.path.join(args.model_dir, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded model configuration from {config_path}")
    else:
        # Fallback configuration - try to detect from model file
        model_path = os.path.join(args.model_dir, 'memory_efficient_model.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            encoder_weight = state_dict['encoder.convs.0.lin.weight']
            input_dim = encoder_weight.size(1)
            output_weight = state_dict['masked_node_head.3.weight']
            output_dim = output_weight.size(1)
            
            model_config = {
                'input_dim': input_dim,
                'hidden_dims': [64, 32],
                'output_dim': output_dim,
                'encoder_type': 'gat',
                'num_heads': 4,
                'dropout': 0.3
            }
            print(f"Auto-detected model configuration: input_dim={input_dim}, output_dim={output_dim}")
        else:
            print("Could not find model file or configuration")
            return
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(device=device)
    
    # Run evaluation
    try:
        results = evaluator.run_comprehensive_evaluation(
            args.model_dir, args.test_data, args.test_labels, model_config
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"comprehensive_eval_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate report
        report_lines = []
        report_lines.append("# LogGraph-SSL Comprehensive Evaluation Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Test data info
        if 'test_data_info' in results:
            info = results['test_data_info']
            report_lines.append("## Test Data Information")
            report_lines.append(f"- Number of messages: {info['num_messages']}")
            report_lines.append(f"- Number of anomalies: {info['num_anomalies']}")
            report_lines.append(f"- Anomaly rate: {info['anomaly_rate']:.2%}")
            report_lines.append("")
        
        # SSL tasks results
        if 'ssl_tasks' in results and 'error' not in results['ssl_tasks']:
            ssl = results['ssl_tasks']
            report_lines.append("## SSL Tasks Performance")
            report_lines.append(f"- **Embedding Variance**: {ssl['embedding_variance']:.6f}")
            report_lines.append(f"- **Cosine Similarity**: {ssl['cosine_similarity']:.4f}")
            report_lines.append(f"- **Edge Prediction AUC**: {ssl['edge_prediction_auc']:.4f}")
            report_lines.append(f"- **Node Classification Accuracy**: {ssl['node_classification_accuracy']:.4f}")
            report_lines.append("")
        
        # Anomaly detection results
        if 'anomaly_detection' in results:
            report_lines.append("## Anomaly Detection Performance")
            
            for method, metrics in results['anomaly_detection'].items():
                if 'error' not in metrics:
                    report_lines.append(f"### {method.replace('_', ' ').title()}")
                    report_lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
                    report_lines.append(f"- AUC: {metrics['auc']:.4f}")
                    report_lines.append("")
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation completed! Results saved to {output_dir}")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        if 'ssl_tasks' in results and 'error' not in results['ssl_tasks']:
            ssl = results['ssl_tasks']
            print(f"Embedding Variance: {ssl['embedding_variance']:.6f}")
            print(f"Edge Prediction AUC: {ssl['edge_prediction_auc']:.4f}")
            print(f"Node Classification Acc: {ssl['node_classification_accuracy']:.4f}")
        
        if 'anomaly_detection' in results:
            print("\nAnomaly Detection Results:")
            for method, metrics in results['anomaly_detection'].items():
                if 'error' not in metrics:
                    print(f"  {method}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
