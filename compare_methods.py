"""
Comprehensive comparison between LogGraph-SSL and traditional anomaly detection methods.
Evaluates performance across multiple metrics and provides detailed analysis.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from utils import load_log_data, load_log_labels, preprocess_log_message


class ComprehensiveComparison:
    """
    Comprehensive comparison framework between SSL and traditional methods.
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.results = {}
        self.detailed_results = {}
    
    def load_ssl_model(self, model_dir: str) -> Tuple[LogGraphSSL, LogGraphBuilder]:
        """Load trained SSL model and graph builder."""
        # Load configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Load model
        model_path = os.path.join(model_dir, 'full_dataset_model.pth')
        if not os.path.exists(model_path):
            # Fallback to memory efficient model
            model_path = os.path.join(model_dir, 'memory_efficient_model.pth')
        
        model = LogGraphSSL(**model_config)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        
        # Load graph builder
        graph_builder_path = os.path.join(model_dir, 'graph_builder.pth')
        graph_builder = torch.load(graph_builder_path, map_location=self.device, weights_only=False)
        
        return model, graph_builder
    
    def extract_ssl_features(self, model: LogGraphSSL, graph_builder: LogGraphBuilder, 
                           messages: List[str]) -> np.ndarray:
        """Extract SSL-based features from log messages."""
        # Build graph
        graph = graph_builder.build_graph_from_logs(messages)
        graph = graph.to(str(self.device))
        
        if graph.x is None or graph.edge_index is None:
            raise ValueError("Graph construction failed")
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(graph.x, graph.edge_index)
        
        return embeddings.cpu().numpy()
    
    def extract_traditional_features(self, messages: List[str]) -> Dict[str, np.ndarray]:
        """Extract traditional features for comparison."""
        features = {}
        
        # Preprocess messages
        processed_messages = [preprocess_log_message(msg) for msg in messages]
        
        # 1. TF-IDF Features
        print("Extracting TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        tfidf_matrix = tfidf.fit_transform(processed_messages)
        # Convert sparse matrix to dense array
        tfidf_features = np.array(tfidf_matrix.todense())
        features['tfidf'] = tfidf_features
        
        # 2. Statistical Features
        print("Extracting statistical features...")
        stat_features = []
        for msg in processed_messages:
            # Length-based features
            msg_len = len(msg)
            word_count = len(msg.split())
            avg_word_len = np.mean([len(word) for word in msg.split()]) if msg.split() else 0
            
            # Character-based features
            digit_ratio = sum(c.isdigit() for c in msg) / max(len(msg), 1)
            upper_ratio = sum(c.isupper() for c in msg) / max(len(msg), 1)
            special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in msg) / max(len(msg), 1)
            
            # Frequency-based features
            unique_chars = len(set(msg))
            char_entropy = -sum((msg.count(c) / len(msg)) * np.log2(msg.count(c) / len(msg)) 
                               for c in set(msg) if msg.count(c) > 0) if msg else 0
            
            stat_features.append([
                msg_len, word_count, avg_word_len,
                digit_ratio, upper_ratio, special_char_ratio,
                unique_chars, char_entropy
            ])
        
        features['statistical'] = np.array(stat_features)
        
        # 3. Combined Features (TF-IDF + Statistical)
        features['combined'] = np.hstack([tfidf_features, features['statistical']])
        
        return features
    
    def evaluate_ssl_anomaly_detection(self, ssl_features: np.ndarray, labels: List[int]) -> Dict:
        """Evaluate SSL-based anomaly detection methods."""
        results = {}
        
        # 1. SSL + Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            iso_preds = iso_forest.fit_predict(ssl_features)
            iso_scores = iso_forest.score_samples(ssl_features)
            iso_labels = (iso_preds == -1).astype(int)
            
            results['ssl_isolation_forest'] = self._calculate_metrics(labels, iso_labels, -iso_scores)
        except Exception as e:
            results['ssl_isolation_forest'] = {'error': str(e)}
        
        # 2. SSL + One-Class SVM
        try:
            svm = OneClassSVM(nu=0.1, gamma='scale')
            svm_preds = svm.fit_predict(ssl_features)
            svm_scores = svm.score_samples(ssl_features)
            svm_labels = (svm_preds == -1).astype(int)
            
            results['ssl_one_class_svm'] = self._calculate_metrics(labels, svm_labels, -svm_scores)
        except Exception as e:
            results['ssl_one_class_svm'] = {'error': str(e)}
        
        # 3. SSL + Statistical Threshold
        try:
            # Use distance from centroid as anomaly score
            centroid = np.mean(ssl_features, axis=0)
            distances = np.linalg.norm(ssl_features - centroid, axis=1)
            threshold = np.percentile(distances, 90)
            stat_labels = (distances > threshold).astype(int)
            
            results['ssl_statistical'] = self._calculate_metrics(labels, stat_labels, distances)
        except Exception as e:
            results['ssl_statistical'] = {'error': str(e)}
        
        # 4. SSL + DBSCAN
        try:
            # Use PCA for dimensionality reduction before DBSCAN
            pca = PCA(n_components=min(50, ssl_features.shape[1]))
            ssl_pca = pca.fit_transform(ssl_features)
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(ssl_pca)
            # Outliers are labeled as -1, consider them as anomalies
            dbscan_anomaly_labels = (dbscan_labels == -1).astype(int)
            
            results['ssl_dbscan'] = self._calculate_metrics(labels, dbscan_anomaly_labels, dbscan_anomaly_labels)
        except Exception as e:
            results['ssl_dbscan'] = {'error': str(e)}
        
        return results
    
    def evaluate_traditional_anomaly_detection(self, features_dict: Dict[str, np.ndarray], 
                                             labels: List[int]) -> Dict:
        """Evaluate traditional anomaly detection methods."""
        results = {}
        
        for feature_name, features in features_dict.items():
            print(f"Evaluating traditional methods with {feature_name} features...")
            
            # 1. Isolation Forest
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
                iso_preds = iso_forest.fit_predict(features)
                iso_scores = iso_forest.score_samples(features)
                iso_labels = (iso_preds == -1).astype(int)
                
                results[f'{feature_name}_isolation_forest'] = self._calculate_metrics(labels, iso_labels, -iso_scores)
            except Exception as e:
                results[f'{feature_name}_isolation_forest'] = {'error': str(e)}
            
            # 2. One-Class SVM
            try:
                svm = OneClassSVM(nu=0.1, gamma='scale')
                svm_preds = svm.fit_predict(features)
                svm_scores = svm.score_samples(features)
                svm_labels = (svm_preds == -1).astype(int)
                
                results[f'{feature_name}_one_class_svm'] = self._calculate_metrics(labels, svm_labels, -svm_scores)
            except Exception as e:
                results[f'{feature_name}_one_class_svm'] = {'error': str(e)}
            
            # 3. DBSCAN
            try:
                # Use PCA for dimensionality reduction if needed
                if features.shape[1] > 50:
                    pca = PCA(n_components=50)
                    features_reduced = pca.fit_transform(features)
                else:
                    features_reduced = features
                
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(features_reduced)
                dbscan_anomaly_labels = (dbscan_labels == -1).astype(int)
                
                results[f'{feature_name}_dbscan'] = self._calculate_metrics(labels, dbscan_anomaly_labels, dbscan_anomaly_labels)
            except Exception as e:
                results[f'{feature_name}_dbscan'] = {'error': str(e)}
        
        return results
    
    def _calculate_metrics(self, y_true: List[int], y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            # Add AUC if we have meaningful scores
            if len(np.unique(y_true)) > 1 and not np.allclose(y_scores, y_scores[0]):
                metrics['auc'] = roc_auc_score(y_true, y_scores)
            else:
                metrics['auc'] = 0.5
            
            return metrics
        except Exception as e:
            return {'error': str(e)}
    
    def run_comprehensive_comparison(self, model_dir: str, test_data_path: str, 
                                   test_labels_path: str, sample_size: int = 10000) -> Dict:
        """Run comprehensive comparison between SSL and traditional methods."""
        print("Starting comprehensive comparison...")
        
        # Load data
        test_messages = load_log_data(test_data_path)
        test_labels = load_log_labels(test_labels_path)
        
        # Ensure matching lengths
        min_len = min(len(test_messages), len(test_labels))
        test_messages = test_messages[:min_len]
        test_labels = test_labels[:min_len]
        
        # Sample for manageable computation
        if len(test_messages) > sample_size:
            indices = np.random.choice(len(test_messages), sample_size, replace=False)
            test_messages = [test_messages[i] for i in indices]
            test_labels = [test_labels[i] for i in indices]
        
        print(f"Evaluating on {len(test_messages)} samples with {sum(test_labels)} anomalies ({100*sum(test_labels)/len(test_labels):.2f}%)")
        
        # Load SSL model
        print("Loading SSL model...")
        ssl_model, graph_builder = self.load_ssl_model(model_dir)
        
        # Extract SSL features
        print("Extracting SSL-based features...")
        ssl_features = self.extract_ssl_features(ssl_model, graph_builder, test_messages)
        print(f"SSL features shape: {ssl_features.shape}")
        
        # Extract traditional features
        print("Extracting traditional features...")
        traditional_features = self.extract_traditional_features(test_messages)
        for name, features in traditional_features.items():
            print(f"{name} features shape: {features.shape}")
        
        # Evaluate SSL methods
        print("Evaluating SSL-based anomaly detection...")
        ssl_results = self.evaluate_ssl_anomaly_detection(ssl_features, test_labels)
        
        # Evaluate traditional methods
        print("Evaluating traditional anomaly detection...")
        traditional_results = self.evaluate_traditional_anomaly_detection(traditional_features, test_labels)
        
        # Combine results
        all_results = {
            'dataset_info': {
                'num_samples': len(test_messages),
                'num_anomalies': sum(test_labels),
                'anomaly_rate': sum(test_labels) / len(test_labels),
                'ssl_feature_dim': ssl_features.shape[1]
            },
            'ssl_methods': ssl_results,
            'traditional_methods': traditional_results
        }
        
        return all_results
    
    def generate_comparison_report(self, results: Dict, output_dir: str):
        """Generate comprehensive comparison report."""
        # Create detailed performance table
        performance_data = []
        
        # Extract metrics for all methods
        all_methods = {}
        all_methods.update(results.get('ssl_methods', {}))
        all_methods.update(results.get('traditional_methods', {}))
        
        for method_name, metrics in all_methods.items():
            if 'error' not in metrics:
                performance_data.append({
                    'Method': method_name,
                    'Type': 'SSL-based' if 'ssl_' in method_name else 'Traditional',
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'AUC': metrics.get('auc', 0)
                })
        
        df = pd.DataFrame(performance_data)
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'detailed_comparison.csv'), index=False)
        
        # Generate visualizations
        if not df.empty:
            plt.figure(figsize=(20, 12))
            
            # Performance comparison by metric
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
            
            for i, metric in enumerate(metrics):
                plt.subplot(2, 3, i + 1)
                
                # Separate SSL and traditional methods
                ssl_data = df[df['Type'] == 'SSL-based'][metric]
                traditional_data = df[df['Type'] == 'Traditional'][metric]
                
                x_pos = np.arange(len(ssl_data))
                width = 0.35
                
                if len(ssl_data) > 0:
                    plt.bar(x_pos - width/2, ssl_data, width, label='SSL-based', alpha=0.8)
                if len(traditional_data) > 0:
                    plt.bar(x_pos + width/2, traditional_data, width, label='Traditional', alpha=0.8)
                
                plt.title(f'{metric} Comparison')
                plt.ylabel(metric)
                plt.legend()
                plt.xticks(x_pos, [name.replace('ssl_', '').replace('_', ' ') for name in df[df['Type'] == 'SSL-based']['Method']], rotation=45)
            
            # Overall performance heatmap
            plt.subplot(2, 3, 6)
            pivot_df = df.pivot_table(values=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'], 
                                     index='Method', aggfunc='mean')
            sns.heatmap(pivot_df.T, annot=True, cmap='Blues', fmt='.3f')
            plt.title('Performance Heatmap')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Generate text report
        report_lines = [
            "# LogGraph-SSL vs Traditional Methods Comparison Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Information",
            f"- Total samples: {results['dataset_info']['num_samples']}",
            f"- Anomalies: {results['dataset_info']['num_anomalies']}",
            f"- Anomaly rate: {results['dataset_info']['anomaly_rate']:.2%}",
            f"- SSL feature dimension: {results['dataset_info']['ssl_feature_dim']}",
            "",
            "## Performance Summary",
            ""
        ]
        
        if not df.empty:
            # Best performing methods
            best_methods = {}
            for metric in ['Accuracy', 'F1-Score', 'AUC']:
                best_idx = df[metric].idxmax()
                best_method = df.loc[best_idx]
                best_methods[metric] = {
                    'method': best_method['Method'],
                    'type': best_method['Type'],
                    'score': best_method[metric]
                }
                
                report_lines.append(f"### Best {metric}")
                report_lines.append(f"- **{best_method['Method']}** ({best_method['Type']}): {best_method[metric]:.4f}")
                report_lines.append("")
            
            # SSL vs Traditional average performance
            ssl_avg = df[df['Type'] == 'SSL-based'][['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].mean()
            traditional_avg = df[df['Type'] == 'Traditional'][['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].mean()
            
            report_lines.extend([
                "## Average Performance Comparison",
                "",
                "### SSL-based Methods",
                f"- Accuracy: {ssl_avg['Accuracy']:.4f}",
                f"- Precision: {ssl_avg['Precision']:.4f}",
                f"- Recall: {ssl_avg['Recall']:.4f}",
                f"- F1-Score: {ssl_avg['F1-Score']:.4f}",
                f"- AUC: {ssl_avg['AUC']:.4f}",
                "",
                "### Traditional Methods",
                f"- Accuracy: {traditional_avg['Accuracy']:.4f}",
                f"- Precision: {traditional_avg['Precision']:.4f}",
                f"- Recall: {traditional_avg['Recall']:.4f}",
                f"- F1-Score: {traditional_avg['F1-Score']:.4f}",
                f"- AUC: {traditional_avg['AUC']:.4f}",
                ""
            ])
        
        # Save report
        with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
            f.write('\n'.join(report_lines))
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Comprehensive SSL vs Traditional Methods Comparison')
    parser.add_argument('--model_dir', type=str, required=True, help='SSL model directory')
    parser.add_argument('--test_data', type=str, required=True, help='Test data path')
    parser.add_argument('--test_labels', type=str, required=True, help='Test labels path')
    parser.add_argument('--output_dir', type=str, default='./comparison_results', help='Output directory')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size for evaluation')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize comparison framework
    comparator = ComprehensiveComparison(device=device)
    
    # Run comparison
    try:
        results = comparator.run_comprehensive_comparison(
            args.model_dir, args.test_data, args.test_labels, args.sample_size
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"ssl_vs_traditional_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate report
        df = comparator.generate_comparison_report(results, output_dir)
        
        print(f"\n🎉 Comprehensive comparison completed!")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        if not df.empty:
            print("\n=== PERFORMANCE SUMMARY ===")
            print("\nBest performing methods:")
            for metric in ['Accuracy', 'F1-Score', 'AUC']:
                best_idx = df[metric].idxmax()
                best_method = df.loc[best_idx]
                print(f"  {metric}: {best_method['Method']} ({best_method['Type']}) - {best_method[metric]:.4f}")
            
            print(f"\nSSL-based methods average F1-Score: {df[df['Type'] == 'SSL-based']['F1-Score'].mean():.4f}")
            print(f"Traditional methods average F1-Score: {df[df['Type'] == 'Traditional']['F1-Score'].mean():.4f}")
    
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
