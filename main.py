"""
Main entry point for LogGraph-SSL framework.
Provides a command-line interface for training and evaluation.
"""

import argparse
import os
import sys
from typing import List, Dict, Any

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import create_sample_log_data, save_results


def main():
    """Main entry point for LogGraph-SSL framework."""
    parser = argparse.ArgumentParser(
        description='LogGraph-SSL: Parsing-free anomaly detection for distributed system logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample data
  python main.py create-data --output sample_data.txt --num_samples 1000

  # Train model
  python train.py --data_path sample_data.txt --output_dir ./outputs

  # Evaluate model
  python evaluate.py --model_path ./outputs/best_model.pth --vocab_path ./outputs/vocabulary.pth --test_data_path sample_data.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create sample data command
    create_parser = subparsers.add_parser('create-data', help='Create sample log data for testing')
    create_parser.add_argument('--output', type=str, required=True, help='Output file path')
    create_parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    create_parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='Ratio of anomalous samples')
    create_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show framework information')
    
    args = parser.parse_args()
    
    if args.command == 'create-data':
        create_sample_data_command(args)
    elif args.command == 'info':
        show_info()
    else:
        parser.print_help()


def create_sample_data_command(args):
    """Create sample log data."""
    print(f"Creating {args.num_samples} sample log messages...")
    print(f"Anomaly ratio: {args.anomaly_ratio}")
    print(f"Random seed: {args.seed}")
    
    messages, labels = create_sample_log_data(
        num_samples=args.num_samples,
        anomaly_ratio=args.anomaly_ratio,
        random_seed=args.seed
    )
    
    # Save messages
    with open(args.output, 'w') as f:
        for message in messages:
            f.write(message + '\n')
    
    # Save labels
    label_file = args.output.replace('.txt', '_labels.txt')
    with open(label_file, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')
    
    print(f"Sample data saved to: {args.output}")
    print(f"Labels saved to: {label_file}")
    print(f"Normal messages: {labels.count(0)}")
    print(f"Anomalous messages: {labels.count(1)}")


def show_info():
    """Show framework information."""
    print("""
LogGraph-SSL: Parsing-free Anomaly Detection Framework
====================================================

A Graph Neural Network (GNN) based framework for anomaly detection in 
distributed system logs using Self-Supervised Learning (SSL).

Key Features:
- Parsing-free approach using token co-occurrence graphs
- Self-supervised learning with multiple pretext tasks
- Support for GCN, GAT, and GraphSAGE architectures
- Multiple anomaly detection methods
- Evaluation on benchmark datasets (HDFS, BGL, Thunderbird)

Components:
- log_graph_builder.py: Graph construction from raw log messages
- ssl_tasks.py: Self-supervised learning tasks implementation
- gnn_model.py: Graph Neural Network architectures
- anomaly_detector.py: Downstream anomaly detection methods
- train.py: Model training script
- evaluate.py: Model evaluation script
- utils.py: Utility functions

Usage:
1. Create or prepare log data
2. Train the model using SSL tasks: python train.py --data_path <data>
3. Evaluate on test data: python evaluate.py --model_path <model> --test_data_path <test>

For more information, see the README.md file.
    """)


if __name__ == "__main__":
    main()
