#!/usr/bin/env python3
"""
HDFS Dataset Preprocessing Script for LogGraph-SSL Framework.

This script preprocesses the HDFS dataset by:
1. Extracting BlockIds from log messages
2. Mapping block-level labels to message-level labels
3. Creating training/test splits
4. Generating the format expected by our framework
"""

import pandas as pd
import re
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random


def extract_block_id(log_message: str) -> Optional[str]:
    """Extract BlockId from log message."""
    # Look for block ID pattern: blk_[number]
    match = re.search(r'blk_-?\d+', log_message)
    return match.group(0) if match else None


def preprocess_hdfs_dataset(log_file: str, label_file: str, 
                          output_prefix: str = 'hdfs',
                          train_ratio: float = 0.7,
                          max_messages: Optional[int] = None) -> None:
    """
    Preprocess HDFS dataset for LogGraph-SSL framework.
    
    Args:
        log_file: Path to HDFS.log file
        label_file: Path to anomaly_label.csv file
        output_prefix: Prefix for output files
        train_ratio: Ratio of data to use for training
        max_messages: Maximum number of messages to process (for testing)
    """
    print("Loading HDFS dataset...")
    
    # Load labels
    labels_df = pd.read_csv(label_file)
    label_map = dict(zip(labels_df['BlockId'], labels_df['Label']))
    print(f"Loaded {len(label_map)} block labels")
    print(f"Normal blocks: {sum(1 for label in label_map.values() if label == 'Normal')}")
    print(f"Anomaly blocks: {sum(1 for label in label_map.values() if label == 'Anomaly')}")
    
    # Process log messages
    print("Processing log messages...")
    messages = []
    message_labels = []
    block_message_count = defaultdict(int)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_messages and i >= max_messages:
                break
                
            line = line.strip()
            if not line:
                continue
            
            # Extract block ID from message
            block_id = extract_block_id(line)
            
            if block_id and block_id in label_map:
                messages.append(line)
                # Convert block label to message label (0=Normal, 1=Anomaly)
                label = 1 if label_map[block_id] == 'Anomaly' else 0
                message_labels.append(label)
                block_message_count[block_id] += 1
            
            if i % 100000 == 0:
                print(f"Processed {i} lines, found {len(messages)} labeled messages")
    
    print(f"\nDataset Statistics:")
    print(f"Total labeled messages: {len(messages)}")
    print(f"Normal messages: {sum(1 for label in message_labels if label == 0)}")
    print(f"Anomaly messages: {sum(1 for label in message_labels if label == 1)}")
    print(f"Unique blocks with messages: {len(block_message_count)}")
    
    # Create train/test split
    print(f"\nCreating train/test split ({train_ratio:.1%} train)...")
    
    # Shuffle data while keeping messages and labels aligned
    combined = list(zip(messages, message_labels))
    random.seed(42)  # For reproducibility
    random.shuffle(combined)
    
    split_idx = int(len(combined) * train_ratio)
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]
    
    # Separate messages and labels
    train_messages, train_labels = zip(*train_data)
    test_messages, test_labels = zip(*test_data)
    
    print(f"Training set: {len(train_messages)} messages")
    print(f"  Normal: {sum(1 for label in train_labels if label == 0)}")
    print(f"  Anomaly: {sum(1 for label in train_labels if label == 1)}")
    
    print(f"Test set: {len(test_messages)} messages")
    print(f"  Normal: {sum(1 for label in test_labels if label == 0)}")
    print(f"  Anomaly: {sum(1 for label in test_labels if label == 1)}")
    
    # Save processed data
    print(f"\nSaving processed data...")
    
    # Training data (for SSL training - only normal messages)
    normal_train_messages = [msg for msg, label in train_data if label == 0]
    
    with open(f'{output_prefix}_train.txt', 'w') as f:
        for message in normal_train_messages:
            f.write(message + '\n')
    
    # Test data
    with open(f'{output_prefix}_test.txt', 'w') as f:
        for message in test_messages:
            f.write(message + '\n')
    
    with open(f'{output_prefix}_test_labels.txt', 'w') as f:
        for label in test_labels:
            f.write(str(label) + '\n')
    
    # Full dataset (for evaluation)
    with open(f'{output_prefix}_full.txt', 'w') as f:
        for message in messages:
            f.write(message + '\n')
    
    with open(f'{output_prefix}_full_labels.txt', 'w') as f:
        for label in message_labels:
            f.write(str(label) + '\n')
    
    print(f"Files saved:")
    print(f"  {output_prefix}_train.txt: {len(normal_train_messages)} normal messages for SSL training")
    print(f"  {output_prefix}_test.txt: {len(test_messages)} test messages")
    print(f"  {output_prefix}_test_labels.txt: {len(test_labels)} test labels")
    print(f"  {output_prefix}_full.txt: {len(messages)} full dataset messages")
    print(f"  {output_prefix}_full_labels.txt: {len(message_labels)} full dataset labels")


def main():
    parser = argparse.ArgumentParser(description='Preprocess HDFS dataset for LogGraph-SSL')
    parser.add_argument('--log_file', default='HDFS.log', help='Path to HDFS log file')
    parser.add_argument('--label_file', default='anomaly_label.csv', help='Path to anomaly label file')
    parser.add_argument('--output_prefix', default='hdfs', help='Prefix for output files')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--max_messages', type=int, help='Maximum messages to process (for testing)')
    
    args = parser.parse_args()
    
    preprocess_hdfs_dataset(
        log_file=args.log_file,
        label_file=args.label_file,
        output_prefix=args.output_prefix,
        train_ratio=args.train_ratio,
        max_messages=args.max_messages
    )


if __name__ == "__main__":
    main()
