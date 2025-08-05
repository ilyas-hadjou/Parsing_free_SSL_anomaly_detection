#!/usr/bin/env python3
"""
Create a proper train/test split from the full HDFS dataset 
maintaining anomaly distribution in both splits.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_stratified_split():
    print("🔄 Creating stratified train/test split from full HDFS dataset...")
    
    # Load full dataset
    with open('hdfs_full.txt', 'r') as f:
        messages = [line.strip() for line in f.readlines()]
    
    with open('hdfs_full_labels.txt', 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    print(f"Total messages: {len(messages):,}")
    print(f"Normal messages: {labels.count(0):,} ({labels.count(0)/len(labels)*100:.1f}%)")
    print(f"Anomalous messages: {labels.count(1):,} ({labels.count(1)/len(labels)*100:.1f}%)")
    
    # Create stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"\n📊 After split:")
    print(f"Training set: {len(X_train):,} messages")
    print(f"  Normal: {y_train.count(0):,} ({y_train.count(0)/len(y_train)*100:.1f}%)")
    print(f"  Anomalous: {y_train.count(1):,} ({y_train.count(1)/len(y_train)*100:.1f}%)")
    
    print(f"Test set: {len(X_test):,} messages") 
    print(f"  Normal: {y_test.count(0):,} ({y_test.count(0)/len(y_test)*100:.1f}%)")
    print(f"  Anomalous: {y_test.count(1):,} ({y_test.count(1)/len(y_test)*100:.1f}%)")
    
    # Save new splits
    with open('hdfs_full_train.txt', 'w') as f:
        for message in X_train:
            f.write(message + '\n')
    
    with open('hdfs_full_train_labels.txt', 'w') as f:
        for label in y_train:
            f.write(str(label) + '\n')
    
    with open('hdfs_full_test.txt', 'w') as f:
        for message in X_test:
            f.write(message + '\n')
    
    with open('hdfs_full_test_labels.txt', 'w') as f:
        for label in y_test:
            f.write(str(label) + '\n')
    
    print(f"\n✅ Files created:")
    print(f"  📝 hdfs_full_train.txt ({len(X_train):,} messages)")
    print(f"  📝 hdfs_full_train_labels.txt")
    print(f"  📝 hdfs_full_test.txt ({len(X_test):,} messages)")
    print(f"  📝 hdfs_full_test_labels.txt")
    
    # Also create a normal-only training file for SSL (standard practice)
    normal_train_messages = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0]
    
    with open('hdfs_full_train_normal.txt', 'w') as f:
        for message in normal_train_messages:
            f.write(message + '\n')
    
    print(f"  📝 hdfs_full_train_normal.txt ({len(normal_train_messages):,} normal messages for SSL training)")

if __name__ == "__main__":
    create_stratified_split()
