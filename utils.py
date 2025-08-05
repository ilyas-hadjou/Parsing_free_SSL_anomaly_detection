"""
Utility functions for LogGraph-SSL framework.
Includes data loading, preprocessing, and evaluation metrics.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import random
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import re
from datetime import datetime


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_log_data(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    Load log data from file.
    
    Args:
        file_path: Path to log file
        encoding: File encoding
        
    Returns:
        List of log messages
    """
    log_messages = []
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    log_messages.append(line)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except UnicodeDecodeError:
        print(f"Error: Could not decode file {file_path} with encoding {encoding}")
        return []
    
    return log_messages


def load_log_labels(file_path: str) -> List[int]:
    """
    Load log labels from file.
    
    Args:
        file_path: Path to label file
        
    Returns:
        List of labels (0=normal, 1=anomaly)
    """
    labels = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Try to parse as integer
                    try:
                        label = int(line)
                        labels.append(label)
                    except ValueError:
                        # Try to parse as string (Normal/Anomaly)
                        if line.lower() in ['normal', '0', 'false']:
                            labels.append(0)
                        elif line.lower() in ['anomaly', 'abnormal', '1', 'true']:
                            labels.append(1)
                        else:
                            print(f"Warning: Unknown label '{line}', treating as normal")
                            labels.append(0)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    
    return labels


def preprocess_log_message(message: str) -> str:
    """
    Preprocess a log message.
    
    Args:
        message: Raw log message
        
    Returns:
        Preprocessed log message
    """
    # Remove timestamps (common patterns)
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # MM/DD/YYYY HH:MM:SS
        r'\w{3} \d{1,2} \d{2}:\d{2}:\d{2}',       # Mon DD HH:MM:SS
        r'\d{2}:\d{2}:\d{2}',                     # HH:MM:SS
    ]
    
    for pattern in timestamp_patterns:
        message = re.sub(pattern, '', message)
    
    # Remove IP addresses
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    message = re.sub(ip_pattern, '<IP>', message)
    
    # Remove hex numbers (memory addresses, etc.)
    hex_pattern = r'0x[0-9a-fA-F]+'
    message = re.sub(hex_pattern, '<HEX>', message)
    
    # Remove long numbers (IDs, etc.)
    number_pattern = r'\b\d{6,}\b'
    message = re.sub(number_pattern, '<NUM>', message)
    
    # Remove file paths
    path_pattern = r'[/\\][\w/\\.-]*'
    message = re.sub(path_pattern, '<PATH>', message)
    
    # Remove URLs
    url_pattern = r'https?://[^\s]+'
    message = re.sub(url_pattern, '<URL>', message)
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    message = re.sub(email_pattern, '<EMAIL>', message)
    
    # Normalize whitespace
    message = re.sub(r'\s+', ' ', message)
    message = message.strip()
    
    return message


def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   **kwargs) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, 
                   device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def calculate_metrics(y_true: List[int], 
                     y_pred: List[int], 
                     y_scores: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # AUC if scores provided
    if y_scores is not None:
        try:
            metrics['auc_score'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['auc_score'] = 0.0
    
    return metrics


def create_log_dataset_splits(log_messages: List[str],
                            labels: Optional[List[int]] = None,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_seed: int = 42) -> Dict[str, Tuple[List[str], Optional[List[int]]]]:
    """
    Split log dataset into train/validation/test sets.
    
    Args:
        log_messages: List of log messages
        labels: Optional list of labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for splitting
        
    Returns:
        Dictionary with split data
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Set seed for reproducible splits
    np.random.seed(random_seed)
    
    n_samples = len(log_messages)
    indices = np.random.permutation(n_samples)
    
    # Calculate split indices
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create splits
    splits = {}
    
    # Training split
    train_messages = [log_messages[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices] if labels else None
    splits['train'] = (train_messages, train_labels)
    
    # Validation split
    val_messages = [log_messages[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices] if labels else None
    splits['val'] = (val_messages, val_labels)
    
    # Test split
    test_messages = [log_messages[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices] if labels else None
    splits['test'] = (test_messages, test_labels)
    
    return splits


def load_benchmark_dataset(dataset_name: str, 
                          data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load benchmark datasets (HDFS, BGL, etc.).
    
    Args:
        dataset_name: Name of the dataset ('hdfs', 'bgl', 'thunderbird')
        data_dir: Directory containing dataset files
        
    Returns:
        Tuple of (log_messages, labels)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'hdfs':
        return load_hdfs_dataset(data_dir)
    elif dataset_name == 'bgl':
        return load_bgl_dataset(data_dir)
    elif dataset_name == 'thunderbird':
        return load_thunderbird_dataset(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_hdfs_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """Load HDFS dataset."""
    log_file = os.path.join(data_dir, 'HDFS.log')
    label_file = os.path.join(data_dir, 'HDFS_labels.txt')
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"HDFS log file not found: {log_file}")
    
    # Load log messages
    log_messages = load_log_data(log_file)
    
    # Load labels if available
    labels = []
    if os.path.exists(label_file):
        labels = load_log_labels(label_file)
    else:
        # Create dummy labels (all normal) if not available
        labels = [0] * len(log_messages)
        print("Warning: HDFS labels not found, using all normal labels")
    
    return log_messages, labels


def load_bgl_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """Load BGL dataset."""
    log_file = os.path.join(data_dir, 'BGL.log')
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"BGL log file not found: {log_file}")
    
    log_messages = []
    labels = []
    
    # BGL dataset typically has labels in the log file
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract label and message (format may vary)
                if line.startswith('-') or 'FAILURE' in line:
                    labels.append(1)  # Anomaly
                else:
                    labels.append(0)  # Normal
                
                log_messages.append(line)
    
    return log_messages, labels


def load_thunderbird_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    """Load Thunderbird dataset."""
    log_file = os.path.join(data_dir, 'Thunderbird.log')
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Thunderbird log file not found: {log_file}")
    
    log_messages = []
    labels = []
    
    # Thunderbird dataset typically has labels in the log file
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract label and message (format may vary)
                if 'FAILURE' in line or 'ERROR' in line:
                    labels.append(1)  # Anomaly
                else:
                    labels.append(0)  # Normal
                
                log_messages.append(line)
    
    return log_messages, labels


def save_results(results: Dict[str, Any], 
                filepath: str,
                format: str = 'json') -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Output file path
        format: Output format ('json', 'pickle')
    """
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str, format: str = 'json') -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        filepath: Input file path
        format: Input format ('json', 'pickle')
        
    Returns:
        Results dictionary
    """
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def print_model_summary(model: nn.Module) -> None:
    """Print model summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)


def create_sample_log_data(num_samples: int = 1000, 
                          anomaly_ratio: float = 0.1,
                          random_seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Create sample log data for testing.
    
    Args:
        num_samples: Number of log samples
        anomaly_ratio: Ratio of anomalous samples
        random_seed: Random seed
        
    Returns:
        Tuple of (log_messages, labels)
    """
    np.random.seed(random_seed)
    
    # Normal log templates
    normal_templates = [
        "INFO [main] Application started successfully on port {}",
        "INFO [worker-{}] Processing request id={}",
        "INFO [db] Connection established to database",
        "INFO [cache] Cache hit for key={}",
        "INFO [auth] User {} authenticated successfully",
        "DEBUG [service] Executing query: SELECT * FROM {}",
        "INFO [scheduler] Task {} completed in {} ms",
        "INFO [monitor] System health check passed",
        "INFO [api] GET /users/{} returned 200",
        "INFO [session] Session {} created for user {}",
    ]
    
    # Anomalous log templates
    anomaly_templates = [
        "ERROR [main] OutOfMemoryError: Java heap space",
        "ERROR [worker-{}] Connection timeout to external service",
        "ERROR [db] SQLException: Connection refused",
        "FATAL [system] Critical system failure detected",
        "ERROR [auth] Authentication failed for user {}",
        "ERROR [api] Internal server error: {}",
        "ERROR [disk] Disk space critically low: {}% used",
        "ERROR [network] Network unreachable: {}",
        "ERROR [security] Unauthorized access attempt from {}",
        "ERROR [service] Service unavailable: {}",
    ]
    
    log_messages = []
    labels = []
    
    num_anomalies = int(num_samples * anomaly_ratio)
    num_normal = num_samples - num_anomalies
    
    # Generate normal logs
    for _ in range(num_normal):
        template = np.random.choice(normal_templates)
        
        # Fill in placeholders
        if '{}' in template:
            if 'port' in template:
                message = template.format(np.random.randint(8000, 9000))
            elif 'worker' in template:
                message = template.format(
                    np.random.randint(1, 10),
                    np.random.randint(10000, 99999)
                )
            elif 'key=' in template:
                message = template.format(f"key_{np.random.randint(1000, 9999)}")
            elif 'User' in template:
                message = template.format(f"user_{np.random.randint(100, 999)}")
            elif 'query' in template:
                message = template.format(f"table_{np.random.randint(1, 10)}")
            elif 'Task' in template:
                message = template.format(
                    f"task_{np.random.randint(1, 100)}",
                    np.random.randint(100, 5000)
                )
            elif '/users/' in template:
                message = template.format(np.random.randint(1, 1000))
            elif 'Session' in template:
                message = template.format(
                    f"sess_{np.random.randint(10000, 99999)}",
                    f"user_{np.random.randint(100, 999)}"
                )
            else:
                message = template.format(np.random.randint(1, 100))
        else:
            message = template
        
        log_messages.append(message)
        labels.append(0)
    
    # Generate anomalous logs
    for _ in range(num_anomalies):
        template = np.random.choice(anomaly_templates)
        
        # Fill in placeholders
        if '{}' in template:
            if 'worker' in template:
                message = template.format(np.random.randint(1, 10))
            elif 'user' in template:
                message = template.format(f"user_{np.random.randint(100, 999)}")
            elif 'error:' in template:
                message = template.format("NullPointerException")
            elif 'Disk space' in template:
                message = template.format(np.random.randint(90, 99))
            elif 'Network' in template:
                message = template.format(f"10.0.0.{np.random.randint(1, 255)}")
            elif 'access attempt' in template:
                message = template.format(f"192.168.1.{np.random.randint(1, 255)}")
            elif 'Service' in template:
                message = template.format(f"service_{np.random.randint(1, 10)}")
            else:
                message = template.format(np.random.randint(1, 100))
        else:
            message = template
        
        log_messages.append(message)
        labels.append(1)
    
    # Shuffle the data
    combined = list(zip(log_messages, labels))
    np.random.shuffle(combined)
    log_messages, labels = zip(*combined)
    
    return list(log_messages), list(labels)


# Example usage and testing
if __name__ == "__main__":
    # Test sample data creation
    print("Creating sample log data...")
    messages, labels = create_sample_log_data(num_samples=100, anomaly_ratio=0.1)
    
    print(f"Created {len(messages)} log messages")
    print(f"Normal messages: {labels.count(0)}")
    print(f"Anomalous messages: {labels.count(1)}")
    
    # Show some examples
    print("\nSample normal messages:")
    normal_indices = [i for i, label in enumerate(labels) if label == 0][:3]
    for i in normal_indices:
        print(f"  {messages[i]}")
    
    print("\nSample anomalous messages:")
    anomaly_indices = [i for i, label in enumerate(labels) if label == 1][:3]
    for i in anomaly_indices:
        print(f"  {messages[i]}")
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    test_message = "2024-01-01 10:00:01 INFO [main] User user_123 connected from 192.168.1.100"
    preprocessed = preprocess_log_message(test_message)
    print(f"Original: {test_message}")
    print(f"Preprocessed: {preprocessed}")
    
    # Test dataset splits
    print("\nTesting dataset splits...")
    splits = create_log_dataset_splits(messages, labels)
    for split_name, (split_messages, split_labels) in splits.items():
        print(f"{split_name}: {len(split_messages)} messages")
    
    print("Utils module testing completed!")
