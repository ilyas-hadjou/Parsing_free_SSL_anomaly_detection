# LogGraph-SSL: Parsing-free Anomaly Detection Framework

A comprehensive Graph Neural Network (GNN) based framework for anomaly detection in distributed system logs using Self-Supervised Learning (SSL). This implementation provides a parsing-free approach that builds token co-occurrence graphs directly from raw log messages.

## Features

- **Parsing-free Architecture**: Directly processes raw log messages without requiring log parsing templates
- **Graph-based Representation**: Constructs token co-occurrence graphs to capture semantic relationships
- **Self-Supervised Learning**: Multiple SSL tasks including masked node prediction, edge prediction, and contrastive learning
- **Multiple GNN Architectures**: Support for GCN, GAT, and GraphSAGE models
- **Flexible Anomaly Detection**: Neural networks, Isolation Forest, One-Class SVM, and statistical methods
- **Benchmark Evaluation**: Support for HDFS, BGL, and Thunderbird datasets
- **Modular Design**: Clean separation of concerns with well-defined interfaces

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation by running:
```bash
python main.py info
```

## Quick Start

### 1. Create Sample Data
```bash
python main.py create-data --output sample_data.txt --num_samples 1000 --anomaly_ratio 0.1
```

### 2. Train the Model
```bash
python train.py --data_path sample_data.txt --output_dir ./outputs --num_epochs 50
```

### 3. Evaluate the Model
```bash
python evaluate.py \
    --model_path ./outputs/best_model.pth \
    --vocab_path ./outputs/vocabulary.pth \
    --test_data_path sample_data.txt \
    --test_labels_path sample_data_labels.txt
```

## Architecture Overview

### Core Components

1. **LogGraphBuilder** (`log_graph_builder.py`)
   - Tokenizes raw log messages using regex patterns
   - Builds token co-occurrence graphs with sliding window approach
   - Supports both global graphs and sequence graphs
   - Handles vocabulary management and graph construction

2. **SSL Tasks** (`ssl_tasks.py`)
   - **Masked Node Prediction**: Predicts masked token features (similar to BERT)
   - **Edge Prediction**: Predicts existence of edges between token pairs
   - **Graph Contrastive Learning**: Learns representations through data augmentation
   - **Node Classification**: Classifies nodes based on local graph structure

3. **GNN Models** (`gnn_model.py`)
   - **GCN Encoder**: Graph Convolutional Network implementation
   - **GAT Encoder**: Graph Attention Network with multi-head attention
   - **GraphSAGE Encoder**: Inductive representation learning
   - **LogGraph-SSL**: Main model combining encoder with SSL task heads

4. **Anomaly Detector** (`anomaly_detector.py`)
   - **Neural Detection**: Learnable anomaly detection head
   - **Isolation Forest**: Ensemble-based outlier detection
   - **One-Class SVM**: Support vector machine for novelty detection
   - **Statistical Detection**: Mahalanobis distance-based detection

5. **Training** (`train.py`)
   - Multi-task SSL pre-training with combined loss
   - Learning rate scheduling and early stopping
   - Comprehensive checkpointing and logging
   - Training history visualization

6. **Evaluation** (`evaluate.py`)
   - Performance evaluation on benchmark datasets
   - Representation quality assessment
   - SSL task validation
   - Comprehensive reporting and visualization

### Data Flow

```
Raw Log Messages → Tokenization → Co-occurrence Graph → GNN Encoder → 
SSL Tasks (Pre-training) → Graph Embeddings → Anomaly Detection → Results
```

## Usage Examples

### Custom Dataset Training

```python
from log_graph_builder import LogGraphBuilder
from gnn_model import LogGraphSSL
from ssl_tasks import CombinedSSLTasks, MaskedNodePrediction, EdgePrediction
from utils import load_log_data

# Load your log data
log_messages = load_log_data('your_logs.txt')

# Build graphs
builder = LogGraphBuilder(window_size=5, min_token_freq=2)
graph = builder.build_graph_from_logs(log_messages)

# Initialize model
model = LogGraphSSL(
    input_dim=graph.x.size(1),
    hidden_dims=[128, 64],
    output_dim=32,
    encoder_type='gcn'
)

# Setup SSL tasks
ssl_tasks = CombinedSSLTasks([
    MaskedNodePrediction(mask_rate=0.15),
    EdgePrediction(neg_sampling_ratio=1.0)
])

# Train using train.py script or custom training loop
```

### Anomaly Detection

```python
from anomaly_detector import AnomalyDetector

# Load pre-trained model and create detector
detector = AnomalyDetector(
    pretrained_model=model,
    detection_method='neural'
)

# Fit on normal data
detector.fit(normal_graphs)

# Predict anomalies
predictions, scores = detector.predict(test_graphs)
metrics = detector.evaluate(test_graphs, test_labels)
```

## Configuration Options

### Model Configuration
- `input_dim`: Input feature dimension
- `hidden_dims`: List of hidden layer dimensions
- `output_dim`: Output embedding dimension
- `encoder_type`: GNN architecture ('gcn', 'gat', 'sage')
- `dropout`: Dropout rate for regularization

### Training Configuration
- `learning_rate`: Learning rate (default: 0.001)
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `early_stopping`: Enable early stopping
- `save_every`: Checkpoint saving frequency

### Graph Construction
- `window_size`: Token co-occurrence window size
- `min_token_freq`: Minimum token frequency threshold
- `max_vocab_size`: Maximum vocabulary size
- `token_pattern`: Regex pattern for tokenization

## Benchmark Datasets

The framework supports evaluation on standard log anomaly detection benchmarks:

- **HDFS**: Hadoop Distributed File System logs
- **BGL**: Blue Gene/L supercomputer logs  
- **Thunderbird**: Thunderbird supercomputer logs

Dataset files should be placed in the following structure:
```
data/
├── hdfs/
│   ├── HDFS.log
│   └── HDFS_labels.txt
├── bgl/
│   └── BGL.log
└── thunderbird/
    └── Thunderbird.log
```

## Evaluation Metrics

The framework computes comprehensive evaluation metrics:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Ranking Metrics**: AUC-ROC, AUC-PR
- **Representation Quality**: Embedding variance, cosine similarity statistics
- **SSL Task Performance**: Task-specific losses and accuracies

## Extending the Framework

### Adding New SSL Tasks

```python
from ssl_tasks import SSLTask

class CustomSSLTask(SSLTask):
    def create_task(self, data):
        # Implement task creation logic
        return modified_data, targets
    
    def compute_loss(self, predictions, targets):
        # Implement loss computation
        return loss
```

### Adding New GNN Architectures

```python
class CustomGNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # Implement custom architecture
    
    def forward(self, x, edge_index):
        # Implement forward pass
        return embeddings
```

### Adding New Detection Methods

```python
from anomaly_detector import AnomalyDetector

class CustomDetector(AnomalyDetector):
    def __init__(self, pretrained_model):
        super().__init__(pretrained_model, detection_method='custom')
        # Initialize custom detector
    
    def fit(self, normal_data):
        # Implement fitting logic
        pass
    
    def predict(self, test_data):
        # Implement prediction logic
        return predictions, scores
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Poor Performance**: Increase training epochs or adjust learning rate
3. **Graph Construction Errors**: Check tokenization patterns and vocabulary settings

### Performance Optimization

- Use GPU acceleration when available
- Adjust batch size based on memory constraints
- Use mixed precision training for larger models
- Consider graph sampling for very large graphs

## Citation

If you use LogGraph-SSL in your research, please cite:

```bibtex
@article{loggraph-ssl,
  title={LogGraph-SSL: Parsing-free Anomaly Detection in Distributed System Logs via Graph Neural Networks and Self-Supervised Learning},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- PyTorch Geometric team for the excellent graph neural network library
- The log anomaly detection research community for benchmark datasets
- Open source contributors and maintainers
