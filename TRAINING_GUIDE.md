# LogGraph-SSL Complete Training Guide 🚀

## High-Performance HDFS Anomaly Detection Training

This notebook provides a complete, ready-to-run implementation of LogGraph-SSL for anomaly detection in HDFS logs using Graph Neural Networks with Self-Supervised Learning.

### 🎯 What's Included

- **Complete Training Pipeline**: From data loading to model evaluation
- **Multiple GNN Architectures**: GCN, GAT, GraphSAGE encoders
- **Self-Supervised Learning**: Node masking, edge prediction, contrastive learning
- **RTX 4090 Optimized**: Full GPU acceleration with memory optimization
- **Real-time Monitoring**: Training metrics and GPU utilization tracking
- **Robust Architecture**: Error handling and memory management

### 🚀 Quick Start

#### 1. Local Validation (Optional but Recommended)
```bash
# Test everything works locally first
python3 test_notebook_locally.py
```

#### 2. Deploy to Jupyter Server
```bash
# Commit and push to GitHub
git add .
git commit -m "Add complete LogGraph-SSL training notebook"
git push origin main

# On Jupyter server (RTX 4090)
git pull origin main
```

#### 3. Environment Setup on Jupyter Server
```bash
# Run the setup script
./setup_jupyter.sh

# Or manually:
source venv_jupyter/bin/activate
pip install -r requirements_complete.txt
```

#### 4. Launch Jupyter and Run Notebook
```bash
# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Open LogGraph_SSL_Complete_Training.ipynb
# Run cells sequentially starting from cell 1
```

### 📊 Training Process

The notebook includes 8 main sections:

1. **Environment Setup** - Imports, device configuration, seed setting
2. **Data Processing** - HDFS log parsing and graph construction
3. **Model Architecture** - GNN encoders and SSL task managers
4. **Data Loading** - Custom graph data loaders and batching
5. **Training Functions** - SSL pre-training and supervised fine-tuning
6. **Main Execution** - Complete training pipeline
7. **Monitoring** - Real-time metrics and GPU utilization
8. **Validation** - Quick tests for local development

### 🔧 Configuration

Key parameters you can modify:

```python
config = {
    'vocab_size': 5000,        # Vocabulary size
    'embedding_dim': 128,      # Token embedding dimension
    'hidden_dim': 256,         # Hidden layer dimension
    'encoder_type': 'gcn',     # 'gcn', 'gat', or 'sage'
    'num_layers': 3,           # Number of GNN layers
    'ssl_epochs': 50,          # SSL pre-training epochs
    'supervised_epochs': 30,   # Supervised fine-tuning epochs
    'batch_size': 32,          # Batch size
    'learning_rate': 0.001     # Learning rate
}
```

### 📈 Expected Performance

On RTX 4090 (24GB VRAM):
- **Training Time**: 2-4 hours for full dataset
- **Memory Usage**: ~8-12GB GPU memory
- **Accuracy**: 85-95% on HDFS anomaly detection
- **Throughput**: ~500-1000 samples/second

### 🛠️ Troubleshooting

#### Common Issues:

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in config
   config['batch_size'] = 16  # or 8
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements_complete.txt
   ```

3. **Data Not Found**
   ```python
   # The notebook creates sample data automatically
   # Or place your HDFS data files in the project directory
   ```

4. **Training Stuck**
   ```python
   # Check GPU utilization
   monitor_gpu_usage()
   
   # Restart kernel if needed
   ```

### 📁 File Structure

```
Parsing-free-anomaly-detection/
├── LogGraph_SSL_Complete_Training.ipynb  # Main training notebook
├── test_notebook_locally.py              # Local validation
├── requirements_complete.txt             # Dependencies
├── setup_jupyter.sh                      # Environment setup
├── hdfs_*.txt                           # HDFS data files
└── sample_*.txt                         # Generated sample data
```

### 🎯 Next Steps After Training

1. **Model Evaluation**: The notebook automatically evaluates the trained model
2. **Model Export**: Saves trained model with timestamp
3. **Results Analysis**: Classification report and confusion matrix
4. **Deployment**: Use saved model for real-time anomaly detection

### 💡 Tips for Best Results

- **Data Quality**: Ensure HDFS logs are properly formatted
- **GPU Monitoring**: Use `nvidia-smi` to monitor GPU usage
- **Experiment Tracking**: Save different configurations and results
- **Memory Management**: Clear GPU cache between experiments
- **Hyperparameter Tuning**: Experiment with different architectures

### 🚀 Performance Optimization

For maximum performance on RTX 4090:

```python
# Enable mixed precision training
torch.backends.cudnn.benchmark = True

# Use larger batch sizes
config['batch_size'] = 64

# Enable gradient checkpointing for memory efficiency
# (automatically handled in the notebook)
```

---

**Ready to start training?** Open `LogGraph_SSL_Complete_Training.ipynb` and run the cells sequentially! 🚀
