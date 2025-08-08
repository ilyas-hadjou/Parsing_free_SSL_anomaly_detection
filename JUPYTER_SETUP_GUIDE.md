# LogGraph-SSL JupyterLab Quick Start Guide

## 🚀 One-Line Setup (For Advanced Users)

```bash
git clone https://github.com/ilyas-hadjou/Parsing_free_SSL_anomaly_detection.git && cd Parsing_free_SSL_anomaly_detection && chmod +x setup_jupyter.sh && ./setup_jupyter.sh
```

## 🚀 Getting Started with High-Performance Training

This guide will help you set up and run the LogGraph-SSL high-performance training notebook on your 24GB GPU server.

## 📋 Prerequisites

- **Hardware**: Server with 24GB+ GPU (Tesla V100, RTX 3090/4090, A100, etc.)
- **Software**: 
  - Python 3.8+
  - CUDA 11.8+ drivers
  - JupyterLab environment
  - SSH access to your server

## � Working with GitHub

### Initial Setup
```bash
# SSH into your server
ssh user@your-server-ip

# Clone the repository
git clone https://github.com/ilyas-hadjou/Parsing_free_SSL_anomaly_detection.git
cd Parsing_free_SSL_anomaly_detection

# Run setup
./setup_jupyter.sh
```

### Getting Updates
```bash
# Pull latest changes
git pull origin main

# If you have local changes, stash them first
git stash
git pull origin main
git stash pop
```

### Checking Repository Status
```bash
# Check current status
git status

# See commit history
git log --oneline -10

# Check remote URL
git remote -v
```

## �🔧 Setup Steps

### 1. Clone the Repository to Your Server

Clone the GitHub repository directly to your server:
```bash
# SSH into your server first
ssh user@your-server-ip

# Clone the repository
git clone https://github.com/ilyas-hadjou/Parsing_free_SSL_anomaly_detection.git

# Navigate to the project directory
cd Parsing_free_SSL_anomaly_detection

# Verify all files are present
ls -la
```

This will download all necessary files:
```bash
# Core notebook and scripts
LogGraph_SSL_HighPerformance_Training.ipynb
setup_jupyter.sh
requirements_jupyter.txt

# Model and utility files
gnn_model.py
log_graph_builder.py
ssl_tasks.py
utils.py

# Data files
hdfs_full_train.txt
hdfs_full_test.txt
hdfs_full_train_labels.txt
hdfs_full_test_labels.txt
```

### 2. Run Setup Script

```bash
# Make script executable and run
chmod +x setup_jupyter.sh
./setup_jupyter.sh
```

This script will:
- ✅ Check your GPU and CUDA setup
- 📦 Install all required Python packages
- 🔧 Set up virtual environment
- ✅ Verify installation

### 3. Start JupyterLab

```bash
# Activate environment
source venv_jupyter/bin/activate

# Start JupyterLab (accessible via browser)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 4. Open the Notebook

- Open your browser and navigate to the JupyterLab URL
- Open `LogGraph_SSL_HighPerformance_Training.ipynb`

## 📊 Running the Notebook

### Cell Execution Order

**IMPORTANT**: Run cells sequentially! Each section builds on the previous one.

1. **Section 10: Installation & Setup Verification**
   - ⚠️ **RUN THIS FIRST**
   - Verifies all dependencies are installed
   - Checks GPU availability and memory

2. **Section 1: Environment Setup**
   - Configures GPU memory management
   - Sets up CUDA optimization flags

3. **Section 2: Import Libraries**
   - Imports all required packages
   - May take 1-2 minutes on first run

4. **Section 3: Data Loading**
   - Loads complete HDFS dataset
   - Builds vocabulary and preprocesses data
   - **Time**: ~5-10 minutes

5. **Section 4: Graph Construction**
   - Creates co-occurrence graphs
   - **Time**: ~10-15 minutes

6. **Section 5: Model Architecture**
   - Initializes GNN models (GCN, GAT, GraphSAGE)
   - Shows model parameter counts

7. **Section 6: Training Configuration**
   - Sets up optimizers and schedulers
   - Configures SSL task manager

8. **Section 7: Training Loop**
   - 🔥 **Main Training** - This is the big one!
   - **Time**: 2-4 hours for 50 epochs
   - Monitor GPU memory usage

9. **Section 8: Evaluation**
   - Comprehensive performance evaluation
   - **Time**: ~10-20 minutes

10. **Section 9: Visualization**
    - Interactive training dashboards
    - Embedding visualizations

11. **Section 10: Model Saving**
    - Saves trained model for deployment

## 📈 Monitoring Training

### Real-Time Monitoring

The notebook includes built-in monitoring:
- 📊 Training loss curves
- 🔥 GPU memory usage
- ⏱️ Training speed and ETA
- 📈 Learning rate scheduling

### External Monitoring

In a separate terminal:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### Expected Performance

With 24GB GPU:
- **Batch Size**: Effective 256 (64 × 4 accumulation)
- **Training Time**: 2-4 hours for full dataset
- **Memory Usage**: ~18-20GB peak
- **Expected Accuracy**: >95% on HDFS dataset

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in TRAINING_CONFIG
   'batch_size': 32,  # Instead of 64
   'accumulation_steps': 8,  # Instead of 4
   ```

2. **Import Errors**
   - Re-run the setup script
   - Check Python version (need 3.8+)
   - Verify virtual environment activation

3. **Slow Training**
   - Check GPU utilization with `nvidia-smi`
   - Ensure CUDA drivers are properly installed
   - Verify TF32 is enabled (automatic on Ampere GPUs)

4. **Missing Data Files**
   - Ensure all HDFS data files are present
   - Check file paths in the data loading section

### Performance Optimization

For maximum performance:
```python
# In the environment setup section
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## 📊 Expected Results

After successful training, you should see:
- **Test Accuracy**: ~95-98%
- **F1-Score**: ~0.92-0.96
- **AUC-ROC**: ~0.97-0.99

## 💾 Saving Results

The notebook automatically saves:
- ✅ Trained model checkpoints
- 📊 Training history and metrics
- 🎯 Evaluation results
- 📈 Visualization plots

## 🎯 Next Steps

After training:
1. **Model Deployment**: Use the saved model for inference
2. **Hyperparameter Tuning**: Experiment with different configurations
3. **Architecture Comparison**: Try different GNN encoders (GCN vs GAT vs GraphSAGE)
4. **Dataset Expansion**: Apply to other log datasets

## 💡 Tips for Success

1. **Start Small**: Test with a subset first to verify everything works
2. **Monitor Closely**: Watch GPU memory and training curves
3. **Save Frequently**: The notebook auto-saves checkpoints
4. **Be Patient**: Full training takes several hours
5. **Document Changes**: Keep notes of any modifications

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Re-run the setup verification cell
4. Check GPU memory and system resources

---

**Happy Training! 🚀**
