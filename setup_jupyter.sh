#!/bin/bash

# LogGraph-SSL JupyterLab Setup Script
# This script helps set up the environment for high-performance techo "🔧🐛 If you encounter issues:"
echo "- Check CUDA compatibility"
echo "- Verify all files are present"
echo "- Ensure sufficient disk space (>10GB recommended)"
echo "- Check the TRAINING_GUIDE.md for troubleshooting"ng

echo "🚀 LogGraph-SSL JupyterLab Setup Script 🚀"
echo "============================================="

# Check if we're in a git repository
if [ -d ".git" ]; then
    echo "✅ Running in Git repository"
    git_remote=$(git remote get-url origin 2>/dev/null || echo "No remote")
    echo "📡 Remote: $git_remote"
    
    # Check if there are any uncommitted changes
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "⚠️  Warning: You have uncommitted changes"
    fi
else
    echo "ℹ️  Not in a Git repository"
fi

# Check if we're in the right directory
if [ ! -f "LogGraph_SSL_Complete_Training.ipynb" ]; then
    echo "❌ Error: LogGraph_SSL_Complete_Training.ipynb not found in current directory"
    echo ""
    echo "💡 If you haven't cloned the repository yet, run:"
    echo "git clone https://github.com/ilyas-hadjou/Parsing_free_SSL_anomaly_detection.git"
    echo "cd Parsing_free_SSL_anomaly_detection"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Found notebook file"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected. Training will use CPU (much slower)."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_jupyter" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_jupyter
fi

echo "🔧 Activating virtual environment..."
source venv_jupyter/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo "🌐 Installing PyTorch Geometric..."
pip install torch-geometric

# Install other requirements
echo "📚 Installing additional requirements..."
pip install -r requirements_complete.txt

# Install Jupyter extensions
echo "🔧 Installing Jupyter extensions..."
pip install jupyter-dash
pip install ipywidgets

# Try to install lab extension (may fail in some environments - non-critical)
echo "📦 Attempting to install JupyterLab widget manager..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager 2>/dev/null || echo "⚠️  JupyterLab extension install failed (non-critical - widgets will still work)"

# Verify installations
echo "✅ Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
fi

python3 -c "import torch_geometric; print(f'PyTorch Geometric version: {torch_geometric.__version__}')"
python3 -c "import plotly; print(f'Plotly version: {plotly.__version__}')"
python3 -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

echo ""
echo "🎉 Setup completed successfully! 🎉"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source venv_jupyter/bin/activate"
echo "2. Start JupyterLab: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "3. Open LogGraph_SSL_Complete_Training.ipynb"
echo "4. Run cells sequentially starting from the first cell"
echo ""
echo "🌐 Access JupyterLab from your browser:"
echo "http://your-server-ip:8888 (replace with your actual server IP)"
echo ""
echo "💡 Tips:"
echo "- Monitor GPU usage with: nvidia-smi"
echo "- The notebook includes real-time monitoring dashboards"
echo "- Training will take 2-4 hours on a 24GB GPU"
echo "- Run local validation first: python3 test_notebook_locally.py"
echo ""
echo "� To update the code from GitHub:"
echo "git pull origin main"
echo ""
echo "�🐛 If you encounter issues:"
echo "- Check CUDA compatibility"
echo "- Verify all files are present"
echo "- Ensure sufficient disk space (>10GB recommended)"
echo "- Check the JUPYTER_SETUP_GUIDE.md for troubleshooting"
