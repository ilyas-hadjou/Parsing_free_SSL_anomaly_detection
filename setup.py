#!/usr/bin/env python3
"""
Setup and validation script for LogGraph-SSL framework.
Checks dependencies and runs basic functionality tests.
"""

import sys
import subprocess
import importlib
import torch
import os


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'tqdm', 'networkx', 'regex'
    ]
    
    optional_packages = [
        'torch_geometric', 'transformers', 'tokenizers'
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_required.append(package)
    
    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional, install for full functionality)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    return True


def check_torch_functionality():
    """Check PyTorch functionality."""
    print("\nChecking PyTorch functionality...")
    
    try:
        # Test basic tensor operations
        x = torch.randn(10, 5)
        y = torch.mm(x, x.t())
        print("✅ Basic tensor operations")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available (devices: {torch.cuda.device_count()})")
            device = torch.device('cuda')
            x_cuda = x.to(device)
            print("✅ GPU tensor operations")
        else:
            print("⚠️  CUDA not available (CPU only)")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality error: {e}")
        return False


def test_framework_components():
    """Test basic framework components."""
    print("\nTesting framework components...")
    
    try:
        # Test imports
        from log_graph_builder import LogGraphBuilder
        from gnn_model import LogGraphSSL
        from ssl_tasks import MaskedNodePrediction, EdgePrediction
        from utils import create_sample_log_data, set_seed
        print("✅ Framework imports")
        
        # Test sample data creation
        messages, labels = create_sample_log_data(num_samples=10)
        if len(messages) == 10:
            print("✅ Sample data creation")
        else:
            print("❌ Sample data creation failed")
            return False
        
        # Test graph building
        builder = LogGraphBuilder(window_size=3, min_token_freq=1, max_vocab_size=100)
        graph = builder.build_graph_from_logs(messages[:5])
        if graph.num_nodes > 0:
            print("✅ Graph construction")
        else:
            print("❌ Graph construction failed")
            return False
        
        # Test model initialization
        model = LogGraphSSL(
            input_dim=graph.x.size(1),
            hidden_dims=[16, 8],
            output_dim=4,
            encoder_type='gcn'
        )
        print("✅ Model initialization")
        
        # Test forward pass
        with torch.no_grad():
            embeddings = model(graph.x, graph.edge_index)
            if embeddings.shape[0] == graph.num_nodes and embeddings.shape[1] == 4:
                print("✅ Model forward pass")
            else:
                print("❌ Model forward pass failed")
                return False
        
        # Test SSL tasks
        mask_task = MaskedNodePrediction(mask_rate=0.2)
        masked_data, targets = mask_task.create_task(graph)
        print("✅ SSL task creation")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework component error: {e}")
        return False


def create_directory_structure():
    """Create necessary directory structure."""
    print("\nCreating directory structure...")
    
    directories = [
        'outputs',
        'data',
        'checkpoints',
        'logs',
        'results'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/")
        else:
            print(f"✅ {directory}/ already exists")


def run_example_test():
    """Run the example script to test full functionality."""
    print("\nRunning example test...")
    
    try:
        from example import simple_example
        simple_example()
        print("✅ Example test completed successfully")
        return True
    except Exception as e:
        print(f"❌ Example test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("LogGraph-SSL Framework Setup")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("PyTorch", check_torch_functionality),
        ("Framework Components", test_framework_components),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"\n❌ {check_name} check failed!")
            break
        print(f"✅ {check_name} check passed")
    
    if all_passed:
        create_directory_structure()
        
        print("\n" + "=" * 40)
        print("✅ Setup completed successfully!")
        print("\nYou can now:")
        print("1. Run example.py for a simple demonstration")
        print("2. Use main.py create-data to generate sample data")
        print("3. Use train.py to train models")
        print("4. Use evaluate.py to evaluate models")
        print("5. Check README.md for detailed instructions")
        
        # Offer to run example
        try:
            response = input("\nRun example test? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_example_test()
        except KeyboardInterrupt:
            print("\nSetup completed.")
    
    else:
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
