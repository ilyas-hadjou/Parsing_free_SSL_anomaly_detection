#!/usr/bin/env python3
"""
Quick validation script for LogGraph-SSL setup
Run this before starting the full training to ensure everything works.
"""

import sys
import time
import torch
import traceback
from pathlib import Path

def test_basic_imports():
    """Test if all basic packages can be imported."""
    print("🔍 Testing basic imports...")
    
    required_packages = [
        'torch', 'torch_geometric', 'numpy', 'pandas', 
        'matplotlib', 'plotly', 'sklearn', 'tqdm', 'umap'
    ]
    
    failed = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed.append(package)
    
    return len(failed) == 0

def test_cuda_setup():
    """Test CUDA setup and GPU memory."""
    print("\n🔥 Testing CUDA setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✅ Found {device_count} GPU(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        if memory_gb < 8:
            print(f"⚠️  GPU {i} has low memory (<8GB)")
    
    # Test basic GPU operations
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.mm(x, y)
        print("✅ Basic GPU operations working")
        del x, y, z
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

def test_model_creation():
    """Test if we can create a simple model."""
    print("\n🧠 Testing model creation...")
    
    try:
        # Import custom modules
        from gnn_model import LogGraphSSL, AnomalyDetectionHead
        
        # Create a small test model
        model = LogGraphSSL(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=32,
            encoder_type='gat'
        )
        
        anomaly_head = AnomalyDetectionHead(input_dim=32)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        anomaly_head = anomaly_head.to(device)
        
        # Test forward pass
        num_nodes = 100
        x = torch.randn(num_nodes, 64, device=device)
        edge_index = torch.randint(0, num_nodes, (2, 200), device=device)
        
        embeddings = model(x, edge_index)
        scores = anomaly_head(embeddings)
        
        print(f"✅ Model created successfully")
        print(f"  Input shape: {x.shape}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Score shape: {scores.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test if we can load a small sample of data."""
    print("\n📁 Testing data loading...")
    
    # Check if we're in git repository
    if Path('.git').exists():
        print("✅ In Git repository")
    else:
        print("ℹ️  Not in Git repository (that's okay)")
    
    required_files = [
        'hdfs_full_train.txt',
        'hdfs_full_test.txt',
        'hdfs_full_train_labels.txt',
        'hdfs_full_test_labels.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"❌ Missing: {file}")
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} required files")
        print("💡 If you cloned from GitHub, ensure all files were downloaded:")
        print("   git status")
        print("   git pull origin main")
        return False
    
    # Test loading a few lines
    try:
        with open('hdfs_full_train.txt', 'r') as f:
            lines = [next(f) for _ in range(10)]  # Read first 10 lines
        
        with open('hdfs_full_train_labels.txt', 'r') as f:
            labels = [next(f) for _ in range(10)]
        
        print(f"✅ Successfully loaded sample data")
        print(f"  Sample line length: {len(lines[0])} characters")
        print(f"  Sample label: {labels[0].strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_memory_allocation():
    """Test if we can allocate significant GPU memory."""
    print("\n💾 Testing memory allocation...")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping (no CUDA)")
        return True
    
    try:
        device = torch.device('cuda')
        
        # Try to allocate ~2GB of memory
        size = 1024 * 1024 * 256  # 256M float32 = ~1GB
        x = torch.randn(size, device=device)
        
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        
        print(f"✅ Allocated {memory_allocated:.1f} GB / {memory_total:.1f} GB")
        
        del x
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Memory allocation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 LogGraph-SSL Validation Suite 🚀")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("CUDA Setup", test_cuda_setup),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Memory Allocation", test_memory_allocation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for training! 🎉")
        print("💡 You can now run the main notebook with confidence.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before training.")
        print("💡 Check the setup guide and ensure all dependencies are installed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
