"""
Comprehensive SSL model evaluation and comparison setup.
Monitors training progress and prepares for final comparison.
"""

import os
import time
import json
import glob
from datetime import datetime

def monitor_training_progress():
    """Monitor the training progress and show status."""
    print("🔍 Monitoring LogGraph-SSL Full Dataset Training...")
    print("=" * 60)
    
    # Check for latest training output
    output_dirs = glob.glob("./outputs/full_dataset_ssl_*")
    if output_dirs:
        latest_dir = max(output_dirs, key=os.path.getctime)
        print(f"Latest training directory: {latest_dir}")
        
        # Check if training is complete
        model_file = os.path.join(latest_dir, 'full_dataset_model.pth')
        history_file = os.path.join(latest_dir, 'training_history.json')
        
        if os.path.exists(model_file) and os.path.exists(history_file):
            print("✅ Training completed!")
            
            # Load and display results
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            print(f"\n📊 Final Training Results:")
            print(f"  - Total epochs: {len(history['edge_aucs'])}")
            print(f"  - Final Edge AUC: {history['edge_aucs'][-1]:.4f}")
            print(f"  - Best Edge AUC: {max(history['edge_aucs']):.4f}")
            print(f"  - Final Node Accuracy: {history['node_accs'][-1]:.4f}")
            print(f"  - Final Embedding Variance: {history['embedding_vars'][-1]:.6f}")
            
            return latest_dir
        else:
            print("⏳ Training still in progress...")
            return None
    else:
        print("❌ No training directories found.")
        return None

def show_comparison_commands(model_dir):
    """Show commands for comprehensive comparison."""
    print("\n🚀 Ready for Comprehensive Evaluation!")
    print("=" * 60)
    
    print("\n📋 Available Evaluation Commands:")
    print("\n1. **Comprehensive SSL Evaluation:**")
    print(f"   python evaluate_comprehensive.py --model_dir {model_dir} --test_data hdfs_full_test.txt --test_labels hdfs_full_test_labels.txt")
    
    print("\n2. **SSL vs Traditional Methods Comparison:**")
    print(f"   python compare_methods.py --model_dir {model_dir} --test_data hdfs_full_test.txt --test_labels hdfs_full_test_labels.txt --sample_size 15000")
    
    print("\n3. **Full Pipeline Summary:**")
    print("   ./run_complete_pipeline.sh")
    
    print("\n📁 Available Models:")
    print(f"   - Full Dataset Model: {model_dir}/full_dataset_model.pth")
    print(f"   - Graph Builder: {model_dir}/graph_builder.pth")
    print(f"   - Model Config: {model_dir}/model_config.json")

def main():
    """Main monitoring and setup function."""
    print(f"🎯 LogGraph-SSL Comprehensive Evaluation Setup")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Monitor training
    model_dir = monitor_training_progress()
    
    if model_dir:
        show_comparison_commands(model_dir)
        
        print("\n🎉 **EVALUATION SUMMARY**")
        print("✅ Full dataset training completed successfully")
        print("✅ Model ready for comprehensive evaluation")
        print("✅ Comparison framework prepared")
        print("✅ Ready to compare SSL vs traditional methods")
        
        print("\n🔬 **NEXT STEPS:**")
        print("1. Run comprehensive SSL evaluation")
        print("2. Execute SSL vs traditional comparison")
        print("3. Analyze performance differences")
        print("4. Generate final research report")
        
        return model_dir
    else:
        print("\n⏳ **WAITING FOR TRAINING COMPLETION**")
        print("Training is still in progress. Please wait...")
        print("You can run this script again to check status.")
        return None

if __name__ == "__main__":
    main()
