"""
Quick script to check model configuration and run evaluation with correct dimensions.
"""

import torch
import os
import json
from gnn_model import LogGraphSSL

def check_model_config(model_path):
    """Check the actual model configuration from saved state dict."""
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Check dimensions from the state dict
    encoder_weight = state_dict['encoder.convs.0.lin.weight']
    input_dim = encoder_weight.size(1)
    hidden_dim = encoder_weight.size(0)
    
    # Check output dimension from masked node head
    output_weight = state_dict['masked_node_head.3.weight']
    output_dim = output_weight.size(1)
    
    print(f"Detected model configuration:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: {output_dim}")
    
    return {
        'input_dim': input_dim,
        'hidden_dims': [hidden_dim, output_dim],
        'output_dim': output_dim,
        'encoder_type': 'gat',
        'num_heads': 4,
        'dropout': 0.3
    }

# Check the model configuration
model_dir = "./outputs/memory_efficient_ssl_20250807_172355"
model_path = os.path.join(model_dir, 'memory_efficient_model.pth')

if os.path.exists(model_path):
    config = check_model_config(model_path)
    
    # Save the configuration
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Configuration saved to model_config.json")
    
    # Test loading the model
    model = LogGraphSSL(**config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("Model loaded successfully!")
    
else:
    print(f"Model not found at {model_path}")
