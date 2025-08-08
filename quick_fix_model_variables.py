# Quick Fix: Define Missing Model Variables
# Add this cell before the error cell in your current notebook

print("🔧 Creating missing model variables...")

# Create configuration
TRAINING_CONFIG = {
    'vocab_size': 5000,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'encoder_type': 'gcn',
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'ssl_epochs': 50,
    'supervised_epochs': 30
}

# Create the LogGraphSSL model
primary_model = LogGraphSSL(
    vocab_size=TRAINING_CONFIG['vocab_size'],
    embedding_dim=TRAINING_CONFIG['embedding_dim'],
    hidden_dim=TRAINING_CONFIG['hidden_dim'],
    encoder_type=TRAINING_CONFIG['encoder_type'],
    num_layers=TRAINING_CONFIG['num_layers'],
    dropout=TRAINING_CONFIG['dropout']
).to(device)

# Get the anomaly detection head
primary_anomaly_head = primary_model.anomaly_head

print("✅ Model variables created successfully!")
print(f"📊 Model parameters: {sum(p.numel() for p in primary_model.parameters()):,}")
print(f"🔧 Device: {next(primary_model.parameters()).device}")
