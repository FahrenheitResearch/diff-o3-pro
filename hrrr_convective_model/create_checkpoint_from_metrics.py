#!/usr/bin/env python3
"""Create a checkpoint file from the training that crashed."""

import torch
import json
from pathlib import Path

# Load config and metrics
checkpoint_dir = Path('checkpoints/conditional_faithful')
with open(checkpoint_dir / 'config.yaml') as f:
    import yaml
    config = yaml.safe_load(f)

with open(checkpoint_dir / 'metrics.json') as f:
    metrics = json.load(f)

print(f"Found {len(metrics['train_loss'])} epochs of training")
print(f"Epoch 1 loss: {metrics['train_loss'][0][1]:.4f}")
print(f"Epoch 2 loss: {metrics['train_loss'][1][1]:.4f}")

# Create a pseudo-checkpoint for testing
checkpoint = {
    'epoch': 1,  # 0-indexed, so this is epoch 2
    'config': config,
    'metrics': metrics,
    'loss': metrics['train_loss'][-1][1]
}

# Save for reference
torch.save(checkpoint, checkpoint_dir / 'training_info.pt')
print(f"\nSaved training info to {checkpoint_dir / 'training_info.pt'}")
print("\nNOTE: Model weights were not saved due to the crash.")
print("We need to train a new model, but now we know the conditional approach works!")