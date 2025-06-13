#!/usr/bin/env python3
"""Quick test to check if conditional model is learning."""

import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Check for checkpoint
checkpoint_dir = Path('checkpoints/conditional_faithful')
checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))

if not checkpoints:
    print("No checkpoints found yet. Training might still be in early stages.")
    # Check metrics
    metrics_file = checkpoint_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        if metrics['train_loss']:
            print(f"\nCurrent status:")
            print(f"Epochs completed: {len(metrics['train_loss'])}")
            print(f"Latest loss: {metrics['train_loss'][-1][1]:.4f}")
            if len(metrics['train_loss']) > 1:
                print(f"Loss reduction: {metrics['train_loss'][0][1]:.4f} → {metrics['train_loss'][-1][1]:.4f}")
else:
    latest = checkpoints[-1]
    print(f"Found checkpoint: {latest}")
    checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
    print(f"Epoch: {checkpoint['epoch'] + 1}")
    print(f"Loss: {checkpoint['loss']:.4f}")
    
    # Plot if we have metrics
    if 'metrics' in checkpoint and checkpoint['metrics']['train_loss']:
        losses = checkpoint['metrics']['train_loss']
        epochs = [e[0] + 1 for e in losses]
        loss_values = [e[1] for e in losses]
        
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_values, 'b-', linewidth=2, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Conditional DDPM Training Progress')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('conditional_training_progress.png')
        print(f"\nSaved training plot to: conditional_training_progress.png")