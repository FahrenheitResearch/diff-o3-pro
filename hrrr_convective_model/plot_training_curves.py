#!/usr/bin/env python3
"""Plot training curves from log files."""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def parse_log_file(log_file):
    """Extract training metrics from log file."""
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse training loss
            if "Epoch" in line and "Train loss:" in line:
                match = re.search(r'Epoch (\d+) - Train loss: ([\d.]+)', line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
            
            # Parse validation loss
            if "Validation - Loss:" in line:
                match = re.search(r'Loss: ([\d.]+)', line)
                if match:
                    val_losses.append(float(match.group(1)))
    
    return epochs, train_losses, val_losses

# Find latest log file
log_dir = Path('logs/deterministic')
log_files = list(log_dir.glob('training_*.log'))
if not log_files:
    print("No log files found!")
    exit(1)

latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
print(f"Parsing: {latest_log}")

epochs, train_losses, val_losses = parse_log_file(latest_log)

if not epochs:
    print("No training data found in log file yet")
    exit(0)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

# Add validation if available
if val_losses:
    val_epochs = [e for e in epochs if (e+1) % 2 == 0][:len(val_losses)]
    plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
output_file = 'training_curves.png'
plt.savefig(output_file, dpi=150)
print(f"Saved plot to: {output_file}")

# Print current status
print(f"\nTraining Status:")
print(f"  Current epoch: {epochs[-1] if epochs else 0}")
print(f"  Latest train loss: {train_losses[-1]:.4f}" if train_losses else "  No data yet")
if val_losses:
    print(f"  Latest val loss: {val_losses[-1]:.4f}")
    if len(val_losses) > 1:
        improvement = val_losses[-2] - val_losses[-1]
        print(f"  Val improvement: {improvement:+.4f}")