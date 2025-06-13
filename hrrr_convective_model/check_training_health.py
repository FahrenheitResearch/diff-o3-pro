#!/usr/bin/env python3
"""Quick diagnostic of training health."""

import torch
from pathlib import Path

# Check if we have any checkpoints
checkpoint_dir = Path('checkpoints/deterministic')
checkpoints = list(checkpoint_dir.glob('*.pt'))

if checkpoints:
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest}")
    
    ckpt = torch.load(latest, map_location='cpu')
    model_dict = ckpt['model_state_dict']
    
    # Check residual scale
    if 'residual_scale' in model_dict:
        print(f"\nResidual scale: {model_dict['residual_scale'].item()}")
    
    # Check output layer weights
    if 'outc.weight' in model_dict:
        outc_weight = model_dict['outc.weight']
        print(f"\nOutput layer stats:")
        print(f"  Mean: {outc_weight.mean().item():.6f}")
        print(f"  Std: {outc_weight.std().item():.6f}")
        print(f"  Max: {outc_weight.abs().max().item():.6f}")
    
    # Check a few other layer statistics
    print("\nOther layer weight statistics:")
    for name, param in model_dict.items():
        if 'weight' in name and param.dim() >= 2:
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            if param.std().item() < 1e-6:
                print(f"    WARNING: Near-zero weights!")
else:
    print("No checkpoints found yet")

# Check current training status
print("\n" + "="*50)
print("Training appears stuck because:")
print("1. Loss is flat (0.0937-0.0938)")
print("2. Learning rate might be too low (warmup starting at 0.0001)")
print("3. Model might be stuck at identity mapping")
print("\nRecommendations:")
print("1. Increase initial learning rate to 0.001 or 0.002")
print("2. Reduce warmup epochs to 2")
print("3. Initialize residual_scale to 0.01 instead of 0.1")
print("4. Consider reducing gradient accumulation to 8")