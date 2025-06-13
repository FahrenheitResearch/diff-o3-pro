#!/usr/bin/env python3
"""Quick test to verify residual model architecture."""

import torch
import numpy as np
from models.unet_residual import UNetResidual
from utils.losses import WeatherLoss

# Test settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on device: {device}")

# Create model
model = UNetResidual(
    in_ch=7,  # 7 variables
    out_ch=7,
    base_features=32,  # Smaller for testing
    use_temporal_encoding=True
).to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
batch_size = 2
x = torch.randn(batch_size, 7, 256, 256).to(device)
timestamps = torch.tensor([100.0, 200.0]).to(device)

print("\nTesting forward pass...")
y = model(x, timestamps)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")

# Check residual property
residuals = y - x
print(f"\nResidual statistics:")
print(f"  Mean: {residuals.mean().item():.6f}")
print(f"  Std: {residuals.std().item():.6f}")
print(f"  Max: {residuals.abs().max().item():.6f}")

# Test that initial model predicts near-identity
print("\nInitial prediction should be near-identity (small residuals)")
assert residuals.abs().max() < 1.0, "Initial residuals too large!"
print("✓ Residual connection working correctly")

# Test loss function
print("\nTesting WeatherLoss...")
criterion = WeatherLoss()

target = x + torch.randn_like(x) * 0.1  # Small perturbation
loss, components = criterion(y, target)

print(f"Total loss: {loss.item():.4f}")
print("Loss components:")
for k, v in components.items():
    if k != 'total':
        print(f"  {k}: {v:.4f}")

# Test gradient flow
print("\nTesting gradient flow...")
loss.backward()

# Check that gradients flow to all parameters
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        if grad_norm == 0:
            print(f"WARNING: Zero gradient in {name}")

print(f"✓ Gradients flowing to {len(grad_norms)} parameters")
print(f"  Mean grad norm: {np.mean(grad_norms):.6f}")
print(f"  Max grad norm: {np.max(grad_norms):.6f}")

print("\n✓ All tests passed! Model architecture is working correctly.")