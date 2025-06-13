#!/usr/bin/env python3
"""Test the forecast model directly without ensemble generation."""

import torch
import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path

from models.unet_attention_fixed import UNetAttn
from utils.normalization import Normalizer

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load checkpoint
checkpoint_path = Path('checkpoints/forecast_model_best.pt')
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']
print(f"Model trained for epoch: {checkpoint['epoch']}")

# Create model
model = UNetAttn(
    in_ch=len(config['data']['variables']),
    out_ch=len(config['data']['variables']),
    base_features=config['training']['base_features'],
    use_temporal_encoding=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
print("Model loaded")

# Load normalizer
normalizer = Normalizer(Path('data/stats.json'))

# Load test data
store = zarr.open('data/zarr/latest/hrrr.zarr', 'r')
variables = config['data']['variables']

# Get first time step
input_data = []
for var in variables:
    data = store[var][0].astype(np.float32)
    # Normalize
    data_norm = normalizer.encode(data, var)
    input_data.append(data_norm)

# Stack and convert to tensor
x = torch.tensor(np.stack(input_data, axis=0)).unsqueeze(0).to(device)
print(f"Input shape: {x.shape}")

# Generate dummy timestamp
timestamp = torch.tensor([0.0], device=device)

# Get direct model prediction
print("\nRunning model prediction...")
with torch.no_grad():
    y_pred = model(x, timestamp)

print(f"Output shape: {y_pred.shape}")

# Analyze the output
y_pred_np = y_pred.cpu().numpy()[0]  # Remove batch dimension

print("\nAnalyzing model output (normalized values):")
for i, var in enumerate(variables):
    data = y_pred_np[i]
    print(f"{var}:")
    print(f"  Min: {np.min(data):.3f}, Max: {np.max(data):.3f}")
    print(f"  Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
    
    # Check if output resembles input
    input_data = x.cpu().numpy()[0, i]
    diff = data - input_data
    print(f"  Diff from input - Mean: {np.mean(diff):.3f}, Std: {np.std(diff):.3f}")

# Denormalize and visualize REFC
print("\nVisualizing reflectivity...")
refc_idx = variables.index('REFC')
refc_pred = y_pred_np[refc_idx]
refc_input = x.cpu().numpy()[0, refc_idx]

# Denormalize
refc_pred_denorm = normalizer.decode(refc_pred, 'REFC')
refc_input_denorm = normalizer.decode(refc_input, 'REFC')

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Input
im1 = axes[0].imshow(refc_input_denorm, cmap='turbo', vmin=-10, vmax=60)
axes[0].set_title('Input REFC')
plt.colorbar(im1, ax=axes[0])

# Prediction
im2 = axes[1].imshow(refc_pred_denorm, cmap='turbo', vmin=-10, vmax=60)
axes[1].set_title('Model Prediction')
plt.colorbar(im2, ax=axes[1])

# Difference
diff = refc_pred_denorm - refc_input_denorm
im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20)
axes[2].set_title('Difference')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('test_forecast_output.png', dpi=150)
print("Saved visualization to test_forecast_output.png")

# Check if model is just adding noise
print(f"\nDifference statistics (dBZ):")
print(f"  Mean: {np.mean(diff):.2f}")
print(f"  Std: {np.std(diff):.2f}")
print(f"  Min: {np.min(diff):.2f}")
print(f"  Max: {np.max(diff):.2f}")