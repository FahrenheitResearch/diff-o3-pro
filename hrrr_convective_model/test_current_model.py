#!/usr/bin/env python3
"""Test the current deterministic model to see if it's producing better predictions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime, timedelta

from models.unet_residual import UNetResidual
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer

def test_model():
    # Load config
    with open('configs/deterministic.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load normalizer
    normalizer = Normalizer(cfg['data']['stats'])
    
    # Create dataset
    dataset = HRRRDataset(
        zarr_path=cfg['data']['zarr'],
        variables=cfg['data']['variables'],
        lead_hours=cfg['training']['lead_hours'],
        stats_path=cfg['data']['stats']
    )
    
    # Load model
    model = UNetResidual(
        in_ch=len(cfg['data']['variables']),
        out_ch=len(cfg['data']['variables']),
        base_features=cfg['training']['base_features'],
        use_temporal_encoding=True
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path('checkpoints/deterministic/best_model.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    else:
        print("No checkpoint found, using random initialization")
    
    model.eval()
    
    # Test on a few samples
    test_indices = [0, 100, 200]  # Test different times
    
    for idx in test_indices:
        # Get sample
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        
        # Create timestamp (hours since epoch for Jan 1 2024 + sample index)
        epoch_start_hours = cfg['data'].get('epoch_start_hours', 473832)
        timestamp_x = torch.tensor([epoch_start_hours + idx]).to(device)
        
        # Predict
        with torch.no_grad():
            y_pred = model(x, timestamp_x, add_noise=False)
        
        # Move to CPU and denormalize
        x_cpu = x[0].cpu().numpy()
        y_cpu = y[0].cpu().numpy()
        y_pred_cpu = y_pred[0].cpu().numpy()
        
        # Plot results for REFC (reflectivity)
        var_idx = 0  # REFC
        var_name = cfg['data']['variables'][var_idx]
        
        # Denormalize
        x_denorm = normalizer.decode(x_cpu[var_idx], var_name)
        y_denorm = normalizer.decode(y_cpu[var_idx], var_name)
        y_pred_denorm = normalizer.decode(y_pred_cpu[var_idx], var_name)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot input
        im0 = axes[0].imshow(x_denorm, cmap='jet', vmin=0, vmax=60)
        axes[0].set_title(f'Input (t=0)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='dBZ')
        
        # Plot target
        im1 = axes[1].imshow(y_denorm, cmap='jet', vmin=0, vmax=60)
        axes[1].set_title(f'Target (t+{cfg["training"]["lead_hours"]}h)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='dBZ')
        
        # Plot prediction
        im2 = axes[2].imshow(y_pred_denorm, cmap='jet', vmin=0, vmax=60)
        axes[2].set_title(f'Prediction (t+{cfg["training"]["lead_hours"]}h)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='dBZ')
        
        # Plot error
        error = y_pred_denorm - y_denorm
        im3 = axes[3].imshow(error, cmap='RdBu_r', vmin=-20, vmax=20)
        axes[3].set_title('Prediction Error')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], label='dBZ')
        
        # Add timestamp
        hours_since_epoch = timestamp_x.item()
        dt = datetime(1970, 1, 1) + timedelta(hours=hours_since_epoch)
        fig.suptitle(f'Sample {idx} - {dt.strftime("%Y-%m-%d %H:%M UTC")}', fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path('test_outputs')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'test_sample_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"\nSample {idx}:")
        print(f"  Input range: [{x_denorm.min():.2f}, {x_denorm.max():.2f}]")
        print(f"  Target range: [{y_denorm.min():.2f}, {y_denorm.max():.2f}]")
        print(f"  Prediction range: [{y_pred_denorm.min():.2f}, {y_pred_denorm.max():.2f}]")
        print(f"  RMSE: {np.sqrt(np.mean((y_pred_denorm - y_denorm)**2)):.2f} dBZ")
        print(f"  MAE: {np.mean(np.abs(y_pred_denorm - y_denorm)):.2f} dBZ")

if __name__ == '__main__':
    test_model()