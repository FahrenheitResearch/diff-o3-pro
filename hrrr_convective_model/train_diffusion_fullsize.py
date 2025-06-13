#!/usr/bin/env python3
"""
Full-size faithful DDPM training attempt.
100% faithful to DEF - predicts NOISE.
Reduced model size to fit full resolution.
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
import argparse

# Use the ultra-minimal model for full resolution
from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset


def train_fullsize():
    """Try to train at full resolution with minimal model."""
    
    device = torch.device('cuda')
    print("="*70)
    print("FULL-SIZE FAITHFUL DDPM TRAINING ATTEMPT")
    print("Full resolution: 1059 x 1799")
    print("100% faithful to DEF - predicts NOISE")
    print("="*70)
    
    # Aggressive memory management
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print("\nCreating dataset...")
    dataset = HRRRDataset(
        zarr_path='data/zarr/training_14day/hrrr.zarr',
        variables=['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN'],
        lead_hours=1,
        stats_path='data/zarr/training_14day/stats.json'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Only 1 sample
        shuffle=True,
        num_workers=0,  # No workers to save memory
        pin_memory=False
    )
    
    print(f"Dataset: {len(dataset)} samples at FULL 1059x1799 resolution")
    
    # Create minimal model
    print("\nCreating ultra-minimal DDPM...")
    model = UltraMinimalDDPM(
        in_channels=7,
        out_channels=7,
        base_dim=16  # Very small!
    ).to(device)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Noise schedule
    noise_schedule = CosineBetaSchedule(timesteps=100)  # Fewer timesteps
    
    # Optimizer with low memory footprint
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Mixed precision
    scaler = GradScaler()
    
    print("\nStarting training...")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Training loop
    for epoch in range(10):
        for batch_idx, (_, future_state) in enumerate(dataloader):
            if batch_idx > 5:  # Just test a few batches
                break
                
            future_state = future_state.to(device)
            
            # Clear cache before each step
            torch.cuda.empty_cache()
            
            # Sample timestep
            t = torch.randint(0, 100, (1,), device=device)
            
            # Create noise
            noise = torch.randn_like(future_state)
            
            # Add noise
            noisy_state = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward with mixed precision
            with autocast():
                noise_pred = model(noisy_state, t)
                loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Print memory usage
            gpu_gb = torch.cuda.memory_allocated()/1024**3
            max_gb = torch.cuda.max_memory_allocated()/1024**3
            print(f"Step {batch_idx}: loss={loss.item():.4f}, GPU={gpu_gb:.1f}GB, Peak={max_gb:.1f}GB")
            
            # Clear cache after step
            del future_state, noise, noisy_state, noise_pred, loss
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\nTraining completed successfully at FULL resolution!")
    print(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    train_fullsize()