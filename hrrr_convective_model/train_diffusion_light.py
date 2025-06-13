#!/usr/bin/env python3
"""
Lightweight faithful DEF training for RTX 4090.
Simplified to avoid OOM while maintaining correct architecture.
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
from datetime import datetime
import argparse

# Simplified model - just U-Net with noise prediction
from models.unet_residual import UNetResidualSmall
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


class SimpleDiffusionSchedule:
    """Simple linear beta schedule for diffusion."""
    def __init__(self, timesteps=100):
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, x0, noise, t):
        """Add noise to data."""
        device = x0.device
        
        # Move schedule tensors to device if needed
        if self.sqrt_alphas_cumprod.device != device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1, 1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


def train_light_diffusion(config_path):
    """Train lightweight diffusion model."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create simple dataset
    print("Creating dataset...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=config['training']['lead_hours'],
        stats_path=config['data']['stats']
    )
    
    # Use very simple dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Always 1 for memory
        shuffle=True,
        num_workers=0,  # No workers
        pin_memory=False  # Save memory
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create lightweight model
    print("Creating lightweight diffusion model...")
    model = UNetResidualSmall(
        in_ch=len(config['data']['variables']),
        out_ch=len(config['data']['variables']),
        use_temporal_encoding=False  # We handle time separately in diffusion
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple noise schedule
    noise_schedule = SimpleDiffusionSchedule(config['diffusion']['timesteps'])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Training loop
    print("\nStarting lightweight diffusion training...")
    print("Key: Model learns to predict NOISE (faithful to DEF)")
    
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'diffusion_light'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (current_state, future_state) in enumerate(pbar):
            # Move to device
            future_state = future_state.to(device)
            batch_size = future_state.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, config['diffusion']['timesteps'], (batch_size,), device=device)
            
            # Create noise
            noise = torch.randn_like(future_state)
            
            # Add noise to create noisy state
            noisy_state = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward pass - predict the noise
            noise_pred = model(noisy_state)
            
            # Simple MSE loss on noise (this is the key!)
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/batch_count:.4f}',
                'gpu_mb': f'{torch.cuda.memory_allocated()/1024**2:.0f}' if torch.cuda.is_available() else 'N/A'
            })
            
            # Clear cache periodically
            if batch_idx % config.get('memory', {}).get('empty_cache_interval', 50) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Epoch complete
        avg_loss = epoch_loss / batch_count
        print(f"\nEpoch {epoch}: Average Loss = {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Also save as best if it's the lowest loss
            if epoch == 0 or avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                print("Saved as best model!")
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Final memory stats
    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/diffusion_4090_light.yaml')
    args = parser.parse_args()
    
    # Set memory fraction to be safe
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of VRAM
    
    train_light_diffusion(args.config)