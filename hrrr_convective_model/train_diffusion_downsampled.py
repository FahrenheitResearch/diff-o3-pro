#!/usr/bin/env python3
"""
Train faithful DDPM at downsampled resolution to fit in memory.
Still 100% faithful to DEF - predicts NOISE.
Downsamples to 1/4 resolution (265x450) for training.
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
import argparse

from models.diffusion.ddpm_18gb import DDPM18GB, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def train_downsampled_diffusion(config_path):
    """Train DDPM at reduced resolution."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("DOWNSAMPLED FAITHFUL DDPM TRAINING")
    print("Training at 1/4 resolution (265x450) to fit in memory")
    print("Still 100% faithful to DEF - predicts NOISE")
    print("="*70 + "\n")
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create dataset
    print("Creating dataset...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=config['training']['lead_hours'],
        stats_path=config['data']['stats']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Can use bigger batch at lower resolution
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Original resolution: 1059 x 1799")
    print(f"Training resolution: 265 x 450 (1/4 scale)")
    
    # Create model
    print("\nCreating DDPM model...")
    model = DDPM18GB(
        in_channels=len(config['data']['variables']),
        out_channels=len(config['data']['variables']),
        base_dim=64
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Noise schedule
    print(f"Using cosine beta schedule with {config['diffusion']['timesteps']} timesteps")
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps'],
        s=config['diffusion']['s']
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    print("\nStarting training...")
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'diffusion_downsampled'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (current_state, future_state) in enumerate(pbar):
            # DOWNSAMPLE to 1/4 resolution
            B, C, H, W = future_state.shape
            future_state = F.interpolate(future_state, size=(H//4, W//4), mode='bilinear', align_corners=False)
            
            # Move to device
            future_state = future_state.to(device)
            batch_size = future_state.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, config['diffusion']['timesteps'], (batch_size,), device=device)
            
            # Create noise
            noise = torch.randn_like(future_state)
            
            # Add noise
            noisy_state = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward pass - predict noise (faithful!)
            noise_pred = model(noisy_state, t)
            
            # MSE loss on noise
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Update progress
            avg_loss = epoch_loss / batch_count
            gpu_mb = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'gpu_mb': f'{gpu_mb:.0f}'
            })
            
            # Memory management
            if batch_idx % 50 == 0:
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
                'config': config,
                'downsampled': True,
                'training_resolution': (265, 450)
            }
            checkpoint_path = checkpoint_dir / f'epoch_{epoch:04d}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                print("Saved as best model!")
        
        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\nGenerating sample...")
            model.eval()
            with torch.no_grad():
                # Start from noise
                x = torch.randn(1, 7, 265, 450).to(device)
                
                # Reverse diffusion
                for i in tqdm(reversed(range(config['diffusion']['timesteps'])), desc="Sampling"):
                    t = torch.tensor([i]).to(device)
                    
                    # Predict noise
                    noise_pred = model(x, t)
                    
                    # Simple DDPM sampling (can be improved with DDIM)
                    alpha_t = noise_schedule.alphas[i]
                    alpha_bar_t = noise_schedule.alphas_cumprod[i]
                    
                    if i > 0:
                        alpha_bar_t_1 = noise_schedule.alphas_cumprod[i-1]
                        beta_t = 1 - alpha_t
                        sigma_t = ((1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t).sqrt()
                        
                        # Remove predicted noise
                        x = (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_t.sqrt()
                        
                        # Add noise for next step
                        if i > 1:
                            x = x + sigma_t * torch.randn_like(x)
                    else:
                        # Final step
                        x = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
                
                print(f"Generated sample stats: min={x.min():.3f}, max={x.max():.3f}, std={x.std():.3f}")
            
            model.train()
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Final memory stats
    if torch.cuda.is_available():
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/diffusion_max.yaml')
    args = parser.parse_args()
    
    train_downsampled_diffusion(args.config)