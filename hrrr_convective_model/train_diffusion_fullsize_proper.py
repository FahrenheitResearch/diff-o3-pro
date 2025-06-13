#!/usr/bin/env python3
"""
Full-size faithful DDPM training with proper setup.
100% faithful to DEF - predicts NOISE at full 1059x1799 resolution.
Uses larger model that still fits in memory.
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
from datetime import datetime
import argparse

from models.diffusion.ddpm_fullsize import DDPMFullsize, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def train_fullsize_ddpm(config_path):
    """Train faithful DDPM at full resolution."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("FULL-SIZE FAITHFUL DDPM TRAINING")
    print("Resolution: 1059 x 1799 (3km HRRR)")
    print("100% faithful to DEF - predicts NOISE")
    print("="*70 + "\n")
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
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
        batch_size=1,  # Always 1 for full resolution
        shuffle=True,
        num_workers=0,  # No workers to save memory
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Full resolution: 1059 x 1799")
    
    # Create model - 11.4M parameters at full resolution!
    print("\nCreating 11.4M parameter DDPM model...")
    model = DDPMFullsize(
        in_channels=len(config['data']['variables']),
        out_channels=len(config['data']['variables']),
        base_dim=32  # 11.4M params
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Noise schedule - faithful to DEF
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
    
    # Mixed precision for memory efficiency
    scaler = GradScaler()
    
    # Training loop
    print("\nStarting training...")
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'diffusion_fullsize'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    log_file = Path('logs') / 'diffusion_fullsize_training.log'
    log_file.parent.mkdir(exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    accumulation_steps = 4  # Gradient accumulation
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        
        for batch_idx, (current_state, future_state) in enumerate(pbar):
            # Move to device
            future_state = future_state.to(device)
            batch_size = future_state.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, config['diffusion']['timesteps'], (batch_size,), device=device)
            
            # Create noise
            noise = torch.randn_like(future_state)
            
            # Add noise according to schedule
            noisy_state = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward pass with mixed precision
            with autocast():
                # Model predicts noise (faithful to DEF!)
                noise_pred = model(noisy_state, t)
                
                # Simple MSE loss on noise
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1
            global_step += 1
            
            # Update progress bar
            avg_loss = epoch_loss / batch_count
            gpu_mb = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                't': t[0].item(),
                'gpu_mb': f'{gpu_mb:.0f}'
            })
            
            # Log to file
            if global_step % 10 == 0:
                with open(log_file, 'a') as f:
                    f.write(f"Step {global_step}: loss={avg_loss:.4f}, gpu_mb={gpu_mb:.0f}\n")
            
            # Memory management
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Quick validation
            if batch_idx % 100 == 0 and batch_idx > 0:
                model.eval()
                with torch.no_grad():
                    # Test denoising at different timesteps
                    test_t = torch.tensor([100, 500, 900], device=device)
                    test_noise = torch.randn(3, *future_state.shape[1:], device=device)
                    test_clean = future_state[0:1].repeat(3, 1, 1, 1)
                    test_noisy = noise_schedule.add_noise(test_clean, test_noise, test_t)
                    test_pred = model(test_noisy, test_t)
                    test_loss = F.mse_loss(test_pred, test_noise)
                    
                    peak_gb = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0
                    print(f"\n[Step {global_step}] Val loss: {test_loss.item():.4f}, Peak GPU: {peak_gb:.1f}GB")
                
                model.train()
        
        # Epoch complete
        avg_loss = epoch_loss / batch_count
        print(f"\nEpoch {epoch}: Average Loss = {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'config': config,
                'full_resolution': True,
                'model_params': total_params
            }
            checkpoint_path = checkpoint_dir / f'epoch_{epoch:04d}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                print("Saved as best model!")
        
        # Generate sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\nGenerating full-resolution sample...")
            model.eval()
            with torch.no_grad():
                # Start from noise at full resolution
                sample_shape = (1, len(config['data']['variables']), 1059, 1799)
                x = torch.randn(sample_shape, device=device)
                
                # Simplified DDPM sampling (just a few steps for demo)
                sample_steps = min(50, config['diffusion']['timesteps'])
                for i in tqdm(range(sample_steps, 0, -1), desc="Sampling"):
                    t = torch.tensor([i-1], device=device)
                    
                    # Predict and remove noise
                    with autocast():
                        noise_pred = model(x, t)
                    
                    # Simple denoising step
                    alpha = noise_schedule.alphas[i-1]
                    alpha_bar = noise_schedule.alphas_cumprod[i-1]
                    x = (x - (1-alpha)/torch.sqrt(1-alpha_bar) * noise_pred) / torch.sqrt(alpha)
                    
                    if i > 1:
                        # Add noise for next step
                        beta = noise_schedule.betas[i-1]
                        x += torch.sqrt(beta) * torch.randn_like(x) * 0.1  # Reduced noise
                
                print(f"Generated sample: shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")
            
            model.train()
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Final memory stats
    if torch.cuda.is_available():
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train faithful DDPM at full resolution')
    parser.add_argument('--config', type=str, default='configs/diffusion_max.yaml')
    args = parser.parse_args()
    
    train_fullsize_ddpm(args.config)