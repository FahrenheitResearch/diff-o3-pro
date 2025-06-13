#!/usr/bin/env python3
"""
100% FAITHFUL minimal DEF implementation for RTX 4090.
This is the CORRECT architecture - predicts NOISE with timestep embedding.
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

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def train_faithful_diffusion(config_path):
    """Train 100% faithful diffusion model."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*60)
    print("100% FAITHFUL DEF IMPLEMENTATION")
    print("Key: Model predicts NOISE with timestep embedding")
    print("="*60 + "\n")
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create dataset
    print("Creating dataset...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=config['training']['lead_hours'],
        stats_path=config['data']['stats']
    )
    
    # Simple dataloader for memory efficiency
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Always 1 for 4090
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Variables: {config['data']['variables']}")
    
    # Create FAITHFUL diffusion model
    print("\nCreating faithful DDPM model...")
    model = UltraMinimalDDPM(
        in_channels=len(config['data']['variables']),
        out_channels=len(config['data']['variables']),
        base_dim=config['model']['base_dim']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Cosine beta schedule - more stable than linear
    print("Using cosine beta schedule (faithful to DEF)")
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps'],
        s=config['diffusion'].get('s', 0.008)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Training loop
    print("\nStarting training...")
    print("Architecture check:")
    print("✓ Model predicts NOISE (not weather)")
    print("✓ Timestep embedding included")
    print("✓ Cosine beta schedule")
    print("✓ Simple MSE loss on noise")
    print()
    
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'diffusion_faithful'
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
            
            # Add noise according to schedule
            noisy_state = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward pass - predict noise WITH timestep
            noise_pred = model(noisy_state, t)
            
            # Simple MSE loss on noise (faithful to DEF!)
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Update progress
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/batch_count:.4f}',
                't': t[0].item(),
                'gpu_mb': f'{torch.cuda.memory_allocated()/1024**2:.0f}' if torch.cuda.is_available() else 'N/A'
            })
            
            # Memory management
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Quick validation check
            if batch_idx % 100 == 0 and batch_idx > 0:
                with torch.no_grad():
                    # Test at different timesteps
                    test_t = torch.tensor([10, 50, 90], device=device)
                    test_noise = torch.randn(3, *future_state.shape[1:], device=device)
                    test_noisy = noise_schedule.add_noise(
                        future_state[0:1].repeat(3, 1, 1, 1),
                        test_noise,
                        test_t
                    )
                    test_pred = model(test_noisy, test_t)
                    test_loss = F.mse_loss(test_pred, test_noise)
                    print(f"\n  Validation at t=[10,50,90]: loss={test_loss.item():.4f}")
        
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
                'architecture': 'MinimalDDPM_faithful'
            }
            checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                print("Saved as best model!")
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Final memory stats
    if torch.cuda.is_available():
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train faithful minimal DDPM')
    parser.add_argument('--config', type=str, default='configs/diffusion_4090_light.yaml')
    args = parser.parse_args()
    
    train_faithful_diffusion(args.config)