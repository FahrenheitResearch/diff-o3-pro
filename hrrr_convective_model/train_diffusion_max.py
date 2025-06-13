#!/usr/bin/env python3
"""
Maximum size faithful DDPM training for RTX 4090.
100% faithful to DEF - predicts NOISE with full architecture.
Optimized for 18GB GPU memory usage.
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
from collections import defaultdict

from models.diffusion.ddpm_18gb import DDPM18GB, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_max_diffusion(config_path):
    """Train maximum size faithful DDPM."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("MAXIMUM SIZE FAITHFUL DDPM TRAINING")
    print("Target: 18GB GPU memory usage on RTX 4090")
    print("Architecture: 100% faithful to DEF - predicts NOISE")
    print("="*70 + "\n")
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.75)  # Use 75% of VRAM (18GB of 24GB)
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create dataset
    print("Creating dataset...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=config['training']['lead_hours'],
        stats_path=config['data']['stats']
    )
    
    # DataLoader with memory-efficient settings
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Variables: {config['data']['variables']}")
    print(f"Spatial resolution: 1059 x 1799 (3km HRRR)")
    
    # Create maximum size model
    print(f"\nCreating 18GB-optimized DDPM...")
    model = DDPM18GB(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_dim=64  # Fixed for 50M params
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Noise schedule - faithful to DEF
    print(f"\nUsing cosine beta schedule with {config['diffusion']['timesteps']} timesteps")
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps'],
        s=config['diffusion']['s']
    )
    
    # Optimizer with gradient accumulation
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * config['training']['epochs']
    warmup_steps = config['training'].get('warmup_steps', 0)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision training
    use_amp = config['training'].get('mixed_precision', True)
    scaler = GradScaler() if use_amp else None
    
    # EMA for better generation
    ema = EMA(model, decay=config['training'].get('ema_decay', 0.9999))
    
    # Training loop
    print("\nStarting training...")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation']} steps")
    print(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation']}")
    print(f"Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'diffusion_max'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    accumulation_steps = config['training']['gradient_accumulation']
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_metrics = defaultdict(float)
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
            
            # Forward diffusion - add noise
            noisy_state = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                # Model predicts noise (faithful to DEF!)
                noise_pred = model(noisy_state, t)
                
                # Simple MSE loss on noise
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / accumulation_steps  # Scale by accumulation
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # Update EMA
                ema.update()
            
            # Update metrics
            epoch_metrics['loss'] += loss.item() * accumulation_steps
            epoch_metrics['lr'] = optimizer.param_groups[0]['lr']
            batch_count += 1
            global_step += 1
            
            # Update progress bar
            avg_loss = epoch_metrics['loss'] / batch_count
            gpu_mb = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                't_mean': f'{t.float().mean().item():.0f}',
                'lr': f'{epoch_metrics["lr"]:.2e}',
                'gpu_mb': f'{gpu_mb:.0f}'
            })
            
            # Memory management
            if batch_idx % config['memory']['empty_cache_interval'] == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Logging
            if global_step % config['logging']['log_every_n_steps'] == 0:
                gpu_gb = torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0
                print(f"\n[Step {global_step}] Loss: {avg_loss:.4f}, LR: {epoch_metrics['lr']:.2e}, GPU: {gpu_gb:.2f}GB")
        
        # Epoch complete
        avg_loss = epoch_metrics['loss'] / batch_count
        print(f"\nEpoch {epoch}: Average Loss = {avg_loss:.6f}")
        
        # Validation with ensemble generation
        if (epoch + 1) % config['validation'].get('sample_every_n_epochs', 10) == 0:
            print("\nGenerating validation ensemble...")
            model.eval()
            ema.apply_shadow()  # Use EMA weights for generation
            
            with torch.no_grad():
                # TODO: Implement ensemble generation
                pass
            
            ema.restore()  # Restore training weights
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config,
                'scaler_state_dict': scaler.state_dict() if use_amp else None
            }
            checkpoint_path = checkpoint_dir / f'epoch_{epoch:04d}.pt'
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
    parser = argparse.ArgumentParser(description='Train maximum size faithful DDPM')
    parser.add_argument('--config', type=str, default='configs/diffusion_max.yaml')
    args = parser.parse_args()
    
    train_max_diffusion(args.config)