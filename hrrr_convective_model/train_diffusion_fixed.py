#!/usr/bin/env python3
"""
Train diffusion model for ensemble generation.
This learns to generate perturbations conditioned on deterministic forecasts.
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from datetime import datetime

from models.unet_residual import UNetResidual
from models.diffusion import GaussianDiffusion, ConditionalDiffusionUNet
from hrrr_dataset.hrrr_data_fixed import HRRRDatasetFixed
from utils.normalization import Normalizer

class DiffusionDataset(HRRRDatasetFixed):
    """Dataset that provides pairs and deterministic forecasts for diffusion training."""
    
    def __init__(self, zarr_path, variables, stats_path, forecast_model, device, **kwargs):
        super().__init__(zarr_path, variables, 1, stats_path, **kwargs)
        self.forecast_model = forecast_model
        self.device = device
        
    def __getitem__(self, idx):
        # Get base data
        x, y, timestamp_x, timestamp_y = super().__getitem__(idx)
        
        # Get deterministic forecast (condition)
        with torch.no_grad():
            x_tensor = x.unsqueeze(0).to(self.device)
            timestamp_tensor = torch.tensor([timestamp_x]).to(self.device)
            y_forecast = self.forecast_model(x_tensor, timestamp_tensor)
            y_forecast = y_forecast.squeeze(0).cpu()
        
        return x, y, y_forecast, timestamp_x

def train_epoch(model, diffusion, dataloader, optimizer, scaler, device, epoch):
    """Train diffusion model for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (x, y, y_forecast, timestamps) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        y_forecast = y_forecast.to(device)
        
        B = x.shape[0]
        
        optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        
        # Forward diffusion - add noise to target
        noise = torch.randn_like(y)
        y_noisy = diffusion.q_sample(y, t, noise)
        
        # Predict noise conditioned on forecast
        with autocast():
            noise_pred = model(y_noisy, t, y_forecast)
            loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient clipping and step
        if (batch_idx + 1) % 8 == 0:  # Gradient accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Clear cache periodically
            if (batch_idx + 1) % 80 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'gpu_mb': torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
        })
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/diffusion.yaml')
    parser.add_argument('--forecast_checkpoint', type=str, 
                       default='checkpoints/deterministic/best_model.pt',
                       help='Trained forecast model checkpoint')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume', type=str, help='Resume diffusion training')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load forecast model
    print(f"Loading forecast model from: {args.forecast_checkpoint}")
    forecast_ckpt = torch.load(args.forecast_checkpoint, map_location=device)
    forecast_cfg = forecast_ckpt['config']
    
    n_channels = len(forecast_cfg['data']['variables'])
    forecast_model = UNetResidual(
        n_channels, n_channels,
        base_features=forecast_cfg['training']['base_features'],
        use_temporal_encoding=True
    )
    forecast_model.load_state_dict(forecast_ckpt['model_state_dict'])
    forecast_model.to(device).eval()
    print("✓ Loaded forecast model")
    
    # Create diffusion model
    diffusion_process = GaussianDiffusion(
        timesteps=cfg['diffusion']['timesteps'],
        beta_schedule=cfg['diffusion']['beta_schedule']
    ).to(device)
    
    diffusion_model = ConditionalDiffusionUNet(
        in_channels=n_channels,
        out_channels=n_channels,
        cond_channels=n_channels,
        base_features=cfg['diffusion']['base_features']
    ).to(device)
    
    print(f"Diffusion model parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    
    # Create optimizer
    optimizer = AdamW(
        diffusion_model.parameters(),
        lr=cfg['diffusion']['lr'],
        weight_decay=cfg['diffusion'].get('weight_decay', 1e-4)
    )
    
    # Create dataset with forecast model
    train_dataset = DiffusionDataset(
        Path(cfg['data']['zarr']),
        forecast_cfg['data']['variables'],
        Path(cfg['data']['stats']),
        forecast_model,
        device,
        sample_stride=1,
        augment=True,
        return_timestamps=True,
        epoch_start_hours=cfg['data'].get('epoch_start_hours', 0)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['diffusion']['batch_size'],
        shuffle=True,
        num_workers=0,  # Less workers since we're GPU-bound
        pin_memory=False,
        drop_last=True
    )
    
    # Scaler for mixed precision
    scaler = GradScaler()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['diffusion']['epochs'],
        eta_min=1e-6
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create directories
    checkpoint_dir = Path('checkpoints/diffusion')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    log_dir = Path('logs/diffusion')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    print(f"\nStarting diffusion training for {cfg['diffusion']['epochs']} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, cfg['diffusion']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg['diffusion']['epochs']}")
        print(f"{'='*50}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        train_loss = train_epoch(
            diffusion_model, diffusion_process, train_loader,
            optimizer, scaler, device, epoch
        )
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': cfg
            }, checkpoint_dir / 'best_model.pt')
            print("✓ Saved best model!")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': cfg
            }, checkpoint_dir / f'epoch_{epoch:03d}.pt')
            print(f"✓ Saved checkpoint at epoch {epoch}")
        
        # Update LR
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save final model
    torch.save({
        'epoch': cfg['diffusion']['epochs'] - 1,
        'model_state_dict': diffusion_model.state_dict(),
        'config': cfg
    }, checkpoint_dir / 'final_model.pt')
    
    print("\n✓ Diffusion training complete!")

if __name__ == '__main__':
    main()