#!/usr/bin/env python3
"""
Deterministic weather forecasting with fixed architecture and training.
Key improvements:
1. Residual UNet architecture
2. 1-hour time steps
3. Spectral + L1 loss
4. Real timestamps
5. Longer training (200 epochs)
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
import os
from datetime import datetime

from models.unet_residual import UNetResidual, UNetResidualSmall
from hrrr_dataset.hrrr_data_fixed import HRRRDatasetFixed, compute_normalization_stats_fixed
from utils.losses import WeatherLoss
import utils.metrics as metrics

def create_model(cfg, device):
    """Create residual model based on config."""
    n_channels = len(cfg['data']['variables'])
    
    # Use smaller model if specified
    if cfg['training'].get('use_small_model', False):
        model = UNetResidualSmall(
            in_ch=n_channels,
            out_ch=n_channels,
            use_temporal_encoding=True
        )
    else:
        model = UNetResidual(
            in_ch=n_channels,
            out_ch=n_channels,
            base_features=cfg['training']['base_features'],
            use_temporal_encoding=True
        )
    
    model = model.to(device)
    return model

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    """Train for one epoch with proper timestamps."""
    model.train()
    
    total_loss = 0
    loss_components = {'l1': 0, 'mse': 0, 'spectral': 0, 'gradient': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        x, y, timestamp_x, timestamp_y = batch
        x, y = x.to(device), y.to(device)
        timestamp_x = timestamp_x.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            y_pred = model(x, timestamp_x)
            loss, components = criterion(y_pred, y)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in components.items():
            if k != 'total':
                loss_components[k] += v
        
        # Compute RMSE
        with torch.no_grad():
            rmse = metrics.rmse(y, y_pred)
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'rmse': rmse.item(),
            'l1': loss_components['l1'] / (batch_idx + 1)
        })
    
    return total_loss / len(dataloader), loss_components

def validate(model, dataloader, criterion, device):
    """Validate model with proper metrics."""
    model.eval()
    
    total_loss = 0
    total_rmse = 0
    loss_components = {'l1': 0, 'mse': 0, 'spectral': 0, 'gradient': 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            x, y, timestamp_x, timestamp_y = batch
            x, y = x.to(device), y.to(device)
            timestamp_x = timestamp_x.to(device)
            
            y_pred = model(x, timestamp_x)
            loss, components = criterion(y_pred, y)
            
            total_loss += loss.item()
            total_rmse += metrics.rmse(y, y_pred).item()
            
            for k, v in components.items():
                if k != 'total':
                    loss_components[k] += v
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'rmse': total_rmse / n,
        'components': {k: v/n for k, v in loss_components.items()}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/deterministic.yaml')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override epochs if specified
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path('checkpoints/deterministic')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    log_dir = Path('logs/deterministic')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create dataset with 1-hour lead time
    print("Creating dataset...")
    train_dataset = HRRRDatasetFixed(
        Path(cfg['data']['zarr']),
        cfg['data']['variables'],
        lead_hours=1,  # 1-hour predictions!
        stats_path=Path(cfg['data']['stats']),
        sample_stride=1,
        augment=True,
        return_timestamps=True,
        epoch_start_hours=cfg['data'].get('epoch_start_hours', 0)
    )
    
    # Create validation dataset (no augmentation)
    val_dataset = HRRRDatasetFixed(
        Path(cfg['data']['zarr']),
        cfg['data']['variables'],
        lead_hours=1,
        stats_path=Path(cfg['data']['stats']),
        sample_stride=10,  # Use every 10th sample for validation
        augment=False,
        return_timestamps=True,
        epoch_start_hours=cfg['data'].get('epoch_start_hours', 0)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_model(cfg, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 1e-4)
    )
    
    # Create loss function (WeatherLoss with spectral component)
    criterion = WeatherLoss(
        l1_weight=0.4,
        spectral_weight=0.3,
        mse_weight=0.2,
        gradient_weight=0.1
    )
    
    # Create scaler for mixed precision
    scaler = GradScaler()
    
    # Learning rate scheduler - cosine annealing with warmup
    warmup_epochs = cfg['training'].get('warmup_epochs', 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'] - warmup_epochs,
        eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training for {cfg['training']['epochs']} epochs...")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Config: {cfg}\n\n")
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg['training']['epochs']}")
        print(f"{'='*60}")
        
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_lr = cfg['training']['lr'] * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.6f}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        
        # Log training metrics
        log_msg = f"Epoch {epoch} - Train loss: {train_loss:.4f}"
        for k, v in train_components.items():
            log_msg += f", {k}: {v/len(train_loader):.4f}"
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
        
        # Validation
        if (epoch + 1) % cfg['training']['validate_every'] == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            
            val_msg = f"Validation - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}"
            for k, v in val_metrics['components'].items():
                val_msg += f", {k}: {v:.4f}"
            print(val_msg)
            
            with open(log_file, 'a') as f:
                f.write(val_msg + "\n")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': cfg
                }, checkpoint_dir / 'best_model.pt')
                print("✓ Saved best model!")
        
        # Save checkpoint
        if (epoch + 1) % cfg['training']['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': cfg
            }, checkpoint_dir / f'epoch_{epoch:03d}.pt')
            print(f"✓ Saved checkpoint at epoch {epoch}")
        
        # Update learning rate
        if epoch >= warmup_epochs:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
    
    print("\n✓ Training complete!")
    
    # Save final model
    torch.save({
        'epoch': cfg['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, checkpoint_dir / 'final_model.pt')

if __name__ == '__main__':
    main()