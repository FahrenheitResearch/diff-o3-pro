#!/usr/bin/env python
"""
Train the deterministic forecast model G_φ for DEF.
Enhanced version with gradient accumulation, mixed precision, and proper logging.
"""
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from models.unet_attention_fixed import UNetAttn
from hrrr_dataset.hrrr_data import HRRRDataset
import utils.metrics as metrics


def load_config(path="configs/expanded.yaml"):
    """Load configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_model(cfg, device):
    """Create forecast model with proper channel count."""
    # Handle both test config (simple variables list) and full config
    if 'variables' in cfg['data']:
        # Simple test config
        total_channels = len(cfg['data']['variables'])
        print(f"Model channels: {total_channels} (test mode)")
    else:
        # Full config with surface/atmospheric/forcing variables
        n_surface = len(cfg['data']['surface_variables'])
        n_pressure_levels = len(cfg['data']['pressure_levels'])
        n_atmos_vars = len(cfg['data']['atmospheric_variables'])
        n_3d = n_atmos_vars * n_pressure_levels
        n_forcing = len(cfg['data'].get('forcing_variables', []))
        
        total_channels = n_surface + n_3d + n_forcing
        print(f"Model channels: {total_channels} (surface: {n_surface}, 3D: {n_3d}, forcing: {n_forcing})")
    
    model = UNetAttn(
        in_ch=total_channels,
        out_ch=total_channels,
        base_features=cfg['training']['base_features'],
        use_temporal_encoding=True
    )
    
    return model.to(device)


def expand_variable_list(cfg):
    """Expand variable list to include all pressure levels."""
    # Handle both test config (simple variables list) and full config
    if 'variables' in cfg['data']:
        # Simple test config - just return the variables list
        return cfg['data']['variables']
    else:
        # Full config with surface/atmospheric/forcing variables
        all_vars = cfg['data']['surface_variables'].copy()
        
        # Add 3D variables at each pressure level
        for var in cfg['data']['atmospheric_variables']:
            for level in cfg['data']['pressure_levels']:
                all_vars.append(f"{var}_{level}")
        
        # Add forcing variables
        all_vars.extend(cfg['data'].get('forcing_variables', []))
        
        return all_vars


def train_epoch(model, dataloader, optimizer, criterion, scaler, cfg, device, epoch):
    """Train model for one epoch with gradient accumulation."""
    model.train()
    
    total_loss = 0
    grad_accum_steps = cfg['training']['gradient_accumulation_steps']
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # Generate timestamps (hours since epoch for temporal encoding)
        # In practice, this should come from the dataset
        B = x.shape[0]
        base_time = epoch * 24 + batch_idx  # Dummy timestamps
        timestamps = torch.arange(B, device=device).float() + base_time
        
        # Forward pass with mixed precision
        with autocast():
            y_pred = model(x, timestamps)
            loss = criterion(y_pred, y)
            loss = loss / grad_accum_steps  # Scale loss for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item() * grad_accum_steps
        
        # Compute additional metrics
        with torch.no_grad():
            rmse = metrics.rmse(y, y_pred)
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'rmse': rmse.item()
        })
        
        # Logging disabled for now
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_rmse = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Validation")):
            x, y = x.to(device), y.to(device)
            
            # Generate timestamps
            B = x.shape[0]
            timestamps = torch.arange(B, device=device).float()
            
            with autocast():
                y_pred = model(x, timestamps)
                loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            total_rmse += metrics.rmse(y, y_pred).item()
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'rmse': total_rmse / n_batches
    }


def compute_normalization_stats(dataset, num_samples=1000):
    """Compute mean and std for each variable."""
    print("Computing normalization statistics...")
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Collect samples
    samples = []
    for idx in tqdm(indices, desc="Collecting samples"):
        x, _ = dataset[idx]
        samples.append(x.numpy())
    
    # Stack and compute statistics
    samples = np.stack(samples, axis=0)  # [N, C, H, W]
    
    # Compute per-channel statistics
    mean = samples.mean(axis=(0, 2, 3))  # [C]
    std = samples.std(axis=(0, 2, 3)) + 1e-8  # [C]
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Train forecast model for DEF')
    parser.add_argument('--config', type=str, default='configs/expanded.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    # Removed wandb arguments for now
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Wandb disabled for now
    
    # Expand variable list
    all_vars = expand_variable_list(cfg)
    cfg['data']['variables'] = all_vars
    print(f"Total variables: {len(all_vars)}")
    
    # Create directories
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create dataset
    train_dataset = HRRRDataset(
        Path(cfg['data']['zarr']),
        all_vars,
        cfg['training']['lead_hours'],
        Path(cfg['data']['stats']),
        sample_stride=1
    )
    
    # If stats file doesn't exist, compute it
    stats_path = Path(cfg['data']['stats'])
    if not stats_path.exists():
        print("Stats file not found. Computing statistics...")
        mean, std = compute_normalization_stats(train_dataset)
        stats = {}
        for i, var in enumerate(all_vars):
            stats[var] = {
                'mean': float(mean[i]),
                'std': float(std[i])
            }
        stats_path.parent.mkdir(exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        drop_last=True  # For consistent batch sizes
    )
    
    # Create model
    model = create_model(cfg, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=1e-4
    )
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create scaler for mixed precision
    scaler = GradScaler()
    
    # Learning rate scheduler
    warmup_epochs = cfg['training'].get('warmup_epochs', 2)
    if cfg['training']['lr_schedule'] == 'cosine':
        # Cosine annealing after warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['training']['epochs'] - warmup_epochs,
            eta_min=1e-6
        )
    else:
        scheduler = None
    
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
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg['training']['epochs']}")
        print(f"{'='*50}")
        
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_lr = cfg['training']['lr'] * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.6f}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, cfg, device, epoch
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % cfg['training']['validate_every'] == 0:
            val_metrics = validate(model, train_loader, criterion, device)
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': cfg
                }, checkpoint_dir / 'forecast_model_best.pt')
                print("Saved best model!")
            
            # Logging disabled for now
        
        # Save checkpoint
        if (epoch + 1) % cfg['training']['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': cfg
            }, checkpoint_dir / f'forecast_epoch_{epoch:03d}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # Update learning rate
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
            
            # Logging disabled for now
    
    print("\nTraining complete!")
    
    # Save final model
    torch.save({
        'epoch': cfg['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, checkpoint_dir / 'forecast_model_final.pt')


if __name__ == '__main__':
    main()