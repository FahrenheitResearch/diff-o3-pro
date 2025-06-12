#!/usr/bin/env python
"""
Train deterministic forecast model on HRRR forecast sequences.
Uses F00->F01, F01->F02, ..., F17->F18 to learn realistic forecast dynamics.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.unet_attention_fixed import UNetAttn
from hrrr_dataset.hrrr_forecast_data import HRRRForecastDataset
from utils.normalization import Normalizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, device, scaler, cfg):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets, timestamps) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        timestamps = timestamps.to(device)
        
        # Use only the input timestamp for temporal encoding
        input_timestamps = timestamps[:, 0]  # Shape: [B]
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=cfg['training'].get('use_amp', True)):
            outputs = model(inputs, input_timestamps)
            loss = nn.functional.mse_loss(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / cfg['training']['gradient_accumulation_steps']
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % cfg['training']['gradient_accumulation_steps'] == 0:
            # Gradient clipping
            if cfg['training'].get('clip_grad_norm', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * cfg['training']['gradient_accumulation_steps']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / num_batches


def validate(model, dataloader, device, cfg):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, timestamps in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            timestamps = timestamps.to(device)
            
            # Use only the input timestamp
            input_timestamps = timestamps[:, 0]
            
            outputs = model(inputs, input_timestamps)
            loss = nn.functional.mse_loss(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main(args):
    # Load configuration
    cfg = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = HRRRForecastDataset(
        zarr_path=Path(cfg['data']['zarr']),
        variables=cfg['data']['variables'],
        stats_path=Path(cfg['data']['stats'])
    )
    
    # For validation, use a subset of the data (last 10%)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
    val_subset = torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))
    
    print(f"Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training'].get('pin_memory', True),
        persistent_workers=cfg['training'].get('persistent_workers', False)
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers']
    )
    
    # Create model
    print("\nInitializing model...")
    model = UNetAttn(
        in_ch=len(cfg['data']['variables']),
        out_ch=len(cfg['data']['variables']),
        base_features=cfg['training']['base_features'],
        use_temporal_encoding=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training'].get('weight_decay', 1e-5)
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'],
        eta_min=cfg['training']['lr'] * 0.01
    )
    
    # Setup gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training'].get('use_amp', True))
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, cfg)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % cfg['training']['validate_every'] == 0:
            val_loss = validate(model, val_loader, device, cfg)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': cfg
                }
                torch.save(checkpoint, 'checkpoints/forecast_model_best.pt')
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % cfg['training']['checkpoint_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'config': cfg
            }
            torch.save(checkpoint, f'checkpoints/forecast_epoch_{epoch:03d}.pt')
            print(f"✓ Saved checkpoint")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': cfg
    }
    torch.save(final_checkpoint, 'checkpoints/forecast_model_final.pt')
    print("\n✓ Training complete! Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_4090.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)