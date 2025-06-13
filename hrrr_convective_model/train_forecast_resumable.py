#!/usr/bin/env python
"""
Fully resumable training for deterministic forecast model.
Saves complete training state and can resume from any checkpoint.
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
import json
import os

from models.unet_attention_fixed import UNetAttn
from hrrr_dataset.hrrr_forecast_data import HRRRForecastDataset
from utils.normalization import Normalizer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(state, filename, is_best=False):
    """Save a checkpoint with all training state."""
    torch.save(state, filename)
    print(f"✓ Saved checkpoint: {filename}")
    
    if is_best:
        best_path = Path(filename).parent / 'forecast_model_best.pt'
        torch.save(state, best_path)
        print(f"✓ Saved best model: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load a checkpoint and restore training state."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✓ Resumed from epoch {start_epoch}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    
    return start_epoch, best_val_loss


def save_training_state(filename, epoch, model, optimizer, scheduler, scaler, 
                       train_loss, val_loss, best_val_loss, cfg, 
                       completed_batches=0, total_batches=0):
    """Save complete training state for resumption."""
    state = {
        'epoch': epoch,
        'completed_batches': completed_batches,
        'total_batches': total_batches,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': cfg,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(state, filename)
    print(f"✓ Saved training state: {filename}")


def train_epoch(model, dataloader, optimizer, device, scaler, cfg, 
                epoch, checkpoint_dir, save_every_n_batches=100):
    """Train one epoch with periodic checkpointing."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Check if we're resuming mid-epoch
    resume_file = checkpoint_dir / 'resume_state.pt'
    start_batch = 0
    
    if resume_file.exists():
        resume_state = torch.load(resume_file)
        if resume_state['epoch'] == epoch:
            start_batch = resume_state['completed_batches']
            print(f"Resuming epoch {epoch} from batch {start_batch}")
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for batch_idx, (inputs, targets, timestamps) in pbar:
        # Skip already completed batches
        if batch_idx < start_batch:
            continue
            
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
        
        # Periodic checkpoint within epoch
        if (batch_idx + 1) % save_every_n_batches == 0:
            save_training_state(
                resume_file, epoch, model, optimizer, None, scaler,
                total_loss/num_batches, 0, float('inf'), cfg,
                completed_batches=batch_idx + 1, total_batches=len(dataloader)
            )
    
    # Clean up resume file after epoch completion
    if resume_file.exists():
        os.remove(resume_file)
    
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
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
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
    
    # Check for resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if args.resume == 'latest':
            # Find latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob('forecast_epoch_*.pt'))
            if checkpoints:
                args.resume = str(checkpoints[-1])
            else:
                print("No checkpoints found to resume from")
                args.resume = None
        
        if args.resume and Path(args.resume).exists():
            start_epoch, best_val_loss = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler
            )
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler, cfg,
            epoch, checkpoint_dir, save_every_n_batches=50
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % cfg['training']['validate_every'] == 0:
            val_loss = validate(model, val_loader, device, cfg)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoint with full state
            checkpoint_path = checkpoint_dir / f'forecast_epoch_{epoch:03d}.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': cfg
            }, checkpoint_path, is_best=is_best)
        else:
            # Save checkpoint even without validation
            if (epoch + 1) % cfg['training']['checkpoint_every'] == 0:
                checkpoint_path = checkpoint_dir / f'forecast_epoch_{epoch:03d}.pt'
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'train_loss': train_loss,
                    'best_val_loss': best_val_loss,
                    'config': cfg
                }, checkpoint_path)
        
        # Update learning rate
        scheduler.step()
        
        # Log current state
        with open(checkpoint_dir / 'training_log.json', 'a') as f:
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss if (epoch + 1) % cfg['training']['validate_every'] == 0 else None,
                'lr': optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }
            f.write(json.dumps(log_entry) + '\n')
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': cfg
    }
    torch.save(final_checkpoint, checkpoint_dir / 'forecast_model_final.pt')
    print("\n✓ Training complete! Final model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_4090.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (or "latest")')
    args = parser.parse_args()
    main(args)