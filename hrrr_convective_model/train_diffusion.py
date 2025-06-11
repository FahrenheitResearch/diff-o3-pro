#!/usr/bin/env python
"""
Train the diffusion perturbation model ε_θ for DEF.
Implements Algorithm 3: randomly advances samples through G_φ with probability 0.5.
"""
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from models.unet_attention_fixed import UNetAttn
from models.diffusion import GaussianDiffusion, ConditionalDiffusionUNet
from hrrr_dataset.hrrr_data import HRRRDataset
import utils.metrics as metrics


def load_config(path="configs/expanded.yaml"):
    """Load configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_paths(cfg):
    """Setup paths for models and checkpoints."""
    paths = {
        'forecast_model': Path('checkpoints/forecast_model_best.pt'),
        'diffusion_dir': Path('checkpoints/diffusion'),
        'logs': Path('logs/diffusion')
    }
    
    # Create directories
    for p in paths.values():
        if p.suffix == '':  # It's a directory
            p.mkdir(parents=True, exist_ok=True)
    
    return paths


def load_forecast_model(cfg, device):
    """Load pretrained deterministic forecast model G_φ."""
    # Handle both test config (simple variables list) and full config
    if 'variables' in cfg['data']:
        # Simple test config
        total_vars = len(cfg['data']['variables'])
    else:
        # Full config
        n_vars = len(cfg['data']['surface_variables'])
        n_3d = len(cfg['data']['atmospheric_variables']) * len(cfg['data']['pressure_levels'])
        n_forcing = len(cfg['data'].get('forcing_variables', []))
        total_vars = n_vars + n_3d + n_forcing
    
    # Create model
    model = UNetAttn(
        total_vars, 
        total_vars, 
        base_features=cfg['training']['base_features'],
        use_temporal_encoding=True
    )
    
    # Load checkpoint if exists
    ckpt_path = Path('checkpoints/forecast_model_best.pt')
    if ckpt_path.exists():
        print(f"Loading forecast model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Handle both old and new checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: No pretrained forecast model found. Using random initialization.")
    
    model.to(device)
    model.eval()  # Keep in eval mode
    return model


def create_diffusion_model(cfg, device):
    """Create diffusion model and diffusion process."""
    # Handle both test config (simple variables list) and full config
    if 'variables' in cfg['data']:
        # Simple test config
        total_vars = len(cfg['data']['variables'])
    else:
        # Full config
        n_vars = len(cfg['data']['surface_variables'])
        n_3d = len(cfg['data']['atmospheric_variables']) * len(cfg['data']['pressure_levels'])
        n_forcing = len(cfg['data'].get('forcing_variables', []))
        total_vars = n_vars + n_3d + n_forcing
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        timesteps=cfg['diffusion']['timesteps'],
        beta_schedule=cfg['diffusion']['beta_schedule'],
        loss_type='mse'
    )
    
    # Create diffusion U-Net
    # Conditioning on forecast state (same number of channels)
    model = ConditionalDiffusionUNet(
        in_channels=total_vars,
        out_channels=total_vars,
        cond_channels=total_vars,  # Condition on forecast
        base_features=cfg['training']['base_features'],
        time_emb_dim=256
    )
    
    return diffusion.to(device), model.to(device)


def train_epoch(
    diffusion_process,
    diffusion_model,
    forecast_model,
    dataloader,
    optimizer,
    scaler,
    cfg,
    device,
    epoch
):
    """Train diffusion model for one epoch."""
    diffusion_model.train()
    forecast_model.eval()
    
    total_loss = 0
    total_loss_cond = 0
    total_loss_uncond = 0
    num_advanced = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (x_past, x_future) in enumerate(pbar):
        x_past = x_past.to(device)
        x_future = x_future.to(device)
        
        # Randomly decide whether to advance through forecast model
        # Algorithm 3: with probability 0.5
        advance_mask = torch.rand(x_past.shape[0]) < 0.5
        num_advanced += advance_mask.sum().item()
        
        # Prepare training targets
        x_target = torch.zeros_like(x_past)
        condition = torch.zeros_like(x_past)
        
        # For samples to be advanced
        if advance_mask.any():
            with torch.no_grad():
                # Get forecast from deterministic model
                # Create mock timestamps (hours since epoch)
                # In production, these would come from the dataset
                batch_size_masked = advance_mask.sum().item()
                # Use batch_idx * 6 + sample_idx to simulate hourly data
                base_hour = batch_idx * 6
                timestamps = torch.arange(base_hour, base_hour + batch_size_masked, 
                                        device=device, dtype=torch.float32)
                x_forecast = forecast_model(x_past[advance_mask], timestamps)
                
            # Use future state as target, forecast as condition
            x_target[advance_mask] = x_future[advance_mask]
            condition[advance_mask] = x_forecast
        
        # For samples not advanced
        not_advance_mask = ~advance_mask
        if not_advance_mask.any():
            # Use current state as both target and condition
            x_target[not_advance_mask] = x_past[not_advance_mask]
            condition[not_advance_mask] = x_past[not_advance_mask]
        
        # Forward diffusion training
        with autocast():
            losses = diffusion_process.training_losses(
                diffusion_model,
                x_target,
                condition=condition,
                dropout_prob=cfg['diffusion']['dropout_prob']
            )
            loss = losses['loss']
        
        # Backward pass with gradient accumulation
        loss = loss / cfg['training']['gradient_accumulation_steps']
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % cfg['training']['gradient_accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        total_loss += losses['loss'].item()
        total_loss_cond += losses['loss_conditional'].item()
        total_loss_uncond += losses['loss_unconditional'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'loss_cond': total_loss_cond / (batch_idx + 1),
            'loss_uncond': total_loss_uncond / (batch_idx + 1),
            'advanced': f"{num_advanced}/{(batch_idx + 1) * x_past.shape[0]}"
        })
        
        # Logging disabled for now
    
    return {
        'loss': total_loss / len(dataloader),
        'loss_conditional': total_loss_cond / len(dataloader),
        'loss_unconditional': total_loss_uncond / len(dataloader),
        'advance_ratio': num_advanced / (len(dataloader) * cfg['training']['batch_size'])
    }


def validate(
    diffusion_process,
    diffusion_model,
    forecast_model,
    dataloader,
    cfg,
    device
):
    """Validate diffusion model."""
    diffusion_model.eval()
    forecast_model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (x_past, x_future) in enumerate(tqdm(dataloader, desc="Validation")):
            x_past = x_past.to(device)
            x_future = x_future.to(device)
            
            # For validation, always use forecast as condition
            # Create mock timestamps
            batch_size = x_past.shape[0]
            base_hour = batch_idx * 6
            timestamps = torch.arange(base_hour, base_hour + batch_size, 
                                    device=device, dtype=torch.float32)
            x_forecast = forecast_model(x_past, timestamps)
            
            # Compute diffusion loss
            losses = diffusion_process.training_losses(
                diffusion_model,
                x_future,
                condition=x_forecast,
                dropout_prob=0.0  # No dropout during validation
            )
            
            total_loss += losses['loss'].item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for DEF')
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
    
    # Setup paths
    paths = setup_model_paths(cfg)
    
    # Create dataset
    # Handle both test config (simple variables list) and full config
    if 'variables' in cfg['data']:
        # Simple test config - variables already listed
        all_vars = cfg['data']['variables']
    else:
        # Full config - expand variable list from config
        all_vars = cfg['data']['surface_variables'].copy()
        for var in cfg['data']['atmospheric_variables']:
            for level in cfg['data']['pressure_levels']:
                all_vars.append(f"{var}_{level}")
        all_vars.extend(cfg['data'].get('forcing_variables', []))
        
        # Update config with full variable list
        cfg['data']['variables'] = all_vars
    
    print(f"Total variables: {len(all_vars)}")
    
    # Create datasets
    train_dataset = HRRRDataset(
        Path(cfg['data']['zarr']),
        all_vars,
        cfg['training']['lead_hours'],
        Path(cfg['data']['stats']),
        sample_stride=1
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    
    # Load models
    forecast_model = load_forecast_model(cfg, device)
    diffusion_process, diffusion_model = create_diffusion_model(cfg, device)
    
    # Create optimizer
    optimizer = AdamW(
        diffusion_model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=1e-4
    )
    
    # Create scaler for mixed precision
    scaler = GradScaler()
    
    # Learning rate scheduler
    if cfg['training']['lr_schedule'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['training']['epochs'],
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg['training']['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            diffusion_process,
            diffusion_model,
            forecast_model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            epoch
        )
        
        print(f"\nTrain metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Validation every N epochs
        if (epoch + 1) % cfg['training']['validate_every'] == 0:
            # For now, use train loader for validation (in practice, use separate val set)
            val_loss = validate(
                diffusion_process,
                diffusion_model,
                forecast_model,
                train_loader,
                cfg,
                device
            )
            print(f"\nValidation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': cfg
                }, paths['diffusion_dir'] / 'best_model.pt')
                print("Saved best model!")
            
            # Logging disabled for now
        
        # Save checkpoint
        if (epoch + 1) % cfg['training']['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'config': cfg
            }, paths['diffusion_dir'] / f'checkpoint_epoch_{epoch:03d}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\nTraining complete!")
    
    # Save final model
    torch.save({
        'epoch': cfg['training']['epochs'] - 1,
        'model_state_dict': diffusion_model.state_dict(),
        'config': cfg
    }, paths['diffusion_dir'] / 'final_model.pt')


if __name__ == '__main__':
    main()