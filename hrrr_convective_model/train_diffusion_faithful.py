#!/usr/bin/env python3
"""
Faithful DEF training implementation for convection prediction.
This trains the model to predict NOISE, not weather states.
Critical for accurate convection forecasting and uncertainty quantification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import yaml
from datetime import datetime
import argparse

from models.diffusion.ddpm_convection import ConvectionDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer
from utils.metrics import crps_ensemble, spread_error_ratio


class ConvectionDiffusionDataset(torch.utils.data.Dataset):
    """Dataset that returns past states for conditioning and future state for denoising."""
    def __init__(self, zarr_path, stats_path, variables, history_length=4, lead_hours=1):
        self.dataset = HRRRDataset(
            zarr_path=zarr_path,
            variables=variables,
            lead_hours=lead_hours,
            stats_path=stats_path
        )
        self.history_length = history_length
        self.lead_hours = lead_hours
        
    def __len__(self):
        # Account for history requirement
        return len(self.dataset) - self.history_length
    
    def __getitem__(self, idx):
        # Offset by history length
        actual_idx = idx + self.history_length
        
        # Get future state (what we want to predict)
        future_state, _ = self.dataset[actual_idx]
        
        # Get past states for conditioning
        past_states = []
        for i in range(self.history_length):
            past_state, _ = self.dataset[actual_idx - self.history_length + i]
            past_states.append(past_state)
        
        past_states = torch.stack(past_states, dim=0)  # (T, C, H, W)
        
        # Also return the timestamp for temporal encoding
        timestamp = actual_idx + self.dataset.lead  # Hours since epoch
        
        return {
            'past_states': past_states,
            'future_state': future_state,
            'timestamp': timestamp
        }


def train_diffusion_faithful(config_path: str, resume: str = None):
    """
    Train the diffusion model faithfully according to DEF paper.
    
    Key principles:
    1. Model predicts noise, not weather
    2. Use cosine beta schedule
    3. Condition on past states
    4. Simple MSE loss on noise
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = ConvectionDiffusionDataset(
        zarr_path=config['data']['zarr'],
        stats_path=config['data']['stats'],
        variables=config['data']['variables'],
        history_length=config['model']['history_length'],
        lead_hours=config['training']['lead_hours']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("Creating diffusion model...")
    model = ConvectionDDPM(
        in_channels=len(config['data']['variables']),
        out_channels=len(config['data']['variables']),
        base_dim=config['model']['base_dim'],
        dim_mults=tuple(config['model']['dim_mults']),
        attention_resolutions=tuple(config['model']['attention_resolutions']),
        num_res_blocks=config['model']['num_res_blocks'],
        dropout=config['model']['dropout'],
        history_length=config['model']['history_length']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(
        timesteps=config['diffusion']['timesteps'],
        s=config['diffusion'].get('s', 0.008)
    )
    
    # Move schedule tensors to device
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'sqrt_alphas_cumprod', 
                 'sqrt_one_minus_alphas_cumprod', 'sqrt_recip_alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    # Optimizer - AdamW with lower learning rate for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Resume if specified
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['training']['checkpoint_dir']) / 'diffusion'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting diffusion training for {config['training']['epochs']} epochs...")
    print("Key: Model learns to predict NOISE, not weather states")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Get data
            past_states = batch['past_states'].to(device)
            future_state = batch['future_state'].to(device)
            batch_size = future_state.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, config['diffusion']['timesteps'], 
                (batch_size,), device=device
            )
            
            # Sample noise
            noise = torch.randn_like(future_state)
            
            # Add noise to future state
            noisy_future = noise_schedule.add_noise(future_state, noise, timesteps)
            
            # Forward pass - predict noise
            with autocast():
                noise_pred = model(noisy_future, timesteps, past_states)
                
                # Simple MSE loss on noise (this is the key!)
                loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': epoch_loss / num_batches
            })
            
            # Clear cache periodically
            if num_batches % 50 == 0:
                torch.cuda.empty_cache()
        
        # Epoch complete
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
        
        # Validation with ensemble generation
        if epoch % config['training']['val_interval'] == 0:
            print(f"Validation skipped for now (implement val_loader)")
            # val_metrics = validate_ensemble(
            #     model, noise_schedule, val_loader, device, config
            # )
            # print(f"Validation - CRPS: {val_metrics['crps']:.4f}, "
            #       f"Spread/Error: {val_metrics['spread_error']:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:04d}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # Step scheduler
        scheduler.step()
    
    print("\nTraining complete!")
    print("Model trained to predict noise for ensemble weather forecasting")


def validate_ensemble(model, noise_schedule, val_loader, device, config):
    """
    Validate by generating ensembles and computing probabilistic metrics.
    This is where DEF shines - we get calibrated uncertainty!
    """
    model.eval()
    num_members = config['validation']['ensemble_size']
    
    all_crps = []
    all_spread_error = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            past_states = batch['past_states'].to(device)
            future_truth = batch['future_state'].to(device)
            batch_size = future_truth.shape[0]
            
            # Generate ensemble forecast
            ensemble_preds = []
            
            for _ in range(num_members):
                # Start from noise
                xt = torch.randn_like(future_truth)
                
                # Reverse diffusion process
                for t in reversed(range(config['diffusion']['timesteps'])):
                    t_batch = torch.full((batch_size,), t, device=device)
                    
                    # Predict noise
                    noise_pred = model(xt, t_batch, past_states)
                    
                    # Denoise step
                    xt = noise_schedule.denoise_step(xt, noise_pred, t)
                
                ensemble_preds.append(xt)
            
            # Stack ensemble
            ensemble_preds = torch.stack(ensemble_preds, dim=0)  # (M, B, C, H, W)
            
            # Compute metrics
            crps = crps_ensemble(ensemble_preds, future_truth)
            spread_error = spread_error_ratio(ensemble_preds, future_truth)
            
            all_crps.append(crps.mean().item())
            all_spread_error.append(spread_error.mean().item())
    
    return {
        'crps': np.mean(all_crps),
        'spread_error': np.mean(all_spread_error)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DEF diffusion model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    train_diffusion_faithful(args.config, args.resume)