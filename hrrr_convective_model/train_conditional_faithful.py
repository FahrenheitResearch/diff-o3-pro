#!/usr/bin/env python3
"""
100% FAITHFUL training of Conditional DDPM for DEF.
This correctly conditions the diffusion model on current atmospheric state.
Without this conditioning, the model cannot forecast!
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
import json
import logging
import time

# CRITICAL: Import the CONDITIONAL model
from models.diffusion.ddpm_conditional_minimal import ConditionalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def setup_logging(log_path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_conditional_ddpm(resume=True):
    """Train FAITHFUL conditional DDPM for weather forecasting."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("100% FAITHFUL CONDITIONAL DDPM TRAINING")
    print("This model CORRECTLY conditions on current state!")
    print("Without conditioning, we CAN'T forecast!")
    print("="*70 + "\n")
    
    # Configuration
    config = {
        'data': {
            'zarr': 'data/zarr/training_14day/hrrr.zarr',
            'stats': 'data/zarr/training_14day/stats.json',
            'variables': ['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN']
        },
        'model': {
            'channels': 7,
            'base_dim': 16  # Further reduced to save VRAM
        },
        'diffusion': {
            'timesteps': 1000,
            's': 0.008
        },
        'training': {
            'lead_hours': 1,
            'batch_size': 1,
            'accumulation_steps': 8,  # Further increased to reduce memory per step
            'epochs': 13,  # Resume for 10 more epochs (3 done + 10 = 13)
            'lr': 0.0002,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'save_interval': 2,
            'log_interval': 50
        }
    }
    
    # Setup directories
    checkpoint_dir = Path('checkpoints/conditional_faithful')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir / 'conditional_faithful.log')
    logger.info("Starting 100% FAITHFUL conditional DDPM training")
    
    # Save config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Memory management - more aggressive settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Disable cudnn benchmark to save memory
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Set memory fraction to leave room for other apps
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of VRAM
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=config['training']['lead_hours'],
        stats_path=config['data']['stats']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Variables: {config['data']['variables']}")
    
    # Create CONDITIONAL model
    logger.info(f"\nCreating CONDITIONAL DDPM model...")
    model = ConditionalDDPM(
        channels=config['model']['channels'],
        base_dim=config['model']['base_dim']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Noise schedule
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
    
    # Mixed precision - disabled to save memory
    use_amp = False  # Disable AMP to save memory
    # scaler = GradScaler()
    
    # Check for existing checkpoint to resume
    start_epoch = 0
    if resume and (checkpoint_dir / 'best_model.pt').exists():
        logger.info("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint.get('metrics', {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'timing': []
        })
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'timing': []
        }
    
    # Training loop
    logger.info("\nStarting training...")
    logger.info(f"Effective batch size: {config['training']['batch_size'] * config['training']['accumulation_steps']}")
    
    global_step = 0
    best_loss = float('inf')
    accumulation_steps = config['training']['accumulation_steps']
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        optimizer.zero_grad()
        
        for batch_idx, (current_state, future_state) in enumerate(pbar):
            # CRITICAL: We now use BOTH current and future states!
            current_state = current_state.to(device)
            future_state = future_state.to(device)
            batch_size = future_state.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, config['diffusion']['timesteps'], (batch_size,), device=device)
            
            # Create noise
            noise = torch.randn_like(future_state)
            
            # Add noise to future state
            noisy_future = noise_schedule.add_noise(future_state, noise, t)
            
            # Forward pass - CONDITIONAL on current state!
            if use_amp:
                with autocast():
                    # THIS IS THE KEY DIFFERENCE - we pass current_state as condition
                    noise_pred = model(noisy_future, t, current_state)
                    
                    # Still predict noise, but now conditioned on current state
                    loss = F.mse_loss(noise_pred, noise)
                    loss = loss / accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
            else:
                # No AMP - saves memory
                noise_pred = model(noisy_future, t, current_state)
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    optimizer.step()
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
                'avg': f'{avg_loss:.4f}',
                't': t[0].item(),
                'gpu': f'{gpu_mb:.0f}MB'
            })
            
            # Log periodically
            if batch_idx % config['training']['log_interval'] == 0:
                logger.info(f"Step {global_step}: loss={loss.item() * accumulation_steps:.4f}, avg={avg_loss:.4f}")
            
            # Memory management
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Epoch complete
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / batch_count
        logger.info(f"\nEpoch {epoch+1} complete: loss={avg_loss:.6f}, time={epoch_time:.1f}s")
        
        metrics['train_loss'].append((epoch, avg_loss))
        metrics['timing'].append((epoch, epoch_time))
        
        # Save metrics
        with open(checkpoint_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0 or epoch == config['training']['epochs'] - 1:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'loss': avg_loss,
                'config': config,
                'metrics': metrics
            }
            checkpoint_path = checkpoint_dir / f'epoch_{epoch+1:03d}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                logger.info("Saved as best model!")
        
        # Quick validation
        if epoch == config['training']['epochs'] - 1:
            logger.info("\nRunning final validation...")
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for i, (curr, future) in enumerate(dataloader):
                    if i >= 20:  # Quick validation on 20 samples
                        break
                    
                    curr = curr.to(device)
                    future = future.to(device)
                    
                    # Test at different timesteps
                    for t_val in [100, 500, 900]:
                        t = torch.tensor([t_val], device=device)
                        noise = torch.randn_like(future)
                        noisy_future = noise_schedule.add_noise(future, noise, t)
                        
                        if use_amp:
                            with autocast():
                                noise_pred = model(noisy_future, t, curr)
                                val_loss = F.mse_loss(noise_pred, noise)
                        else:
                            noise_pred = model(noisy_future, t, curr)
                            val_loss = F.mse_loss(noise_pred, noise)
                        
                        val_losses.append(val_loss.item())
            
            mean_val_loss = np.mean(val_losses)
            logger.info(f"Validation loss: {mean_val_loss:.4f}")
            metrics['val_loss'].append((epoch, mean_val_loss))
    
    # Training complete
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best loss: {best_loss:.6f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("\nNEXT STEPS:")
    logger.info("1. Run conditional sampling to generate forecasts")
    logger.info("2. Verify forecasts are conditioned on current state")
    logger.info("3. If working, train for more epochs")
    logger.info("="*70)
    
    # Final stats
    if torch.cuda.is_available():
        logger.info(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    train_conditional_ddpm()