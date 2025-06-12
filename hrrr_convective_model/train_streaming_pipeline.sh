#!/bin/bash
# Streaming training pipeline - download and train simultaneously

echo "=== HRRR Streaming Training Pipeline ==="
echo "This will download data and start training as soon as enough is available"
echo ""

# Create necessary directories
mkdir -p data/raw/streaming
mkdir -p data/zarr/streaming
mkdir -p logs/streaming

# Step 1: Start streaming download in background
echo "Step 1: Starting streaming download (background)..."
python scripts/download_hrrr_streaming.py \
    --start-date 20250605 \
    --end-date 20250611 \
    --hours 0 6 12 18 \
    --forecast-hours "0-18" \
    --output-dir data/raw/streaming \
    --parallel 8 \
    > logs/streaming/download.log 2>&1 &

DOWNLOAD_PID=$!
echo "Download started (PID: $DOWNLOAD_PID)"

# Step 2: Start streaming preprocessor in background
echo ""
echo "Step 2: Starting streaming preprocessor (background)..."
sleep 5  # Give downloader a head start

python scripts/preprocess_hrrr_streaming.py \
    --src data/raw/streaming \
    --out data/zarr/streaming \
    --check-interval 30 \
    --min-hours 2 \
    > logs/streaming/preprocess.log 2>&1 &

PREPROCESS_PID=$!
echo "Preprocessor started (PID: $PREPROCESS_PID)"

# Step 3: Wait for some initial data and compute stats
echo ""
echo "Step 3: Waiting for initial data..."
while [ ! -f "data/zarr/streaming/hrrr.zarr/.zgroup" ]; do
    echo -n "."
    sleep 5
done

# Wait for at least 20 sequences
echo ""
echo "Waiting for minimum sequences..."
while true; do
    if [ -f "data/zarr/streaming/processing_state.json" ]; then
        SEQUENCES=$(python -c "
import json
with open('data/zarr/streaming/processing_state.json') as f:
    state = json.load(f)
    print(state.get('time_idx', 0) // 2)
")
        echo -ne "\rSequences ready: $SEQUENCES (need 20+)    "
        if [ "$SEQUENCES" -ge 20 ]; then
            break
        fi
    fi
    sleep 5
done

echo ""
echo "Computing initial statistics..."
python scripts/compute_stats.py \
    --zarr data/zarr/streaming/hrrr.zarr \
    --out data/zarr/streaming/stats.json

# Step 4: Create streaming config
echo ""
echo "Step 4: Creating streaming training config..."
cat > configs/train_streaming_4090.yaml << EOF
data:
  zarr: "data/zarr/streaming/hrrr.zarr"
  stats: "data/zarr/streaming/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]

training:
  lead_hours: 1
  batch_size: 1
  num_workers: 0  # Important: use 0 workers for streaming
  epochs: 30
  lr: 2.0e-4
  gradient_accumulation_steps: 8
  checkpoint_every: 5
  validate_every: 2
  base_features: 42
  warmup_epochs: 1
  lr_schedule: "cosine"
  
  # Memory optimization
  pin_memory: true
  persistent_workers: false
  
  # Stability
  clip_grad_norm: 1.0
  weight_decay: 1.0e-5
  
  # Mixed precision
  use_amp: true
  
  # Streaming settings
  dataset_check_interval: 60  # Check for new data every minute
  min_sequences: 20  # Start training with at least 20 sequences

diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
  guidance_weight: 0.1
  dropout_prob: 0.1
  sampler: "dpm_solver_pp"
  solver_order: 2
  num_steps: 20
  ema_decay: 0.999
  
  epochs: 20
  batch_size: 1
  gradient_accumulation_steps: 8

ensemble:
  num_members: 8
  perturbation_samples: 4
  blend_weight: 0.95
EOF

# Step 5: Start training
echo ""
echo "Step 5: Starting streaming training..."
echo "Training will automatically use new data as it becomes available"

# Create a modified training script for streaming
cat > train_streaming_forecast.py << 'EOF'
#!/usr/bin/env python
import sys
sys.path.append('.')

# Import the resumable trainer but use streaming dataset
from train_forecast_resumable import *
from hrrr_dataset.hrrr_streaming_data import HRRRStreamingDataset

# Override the dataset creation
def main(args):
    # Load configuration
    cfg = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create streaming dataset
    print("\nLoading streaming dataset...")
    train_dataset = HRRRStreamingDataset(
        zarr_path=Path(cfg['data']['zarr']),
        variables=cfg['data']['variables'],
        stats_path=Path(cfg['data']['stats']),
        check_interval=cfg['training'].get('dataset_check_interval', 60),
        min_sequences=cfg['training'].get('min_sequences', 20)
    )
    
    # For streaming, we don't split train/val - use all data for training
    print(f"Initial training samples: {len(train_dataset)}")
    print("Dataset will grow as more data is downloaded and processed")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Important: 0 workers for streaming
        pin_memory=cfg['training'].get('pin_memory', True)
    )
    
    # Use a small validation set (first 10 sequences)
    val_indices = list(range(min(10, len(train_dataset))))
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Rest of training is same as resumable version
    # ... [continue with model creation and training loop from train_forecast_resumable.py]
    
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
            checkpoints = sorted(checkpoint_dir.glob('forecast_epoch_*.pt'))
            if checkpoints:
                args.resume = str(checkpoints[-1])
            else:
                args.resume = None
        
        if args.resume and Path(args.resume).exists():
            start_epoch, best_val_loss = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler
            )
    
    # Training loop with dataset growth monitoring
    print("\nStarting streaming training...")
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        # Check dataset growth
        stats = train_dataset.get_current_stats()
        print(f"Dataset size: {stats['sequences']} sequences")
        print(f"{'='*50}")
        
        # Recreate dataloader if dataset has grown significantly
        if epoch > 0 and stats['sequences'] > len(train_loader.dataset) * 1.1:
            print(f"Dataset grew from {len(train_loader.dataset)} to {stats['sequences']} sequences")
            print("Recreating dataloader...")
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg['training']['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=cfg['training'].get('pin_memory', True)
            )
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler, cfg,
            epoch, checkpoint_dir, save_every_n_batches=50
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate periodically
        if (epoch + 1) % cfg['training']['validate_every'] == 0:
            val_loss = validate(model, val_loader, device, cfg)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            checkpoint_path = checkpoint_dir / f'forecast_epoch_{epoch:03d}.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': cfg,
                'dataset_size': len(train_dataset)
            }, checkpoint_path, is_best=is_best)
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'final_dataset_size': len(train_dataset)
    }
    torch.save(final_checkpoint, checkpoint_dir / 'forecast_model_final.pt')
    print("\n✓ Training complete! Final model saved.")
    print(f"Final dataset size: {len(train_dataset)} sequences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_streaming_4090.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (or "latest")')
    args = parser.parse_args()
    main(args)
EOF

python train_streaming_forecast.py --config configs/train_streaming_4090.yaml

# Cleanup
echo ""
echo "Training complete! Stopping background processes..."
kill $DOWNLOAD_PID 2>/dev/null
kill $PREPROCESS_PID 2>/dev/null

echo ""
echo "=== Streaming Pipeline Complete! ==="
echo "Check logs in logs/streaming/ for details"