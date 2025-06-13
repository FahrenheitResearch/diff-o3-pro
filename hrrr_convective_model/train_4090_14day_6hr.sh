#!/bin/bash
# Optimized training pipeline for RTX 4090 - 30 days, 6-hour forecasts
# Downloads 30 days with all 4 cycles (00z, 06z, 12z, 18z) and F00-F06

set -e  # Exit on error

echo "=========================================="
echo "DEF Training Pipeline - 14 Day / 6-Hour"
echo "=========================================="
echo "Optimized for RTX 4090 with 24GB VRAM"
echo "Will download 14 days of HRRR data"
echo "Cycles: 00Z, 06Z, 12Z, 18Z"
echo "Forecast hours: F00-F06 (7 hours each)"
echo "=========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw/training_14day"
ZARR_DIR="$DATA_DIR/zarr/training_14day"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs/training_14day"

# Training parameters optimized for 4090 - ULTRA MEMORY CONSCIOUS
BATCH_SIZE=1  # Keep at 1 to avoid memory spikes
GRAD_ACCUM=32  # Even higher accumulation but same effective batch
BASE_FEATURES=40  # Further reduced to save ~2GB VRAM
FORECAST_EPOCHS=50  # Good amount for 392 files
DIFFUSION_EPOCHS=30

# Create directories
echo "1. Creating directories..."
mkdir -p $RAW_DIR $ZARR_DIR $CHECKPOINT_DIR $LOG_DIR
mkdir -p $CHECKPOINT_DIR/diffusion

# Download 14 days of HRRR data with F00-F06 forecast hours
echo -e "\n2. Downloading 14 days of HRRR data (F00-F06)..."

# Get last 14 days
END_DATE=$(date -d "yesterday" +%Y%m%d)
START_DATE=$(date -d "14 days ago" +%Y%m%d)

echo "Date range: $START_DATE to $END_DATE"
echo "Downloading all 4 daily cycles with 7 forecast hours each"

# Use the download script with proper parameters
python scripts/download_hrrr.py \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --hours 0 6 12 18 \
    --forecast-hours "0-6" \
    --output-dir $RAW_DIR \
    --parallel 4  # More parallel downloads

# Count files
FILE_COUNT=$(ls $RAW_DIR/*.grib2 2>/dev/null | wc -l || echo 0)
echo "Downloaded $FILE_COUNT GRIB2 files"
echo "Expected: ~392 files (14 days × 4 cycles × 7 hours)"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "Error: No files downloaded!"
    exit 1
fi

# Convert to Zarr (using forecast-aware preprocessor)
echo -e "\n3. Converting GRIB2 to Zarr (with forecast hours)..."
python scripts/preprocess_hrrr_forecast.py --src $RAW_DIR --out $ZARR_DIR

# Compute statistics
echo -e "\n4. Computing normalization statistics..."
python scripts/compute_stats.py --zarr $ZARR_DIR/hrrr.zarr --out $ZARR_DIR/stats.json

# Create 14-day optimized configuration
echo -e "\n5. Creating 14-day training configuration..."
cat > configs/train_14day.yaml << EOF
data:
  zarr: "$ZARR_DIR/hrrr.zarr"
  stats: "$ZARR_DIR/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]

training:
  lead_hours: 6  # Train on F00-F06
  batch_size: $BATCH_SIZE
  num_workers: 2  # Reduce workers to save memory
  epochs: $FORECAST_EPOCHS
  lr: 2.0e-4  # Good learning rate for this data size
  gradient_accumulation_steps: $GRAD_ACCUM
  checkpoint_every: 5
  validate_every: 2
  base_features: $BASE_FEATURES
  warmup_epochs: 2
  lr_schedule: "cosine_with_restarts"
  sequence_length: 7  # 7 timesteps (F00-F06)
  
  # Sequence learning improvements
  use_temporal_encoding: true
  temporal_dropout: 0.1
  
  # No curriculum learning needed for 6-hour forecasts
  curriculum_learning: false
  
  # Memory optimization
  pin_memory: false  # Disable to save memory
  persistent_workers: false  # Disable to save memory
  
  # Stability
  clip_grad_norm: 1.0
  weight_decay: 1.0e-5
  
  # Mixed precision
  use_amp: true
  amp_dtype: "float16"  # Use fp16 to save memory
  
  # Data augmentation - disabled to save memory
  augment_probability: 0.0
  
  # Additional memory optimizations
  checkpoint_activations: false  # Don't save activations
  empty_cache_freq: 10  # Clear cache every 10 batches
  
diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
  guidance_weight: 0.15
  dropout_prob: 0.1
  sampler: "dpm_solver_pp"
  solver_order: 2
  num_steps: 25  # Balanced quality/speed
  ema_decay: 0.9995
  
  # Training
  epochs: $DIFFUSION_EPOCHS
  batch_size: $BATCH_SIZE
  gradient_accumulation_steps: $GRAD_ACCUM
  
  # Temporal consistency for 6-hour sequences
  temporal_weight: 0.2
  noise_schedule: "linear"

ensemble:
  num_members: 10  # More members with better data
  perturbation_samples: 5
  blend_weight: 0.9

evaluation:
  eval_variables: ["T2M", "REFC", "U10", "V10"]
  eval_lead_times: [1, 3, 6]
  metrics: ["rmse", "crps", "mae"]
EOF

# Monitor GPU memory
echo -e "\n6. GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Train forecast model with memory monitoring
echo -e "\n7. Training deterministic forecast model..."
echo "Expected time: 4-6 hours on RTX 4090"

# Clear cache before training
python -c "import torch; torch.cuda.empty_cache()"

# Train with resumable checkpointing
python train_forecast_resumable.py --config configs/train_14day.yaml \
    2>&1 | tee $LOG_DIR/forecast_training.log &

# Monitor GPU usage during training
TRAIN_PID=$!
while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $LOG_DIR/gpu_usage.log
    sleep 30
done

wait $TRAIN_PID
FORECAST_STATUS=$?

if [ $FORECAST_STATUS -eq 0 ] && [ -f "$CHECKPOINT_DIR/forecast_model_best.pt" ]; then
    echo "✓ Forecast model training complete!"
else
    echo "✗ Forecast model training failed!"
    exit 1
fi

# Clear cache before diffusion training
python -c "import torch; torch.cuda.empty_cache()"

# Train diffusion model
echo -e "\n8. Training diffusion model..."
echo "Expected time: 3-4 hours on RTX 4090"

python train_diffusion.py --config configs/train_14day.yaml \
    2>&1 | tee $LOG_DIR/diffusion_training.log

if [ -f "$CHECKPOINT_DIR/diffusion/best_model.pt" ]; then
    echo "✓ Diffusion model training complete!"
else
    echo "✗ Diffusion model training failed!"
    exit 1
fi

# Generate validation forecast
echo -e "\n9. Generating validation forecast..."
python inference_ensemble.py \
    --config configs/train_14day.yaml \
    --start-date "$(date +%Y-%m-%d) 00" \
    --cycles 1 \
    --max-lead-hours 6 \
    --ensemble-size 10 \
    --device cuda \
    --output-dir forecasts/validation_14day

# Summary statistics
echo -e "\n=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Training data: $FILE_COUNT files"
echo "Date range: $START_DATE to $END_DATE"
echo "Cycles: 00Z, 06Z, 12Z, 18Z"
echo "Forecast hours: F00-F06"
echo "Forecast model: $FORECAST_EPOCHS epochs"
echo "Diffusion model: $DIFFUSION_EPOCHS epochs"
echo "Configuration: configs/train_14day.yaml"
echo ""
echo "Next steps:"
echo "1. Download latest HRRR: python scripts/download_hrrr.py --start-date \$(date +%Y%m%d) --hours \$(date +%H) --forecast-hours 0-6"
echo "2. Process: python scripts/preprocess_hrrr_forecast.py --src data/raw/new --out data/zarr/new"
echo "3. Forecast: python inference_ensemble.py --config configs/train_14day.yaml --device cuda --max-lead-hours 6"
echo "=========================================="

# Show final GPU memory usage
echo -e "\nFinal GPU state:"
nvidia-smi