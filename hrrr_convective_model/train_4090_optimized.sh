#!/bin/bash
# Optimized training pipeline for RTX 4090 (24GB VRAM)
# Downloads 3-5 days of data for manageable training

set -e  # Exit on error

echo "=========================================="
echo "DEF Training Pipeline - 4090 Optimized"
echo "=========================================="
echo "Optimized for RTX 4090 with 24GB VRAM"
echo "Will download 3-5 days of HRRR data"
echo "=========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw/training_4090"
ZARR_DIR="$DATA_DIR/zarr/training_4090"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs/training_4090"

# Training parameters optimized for 4090 - OVERNIGHT RUN
BATCH_SIZE=1  # Full resolution requires small batch
GRAD_ACCUM=16  # Larger effective batch size = 16 for stability
BASE_FEATURES=48  # Increased capacity for temporal patterns
FORECAST_EPOCHS=60  # More epochs for sequence learning
DIFFUSION_EPOCHS=40  # More diffusion training

# Create directories
echo "1. Creating directories..."
mkdir -p $RAW_DIR $ZARR_DIR $CHECKPOINT_DIR $LOG_DIR
mkdir -p $CHECKPOINT_DIR/diffusion

# Download 10 days of HRRR data with F00-F18 forecast hours for better coverage
echo -e "\n2. Downloading 10 days of HRRR data (F00-F18)..."

# Get last 10 days for more training data
END_DATE=$(date -d "yesterday" +%Y%m%d)
START_DATE=$(date -d "10 days ago" +%Y%m%d)

# Generate date array
DATES=()
current=$START_DATE
while [[ $current -le $END_DATE ]]; do
    DATES+=($current)
    current=$(date -d "$current + 1 day" +%Y%m%d)
done

echo "Downloading dates: ${DATES[@]}"
echo "Cycles: 00Z only"
echo "Forecast hours: F00-F18"

# Download each date with forecast hours 0-18
for DATE in "${DATES[@]}"; do
    echo "Downloading $DATE..."
    python scripts/download_hrrr.py \
        --start-date $DATE \
        --end-date $DATE \
        --hours 0 \
        --forecast-hours "0-18" \
        --output-dir $RAW_DIR \
        --parallel 2  # Limit parallel downloads
done

# Count files
FILE_COUNT=$(ls $RAW_DIR/*.grib2 2>/dev/null | wc -l || echo 0)
echo "Downloaded $FILE_COUNT GRIB2 files"

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

# Create 4090-optimized configuration
echo -e "\n5. Creating 4090-optimized configuration..."
cat > configs/train_4090.yaml << EOF
data:
  zarr: "$ZARR_DIR/hrrr.zarr"
  stats: "$ZARR_DIR/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]

training:
  lead_hours: 18  # Train on F00-F18
  batch_size: $BATCH_SIZE
  num_workers: 4  # More workers for larger dataset
  epochs: $FORECAST_EPOCHS
  lr: 1.0e-4  # Lower LR for stable sequence learning
  gradient_accumulation_steps: $GRAD_ACCUM
  checkpoint_every: 5
  validate_every: 2
  base_features: $BASE_FEATURES
  warmup_epochs: 3  # More warmup for sequence learning
  lr_schedule: "cosine_with_restarts"
  sequence_length: 19  # 19 timesteps (F00-F18)
  
  # Sequence learning improvements
  use_temporal_encoding: true
  temporal_dropout: 0.1
  curriculum_learning: true
  curriculum_schedule: "linear"  # Start with shorter sequences
  min_sequence_length: 6  # Start with 6-hour forecasts
  sequence_increment_epoch: 10  # Add more hours every 10 epochs
  
  # Memory optimization
  pin_memory: true
  persistent_workers: false  # Save memory
  
  # Stability
  clip_grad_norm: 1.0
  weight_decay: 1.0e-5
  
  # Mixed precision
  use_amp: true
  
diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
  guidance_weight: 0.2  # Stronger guidance for sequences
  dropout_prob: 0.15  # More regularization
  sampler: "dpm_solver_pp"
  solver_order: 2
  num_steps: 50  # More steps for better quality
  ema_decay: 0.9995  # More stable EMA
  
  # Training
  epochs: $DIFFUSION_EPOCHS
  batch_size: $BATCH_SIZE
  gradient_accumulation_steps: $GRAD_ACCUM
  
  # Augmentation for temporal consistency
  noise_schedule: "adaptive"  # Adapt noise to forecast hour
  temporal_weight: 0.3  # Weight temporal consistency
  augment_probability: 0.2
  augmentations:
    - "gaussian_noise"
    - "temporal_shift"
    - "intensity_scale"

ensemble:
  num_members: 8  # Reduced for memory
  perturbation_samples: 4
  blend_weight: 0.95

evaluation:
  eval_variables: ["T2M", "REFC"]
  eval_lead_times: [1, 6, 12, 18]
  metrics: ["rmse", "crps"]
EOF

# Monitor GPU memory
echo -e "\n6. GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Train forecast model with memory monitoring
echo -e "\n7. Training deterministic forecast model..."
echo "Expected time: 2-4 hours on RTX 4090"

# Clear cache before training
python -c "import torch; torch.cuda.empty_cache()"

# Train with regular checkpointing
python train_forecast.py --config configs/train_4090.yaml \
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
echo "Expected time: 1-3 hours on RTX 4090"

python train_diffusion.py --config configs/train_4090.yaml \
    2>&1 | tee $LOG_DIR/diffusion_training.log

if [ -f "$CHECKPOINT_DIR/diffusion/best_model.pt" ]; then
    echo "✓ Diffusion model training complete!"
else
    echo "✗ Diffusion model training failed!"
    exit 1
fi

# Quick validation forecast
echo -e "\n9. Generating validation forecast..."
python inference_ensemble.py \
    --config configs/train_4090.yaml \
    --start-date "2025-01-01 00" \
    --cycles 1 \
    --max-lead-hours 18 \
    --ensemble-size 4 \
    --device cuda \
    --output-dir forecasts/validation

# Summary statistics
echo -e "\n=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Training data: $FILE_COUNT files (10 days × 19 forecast hours)"
echo "Date range: $START_DATE to $END_DATE"
echo "Forecast hours: F00-F18"
echo "Forecast model: $FORECAST_EPOCHS epochs"
echo "Diffusion model: $DIFFUSION_EPOCHS epochs"
echo "Configuration: configs/train_4090.yaml"
echo ""
echo "Next steps:"
echo "1. Download latest HRRR: python scripts/download_hrrr.py --start-date \$(date +%Y%m%d) --hours \$(date +%H) --forecast-hours 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"
echo "2. Process: python scripts/preprocess_hrrr_forecast.py --src data/raw/new --out data/zarr/new"
echo "3. Forecast: python inference_ensemble.py --config configs/train_4090.yaml --device cuda"
echo "=========================================="

# Show final GPU memory usage
echo -e "\nFinal GPU state:"
nvidia-smi