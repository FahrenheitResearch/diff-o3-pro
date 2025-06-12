#!/bin/bash
# Extended training pipeline with 1 week of HRRR data
# Optimized for 4090 GPU training

set -e  # Exit on error

echo "=========================================="
echo "DEF Extended Training Pipeline"
echo "=========================================="
echo "This will download ~1 week of HRRR data and train both models"
echo "Estimated download: ~50-100GB"
echo "Estimated training time on 4090: 12-24 hours"
echo "=========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw/training_week"
ZARR_DIR="$DATA_DIR/zarr/training_week"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"

# Create directories
echo "1. Creating directories..."
mkdir -p $RAW_DIR $ZARR_DIR $CHECKPOINT_DIR $LOG_DIR
mkdir -p $CHECKPOINT_DIR/diffusion

# Download 1 week of HRRR data (4 cycles per day x 7 days = 28 files)
echo -e "\n2. Downloading 1 week of HRRR data..."
echo "This will take a while (downloading ~28 files, ~2GB each)..."

# Calculate date range (7 days ago to yesterday)
END_DATE=$(date -d "yesterday" +%Y%m%d)
START_DATE=$(date -d "7 days ago" +%Y%m%d)

echo "Downloading from $START_DATE to $END_DATE"
echo "Cycles: 00Z, 06Z, 12Z, 18Z (4 per day)"

python scripts/download_hrrr.py \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --hours 0 6 12 18 \
    --output-dir $RAW_DIR \
    --parallel 4

# Count downloaded files
FILE_COUNT=$(ls $RAW_DIR/*.grib2 2>/dev/null | wc -l || echo 0)
echo "Downloaded $FILE_COUNT GRIB2 files"

# Convert to Zarr
echo -e "\n3. Converting GRIB2 to Zarr..."
echo "This will process all $FILE_COUNT files..."
python scripts/preprocess_hrrr_fixed.py --src $RAW_DIR --out $ZARR_DIR

# Compute statistics
echo -e "\n4. Computing normalization statistics..."
python scripts/compute_stats.py --zarr $ZARR_DIR/hrrr.zarr --out $ZARR_DIR/stats.json

# Create extended training configuration
echo -e "\n5. Creating extended training configuration..."
cat > configs/extended_training.yaml << EOF
data:
  zarr: "$ZARR_DIR/hrrr.zarr"
  stats: "$ZARR_DIR/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]

training:
  lead_hours: 1
  batch_size: 2  # Can increase if memory allows
  num_workers: 4
  epochs: 50  # More epochs for better convergence
  lr: 1.0e-4
  gradient_accumulation_steps: 4  # Effective batch size = 8
  checkpoint_every: 5
  validate_every: 1
  base_features: 32  # Use 64 if memory allows
  warmup_epochs: 2
  lr_schedule: "cosine"
  
  # Data augmentation
  random_crop: false  # Full resolution
  horizontal_flip: false  # Weather has directionality
  
  # Training stability
  clip_grad_norm: 1.0
  weight_decay: 1.0e-5

diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
  guidance_weight: 0.1
  dropout_prob: 0.1
  sampler: "dpm_solver_pp"
  solver_order: 2
  num_steps: 50  # Can reduce to 10-20 for faster inference
  ema_decay: 0.9999
  
  # Training specific
  loss_type: "mse"
  p2_loss_weight: 0.0  # Standard loss weighting

ensemble:
  num_members: 16  # Full ensemble
  perturbation_samples: 8
  blend_weight: 0.95

evaluation:
  eval_variables: ["T2M", "U10", "V10", "REFC"]
  eval_lead_times: [1, 3, 6, 12, 24]
  metrics: ["rmse", "crps", "spread"]
EOF

# Train forecast model
echo -e "\n6. Training deterministic forecast model..."
echo "This will take several hours on 4090..."
python train_forecast.py --config configs/extended_training.yaml \
    2>&1 | tee $LOG_DIR/forecast_training.log

# Check if forecast model was trained successfully
if [ -f "$CHECKPOINT_DIR/forecast_model_best.pt" ]; then
    echo "✓ Forecast model training complete!"
else
    echo "✗ Forecast model training failed!"
    exit 1
fi

# Train diffusion model
echo -e "\n7. Training diffusion model..."
echo "This will take several more hours..."
python train_diffusion.py --config configs/extended_training.yaml \
    2>&1 | tee $LOG_DIR/diffusion_training.log

# Check if diffusion model was trained successfully
if [ -f "$CHECKPOINT_DIR/diffusion/best_model.pt" ]; then
    echo "✓ Diffusion model training complete!"
else
    echo "✗ Diffusion model training failed!"
    exit 1
fi

# Generate a test forecast
echo -e "\n8. Generating test ensemble forecast..."
# Get the latest time from the training data
FORECAST_DATE=$(python -c "
import zarr
import numpy as np
store = zarr.open('$ZARR_DIR/hrrr.zarr', 'r')
times = store['time'][:]
# Use a time from the middle of the dataset
mid_idx = len(times) // 2
mid_time = times[mid_idx]
# Convert to datetime format for inference
# Assuming times are hours since epoch
from datetime import datetime, timedelta
base = datetime(2020, 1, 1)  # Adjust base as needed
dt = base + timedelta(hours=int(mid_time))
print(dt.strftime('%Y-%m-%d %H'))
")
echo "Test forecast initialization: $FORECAST_DATE"

python inference_ensemble.py \
    --config configs/extended_training.yaml \
    --start-date "$FORECAST_DATE" \
    --cycles 1 \
    --max-lead-hours 24 \
    --output-dir forecasts/extended_test \
    --device cuda

# Create visualizations
echo -e "\n9. Creating visualization..."
python plot_all_hours.py

echo -e "\n=========================================="
echo "Extended training pipeline complete!"
echo "=========================================="
echo "Models saved in: $CHECKPOINT_DIR"
echo "Training logs in: $LOG_DIR"
echo "Test forecast in: forecasts/extended_test"
echo ""
echo "To run inference on new data:"
echo "  python inference_ensemble.py --config configs/extended_training.yaml --start-date 'YYYY-MM-DD HH' --device cuda"
echo ""
echo "Training summary:"
echo "  - Data: $FILE_COUNT HRRR files (~1 week)"
echo "  - Forecast model: 50 epochs"
echo "  - Diffusion model: 50 epochs"
echo "=========================================="