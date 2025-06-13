#!/bin/bash
# Quick training test with existing data

set -e

echo "=========================================="
echo "Quick Training Test - Using Existing Data"
echo "=========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw/test_quick"
ZARR_DIR="$DATA_DIR/zarr/test_quick"
CHECKPOINT_DIR="checkpoints/test_quick"
LOG_DIR="logs/test_quick"

# Minimal training parameters for quick test
BATCH_SIZE=1
GRAD_ACCUM=2
BASE_FEATURES=32
FORECAST_EPOCHS=2  # Just 2 epochs to test
DIFFUSION_EPOCHS=2

# Create directories
echo "1. Creating test directories..."
mkdir -p $RAW_DIR $ZARR_DIR $CHECKPOINT_DIR $LOG_DIR
mkdir -p $CHECKPOINT_DIR/diffusion

# Copy some existing files to test directory
echo -e "\n2. Copying existing GRIB files for test..."
# Copy a few files from streaming directory
cp data/raw/streaming/hrrr.20250608.t00z.wrfprsf0[0-6].grib2 $RAW_DIR/ 2>/dev/null || true
cp data/raw/streaming/hrrr.20250609.t00z.wrfprsf0[0-6].grib2 $RAW_DIR/ 2>/dev/null || true

FILE_COUNT=$(ls $RAW_DIR/*.grib2 2>/dev/null | wc -l || echo 0)
echo "Found $FILE_COUNT GRIB2 files for testing"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "Error: No test files available!"
    exit 1
fi

# Convert to Zarr
echo -e "\n3. Converting GRIB2 to Zarr..."
python scripts/preprocess_hrrr_forecast.py --src $RAW_DIR --out $ZARR_DIR

# Compute statistics
echo -e "\n4. Computing normalization statistics..."
python scripts/compute_stats.py --zarr $ZARR_DIR/hrrr.zarr --out $ZARR_DIR/stats.json

# Create minimal test configuration
echo -e "\n5. Creating test configuration..."
cat > configs/test_quick.yaml << EOF
data:
  zarr: "$ZARR_DIR/hrrr.zarr"
  stats: "$ZARR_DIR/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]

training:
  lead_hours: 6  # Just test up to 6 hours
  batch_size: $BATCH_SIZE
  num_workers: 2
  epochs: $FORECAST_EPOCHS
  lr: 2.0e-4
  gradient_accumulation_steps: $GRAD_ACCUM
  checkpoint_every: 1
  validate_every: 1
  base_features: $BASE_FEATURES
  warmup_epochs: 0
  lr_schedule: "constant"
  sequence_length: 7  # F00-F06
  
  # Disable fancy features for quick test
  use_temporal_encoding: false
  curriculum_learning: false
  
  # Memory optimization
  pin_memory: true
  persistent_workers: false
  
  # Mixed precision
  use_amp: true
  
diffusion:
  timesteps: 100  # Fewer timesteps for quick test
  beta_schedule: "linear"
  guidance_weight: 0.1
  dropout_prob: 0.1
  sampler: "ddpm"
  num_steps: 10
  ema_decay: 0.999
  
  # Training
  epochs: $DIFFUSION_EPOCHS
  batch_size: $BATCH_SIZE
  gradient_accumulation_steps: $GRAD_ACCUM

ensemble:
  num_members: 2
  perturbation_samples: 2
  blend_weight: 0.95

evaluation:
  eval_variables: ["T2M", "REFC"]
  eval_lead_times: [1, 6]
  metrics: ["rmse"]
EOF

echo -e "\n6. GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Quick forecast model test
echo -e "\n7. Testing forecast model training (2 epochs)..."
timeout 300 python train_forecast.py --config configs/test_quick.yaml \
    2>&1 | tee $LOG_DIR/forecast_test.log || true

if [ -f "$CHECKPOINT_DIR/forecast_model_best.pt" ]; then
    echo "✓ Forecast model training works!"
else
    echo "✗ Forecast model training failed - check logs"
    exit 1
fi

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Quick diffusion model test
echo -e "\n8. Testing diffusion model training (2 epochs)..."
timeout 300 python train_diffusion.py --config configs/test_quick.yaml \
    2>&1 | tee $LOG_DIR/diffusion_test.log || true

if [ -f "$CHECKPOINT_DIR/diffusion/best_model.pt" ]; then
    echo "✓ Diffusion model training works!"
else
    echo "✗ Diffusion model training failed - check logs"
    exit 1
fi

echo -e "\n=========================================="
echo "Test Complete - Training Pipeline Works!"
echo "=========================================="
echo "You can now run the full overnight training with confidence."
echo "Clean up test files with: rm -rf $RAW_DIR $ZARR_DIR $CHECKPOINT_DIR $LOG_DIR"
echo "==========================================