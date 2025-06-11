#!/bin/bash
# Test script for the complete DEF pipeline
# This uses minimal data for quick testing

set -e  # Exit on error

echo "=========================================="
echo "DEF Pipeline Test Script"
echo "=========================================="

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
ZARR_DIR="$DATA_DIR/zarr/test"
CHECKPOINT_DIR="checkpoints"
FORECAST_DIR="forecasts_test"
EVAL_DIR="evaluation_test"

# Create directories
echo "1. Creating directories..."
mkdir -p $RAW_DIR $ZARR_DIR $CHECKPOINT_DIR $FORECAST_DIR $EVAL_DIR
mkdir -p $CHECKPOINT_DIR/diffusion

# Download sample HRRR data (just 2 files for testing)
echo -e "\n2. Downloading sample HRRR data..."
python scripts/download_sample_hrrr.py --output-dir $RAW_DIR --num-files 2

# Check if we have GRIB files
if [ ! "$(ls -A $RAW_DIR/*.grib2 2>/dev/null)" ]; then
    echo "No GRIB2 files found. Trying to download from yesterday..."
    python -c "
import subprocess
from datetime import datetime, timedelta
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
cmd = f'python scripts/download_hrrr.py --start-date {yesterday} --end-date {yesterday} --hours 0 6 --output-dir $RAW_DIR'
print(f'Running: {cmd}')
subprocess.run(cmd, shell=True)
"
fi

# Convert to Zarr (using simple converter for test)
echo -e "\n3. Converting GRIB2 to Zarr..."
python scripts/preprocess_hrrr_fixed.py --src $RAW_DIR --out $ZARR_DIR

# Compute statistics
echo -e "\n4. Computing normalization statistics..."
python scripts/compute_stats.py --zarr $ZARR_DIR/hrrr.zarr --out $DATA_DIR/stats.json

# Update config for test data
echo -e "\n5. Creating test configuration..."
cat > configs/test.yaml << EOF
data:
  zarr: "$ZARR_DIR/hrrr.zarr"
  stats: "$DATA_DIR/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]
training:
  lead_hours: 1
  batch_size: 1
  num_workers: 0
  epochs: 2  # Just 2 epochs for testing
  lr: 1.0e-4
  gradient_accumulation_steps: 2
  checkpoint_every: 1
  validate_every: 1
  base_features: 32  # Smaller model for testing
  warmup_epochs: 0
  lr_schedule: "cosine"

diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
  guidance_weight: 0.1
  dropout_prob: 0.1
  sampler: "dpm_solver_pp"
  solver_order: 2
  num_steps: 5  # Reduced for faster testing
  ema_decay: 0.9999

ensemble:
  num_members: 4  # Small ensemble for testing
  perturbation_samples: 2
  blend_weight: 0.95

evaluation:
  eval_variables: ["T2M", "U10", "V10", "REFC"]
  eval_lead_times: [1]
  metrics: ["rmse", "crps", "spread"]
EOF

# Run unit tests first
echo -e "\n6. Running unit tests..."
python tests/test_forward.py || echo "Note: Some tests may fail without GPU"

# Train forecast model (just 2 epochs)
echo -e "\n7. Training forecast model (2 epochs)..."
python train_forecast.py --config configs/test.yaml

# Train diffusion model (just 2 epochs)
echo -e "\n8. Training diffusion model (2 epochs)..."
python train_diffusion.py --config configs/test.yaml

# Generate ensemble forecast
echo -e "\n9. Generating ensemble forecast..."
# Get a recent date from the data
FORECAST_DATE=$(python -c "
import zarr
import pandas as pd
store = zarr.open('$ZARR_DIR/hrrr.zarr')
times = pd.to_datetime(store['time'][:])
print(times[0].strftime('%Y-%m-%d %H'))
")
echo "Forecast initialization time: $FORECAST_DATE"

python inference_ensemble.py \
    --config configs/test.yaml \
    --start-date "$FORECAST_DATE" \
    --cycles 1 \
    --max-lead-hours 3 \
    --output-dir $FORECAST_DIR \
    --device cpu  # Use CPU for compatibility

# Evaluate the forecast
echo -e "\n10. Evaluating ensemble forecast..."
python evaluate_ensemble.py \
    --forecast-dir $FORECAST_DIR \
    --obs-zarr $ZARR_DIR/hrrr.zarr \
    --config configs/test.yaml \
    --output-dir $EVAL_DIR \
    --lead-times 1

# Display results
echo -e "\n=========================================="
echo "Pipeline Test Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Checkpoints: $CHECKPOINT_DIR/"
echo "  - Forecasts: $FORECAST_DIR/"
echo "  - Evaluation: $EVAL_DIR/"
echo ""
echo "Check the following files:"
echo "  - $EVAL_DIR/ensemble_evaluation_results.csv"
echo "  - $EVAL_DIR/spread_skill_plots.png"
echo "  - $FORECAST_DIR/def_forecast_*.nc"