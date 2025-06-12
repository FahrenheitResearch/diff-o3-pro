#!/bin/bash
# Complete training pipeline with forecast sequences

echo "=== HRRR Forecast Training Pipeline ==="
echo "This will download F00-F18 forecast data and train the model"
echo ""

# Step 1: Download forecast data
echo "Step 1: Downloading HRRR forecast data (F00-F18)..."
./download_training_data.sh

# Step 2: Convert to Zarr with forecast sequences
echo ""
echo "Step 2: Converting to Zarr format with forecast sequences..."
python scripts/preprocess_hrrr_forecast.py \
    --src data/raw/training_forecast \
    --out data/zarr/training_forecast

# Step 3: Compute statistics
echo ""
echo "Step 3: Computing normalization statistics..."
python scripts/compute_stats.py \
    --zarr data/zarr/training_forecast/hrrr.zarr \
    --out data/zarr/training_forecast/stats.json

# Step 4: Update config to use new data
echo ""
echo "Step 4: Creating forecast training config..."
cat > configs/train_forecast_4090.yaml << EOF
data:
  zarr: "data/zarr/training_forecast/hrrr.zarr"
  stats: "data/zarr/training_forecast/stats.json"
  variables: ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"]

training:
  lead_hours: 1
  batch_size: 1
  num_workers: 2
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
  
diffusion:
  timesteps: 1000
  beta_schedule: "cosine"
  guidance_weight: 0.1
  dropout_prob: 0.1
  sampler: "dpm_solver_pp"
  solver_order: 2
  num_steps: 20
  ema_decay: 0.999
  
  # Training
  epochs: 20
  batch_size: 1
  gradient_accumulation_steps: 8

ensemble:
  num_members: 8
  perturbation_samples: 4
  blend_weight: 0.95

evaluation:
  eval_variables: ["T2M", "REFC"]
  eval_lead_times: [1, 6]
  metrics: ["rmse", "crps"]
EOF

# Step 5: Train the model (with resume support)
echo ""
echo "Step 5: Training forecast model on sequences..."
# Check if we should resume
if [ -f "checkpoints/resume_state.pt" ] || ls checkpoints/forecast_epoch_*.pt 1> /dev/null 2>&1; then
    echo "Found existing checkpoints, resuming training..."
    python train_forecast_resumable.py --config configs/train_forecast_4090.yaml --resume latest
else
    echo "Starting fresh training..."
    python train_forecast_resumable.py --config configs/train_forecast_4090.yaml
fi

echo ""
echo "=== Training Complete! ==="
echo "The model has been trained on realistic forecast sequences (F00->F01, F01->F02, etc.)"
echo "This should produce much better results than training only on analysis data."