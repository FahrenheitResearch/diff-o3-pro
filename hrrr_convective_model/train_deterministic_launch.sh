#!/bin/bash
# Launch deterministic training with all fixes

echo "Starting deterministic weather model training..."
echo "This will train for 200 epochs with 1-hour forecasts"
echo "Using residual UNet + spectral loss + real timestamps"
echo ""

# Create log directory
mkdir -p logs/deterministic

# Launch training with output logging
python train_deterministic.py \
    --config configs/deterministic.yaml \
    --epochs 200 \
    2>&1 | tee logs/deterministic/training_output.log &

# Get PID
PID=$!
echo "Training started with PID: $PID"
echo "Logs: logs/deterministic/training_output.log"
echo ""
echo "Monitor with: tail -f logs/deterministic/training_output.log"
echo "Check GPU: nvidia-smi -l 1"
echo ""
echo "Training will take approximately 8-12 hours for 200 epochs"