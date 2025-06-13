#!/bin/bash
# Launch memory-optimized deterministic training

echo "Starting memory-optimized deterministic training..."
echo "Configuration:"
echo "  - Batch size: 1 (with gradient accumulation: 16)"
echo "  - Model size: 40 base features"
echo "  - Memory optimization: enabled"
echo "  - Training for 200 epochs"
echo ""

# Create log directory
mkdir -p logs/deterministic

# Activate conda environment and launch training
nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate hrrr_maps && python train_deterministic_optimized.py --config configs/deterministic.yaml --epochs 200" > logs/deterministic/training_output.log 2>&1 &

# Get PID
PID=$!
echo "Training started with PID: $PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/deterministic/training_output.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "The model should maintain <1GB VRAM usage throughout training."
echo "Expected training time: ~10-15 hours for 200 epochs"