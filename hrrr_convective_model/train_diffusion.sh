#!/bin/bash
# Launch diffusion model training

echo "Starting diffusion model training..."
echo "This trains the ensemble perturbation generator"
echo ""
echo "Requirements:"
echo "  - Trained deterministic forecast model"
echo "  - Will load from: checkpoints/deterministic/best_model.pt"
echo ""

# Check if forecast model exists
if [ ! -f "checkpoints/deterministic/best_model.pt" ]; then
    echo "ERROR: Forecast model not found!"
    echo "Please train the deterministic model first with ./train_optimized.sh"
    exit 1
fi

echo "Configuration:"
echo "  - Epochs: 30"
echo "  - Batch size: 2"
echo "  - Base features: 32"
echo "  - Expected VRAM: <2GB"
echo ""

# Create directories
mkdir -p logs/diffusion
mkdir -p checkpoints/diffusion

# Launch training
nohup python train_diffusion_fixed.py \
    --config configs/diffusion.yaml \
    --forecast_checkpoint checkpoints/deterministic/best_model.pt \
    --epochs 30 \
    > logs/diffusion/training_output.log 2>&1 &

PID=$!
echo "Diffusion training started with PID: $PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/diffusion/training_output.log"
echo ""
echo "Expected training time: ~2-3 hours for 30 epochs"
echo ""
echo "After training completes, run ensemble inference with:"
echo "  python inference_ensemble.py"