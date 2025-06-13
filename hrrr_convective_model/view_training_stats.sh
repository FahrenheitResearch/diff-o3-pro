#!/bin/bash
# Quick training statistics viewer

echo "=== DDPM Training Statistics ==="
echo "Training on: REAL HRRR DATA (14 days)"
echo "Resolution: 1059 x 1799 (Full 3km)"
echo "Model: 81K parameters"
echo ""

# Get latest epoch info
LATEST=$(tail -20 logs/diffusion_fullres_final.log | grep "Epoch.*Average Loss" | tail -1)
echo "Latest: $LATEST"

# Count checkpoints
CHECKPOINTS=$(ls -1 checkpoints/diffusion_fullres_final/epoch_*.pt 2>/dev/null | wc -l)
echo "Checkpoints saved: $CHECKPOINTS"

# GPU usage
echo ""
echo "Current GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "Memory: %s/%s MB (%.1f%%), GPU Util: %s%%\n", $1, $2, ($1/$2)*100, $3}'

# Process info
PID=$(ps aux | grep "train_diffusion_fullres_final.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PID" ]; then
    echo "Training process: Running (PID: $PID)"
    RUNTIME=$(ps -o etime= -p $PID | xargs)
    echo "Runtime: $RUNTIME"
else
    echo "Training process: Not running"
fi

echo ""
echo "To view loss plots: Check training_loss_curves.png"
echo "To monitor live: python monitor_training_live.py"