#!/bin/bash
# Monitor diffusion training

echo "=== Diffusion Training Monitor ==="
echo "Process ID: 192432"
echo ""

# Check if process is running
if ps -p 192432 > /dev/null; then
    echo "✓ Training is running"
else
    echo "✗ Training process not found"
fi

echo ""
echo "=== Latest Training Progress ==="
tail -n 20 diffusion_training.log | grep -E "Epoch|loss:|avg_loss:"

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== Loss Statistics ==="
grep "avg_loss:" diffusion_training.log | tail -n 10

echo ""
echo "To see real-time progress: tail -f diffusion_training.log"
echo "To plot loss curves: python plot_training_progress.py --log diffusion_training.log"