#!/bin/bash
# Monitor the 3-epoch training progress

echo "Monitoring Conditional DDPM Training (3 epochs)..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""

# Show epoch completions
tail -f logs/conditional_faithful.log | grep -E "(Epoch.*complete|Step.*loss=|TRAINING COMPLETE)" | while read line; do
    echo "[$(date +%H:%M:%S)] $line"
done