#!/bin/bash
echo "Monitoring resumed training (epochs 4-13)..."
echo "Current status:"
tail -1 logs/conditional_faithful.log | grep -E "(Epoch|Step)"
echo ""
echo "Following progress (Ctrl+C to stop):"
tail -f logs/conditional_faithful.log | grep -E "(Epoch.*complete|Step.*loss=)" | while read line; do
    echo "[$(date +%H:%M:%S)] $line"
done