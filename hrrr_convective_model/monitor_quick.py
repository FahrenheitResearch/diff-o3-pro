#\!/usr/bin/env python3
import os
import time
import torch
from pathlib import Path

checkpoint_dir = Path('checkpoints/diffusion_faithful')
log_file = Path('logs/diffusion_training.log')

print('Monitoring faithful DDPM training...')
print('='*60)

# Check for latest checkpoint
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob('epoch_*.pt'))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(latest, map_location='cpu')
        print(f'Latest checkpoint: {latest.name}')
        print(f'Epoch: {checkpoint["epoch"]}')
        print(f'Loss: {checkpoint["loss"]:.6f}')
else:
    print('No checkpoints yet...')

# Check log tail
if log_file.exists():
    os.system(f'tail -n 5 {log_file}')
