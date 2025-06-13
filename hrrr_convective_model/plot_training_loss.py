#!/usr/bin/env python3
"""
Plot training loss curves for DDPM training.
Reads from both log files and saved metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import re
from datetime import datetime
import argparse


def parse_log_file(log_path):
    """Parse training log file for loss values."""
    losses = []
    steps = []
    val_losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match training loss lines
            match = re.search(r'Epoch \d+:\s+\d+%.*avg=(\d+\.\d+)', line)
            if match:
                losses.append(float(match.group(1)))
                
            # Match validation loss lines
            val_match = re.search(r'\[Step (\d+)\] Train: (\d+\.\d+), Val: (\d+\.\d+)', line)
            if val_match:
                step = int(val_match.group(1))
                train_loss = float(val_match.group(2))
                val_loss = float(val_match.group(3))
                val_losses.append((step, val_loss))
                
            # Match epoch completion lines
            epoch_match = re.search(r'Epoch (\d+): Average Loss = (\d+\.\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                avg_loss = float(epoch_match.group(2))
                steps.append((epoch, avg_loss))
    
    return losses, steps, val_losses


def plot_losses(checkpoint_dir='checkpoints/diffusion_fullres_final'):
    """Plot training and validation losses."""
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('DDPM Training Progress - Full Resolution (1059x1799)', fontsize=16)
    
    # Try to load from metrics.json first
    metrics_path = Path(checkpoint_dir) / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Plot epoch-level losses
        if 'train_loss' in metrics and metrics['train_loss']:
            epochs, losses = zip(*metrics['train_loss'])
            ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss by Epoch')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add annotation for latest loss
            latest_loss = losses[-1]
            ax1.annotate(f'Latest: {latest_loss:.4f}', 
                        xy=(epochs[-1], latest_loss),
                        xytext=(epochs[-1]-5, latest_loss+0.02),
                        arrowprops=dict(arrowstyle='->'))
        
        # Plot validation losses
        if 'val_loss' in metrics and metrics['val_loss']:
            steps, val_losses = zip(*metrics['val_loss'])
            ax2.plot(steps, val_losses, 'r-', linewidth=2, label='Validation Loss')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss by Step')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
    # Also parse log file for more granular data
    log_path = Path('logs/diffusion_fullres_final.log')
    if log_path.exists():
        batch_losses, epoch_losses, val_losses = parse_log_file(log_path)
        
        # Create a third subplot for batch-level losses
        if batch_losses:
            fig2, ax3 = plt.subplots(1, 1, figsize=(12, 6))
            
            # Smooth the batch losses with moving average
            window = 50
            if len(batch_losses) > window:
                smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
                ax3.plot(range(len(smoothed)), smoothed, 'g-', alpha=0.8, 
                        label=f'Smoothed (window={window})')
            
            # Plot raw batch losses with transparency
            ax3.plot(batch_losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw Batch Loss')
            
            ax3.set_xlabel('Batch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Batch-Level Training Loss')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Add statistics
            if batch_losses:
                stats_text = (f'Current Loss: {batch_losses[-1]:.4f}\n'
                             f'Min Loss: {min(batch_losses):.4f}\n'
                             f'Batches: {len(batch_losses)}')
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig('training_loss_batches.png', dpi=150, bbox_inches='tight')
            print("Saved batch-level loss plot to training_loss_batches.png")
    
    plt.tight_layout()
    plt.savefig('training_loss_curves.png', dpi=150, bbox_inches='tight')
    print("Saved loss curves to training_loss_curves.png")
    
    # Also create a simple text summary
    summary_path = Path('training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write("Model: Ultra-Minimal DDPM (81K params)\n")
        f.write("Resolution: 1059 x 1799 (Full 3km HRRR)\n")
        f.write("Data: Real HRRR - 14 days, 7 variables\n\n")
        
        if metrics_path.exists() and 'train_loss' in metrics and metrics['train_loss']:
            f.write(f"Epochs Completed: {len(metrics['train_loss'])}\n")
            f.write(f"Latest Training Loss: {metrics['train_loss'][-1][1]:.4f}\n")
            
            # Calculate improvement
            if len(metrics['train_loss']) > 1:
                initial_loss = metrics['train_loss'][0][1]
                current_loss = metrics['train_loss'][-1][1]
                improvement = (initial_loss - current_loss) / initial_loss * 100
                f.write(f"Loss Improvement: {improvement:.1f}%\n")
    
    print(f"Saved training summary to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='checkpoints/diffusion_fullres_final',
                       help='Checkpoint directory')
    args = parser.parse_args()
    
    plot_losses(args.checkpoint_dir)