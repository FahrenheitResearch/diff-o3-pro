#!/usr/bin/env python3
"""Plot training progress from the log file."""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file, is_diffusion=False):
    """Parse training log to extract metrics."""
    epochs = []
    train_losses = []
    val_losses = []
    train_rmse = []
    val_rmse = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_epoch = -1
    for line in lines:
        # Parse epoch number
        epoch_match = re.search(r'Epoch (\d+)/\d+', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # Parse training loss
        train_match = re.search(r'Epoch \d+ - Train loss: ([\d.]+)', line)
        if train_match and current_epoch >= 0:
            epochs.append(current_epoch)
            train_losses.append(float(train_match.group(1)))
            
        # Parse validation loss and RMSE
        val_match = re.search(r'Val loss: ([\d.]+), Val RMSE: ([\d.]+)', line)
        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_rmse.append(float(val_match.group(2)))
            
        # Parse learning rate
        lr_match = re.search(r'Learning rate: ([\d.e-]+)', line)
        if lr_match:
            learning_rates.append(float(lr_match.group(1)))
    
    # Also try to parse from progress bar lines for real-time data
    batch_losses = []
    batch_rmse = []
    
    # Different parsing for diffusion logs
    if is_diffusion:
        # Parse diffusion logs: 'loss': 0.1234, 'avg_loss': 0.5678
        for line in lines:
            # Parse from progress bar
            loss_match = re.search(r"'loss': ([\d.]+)", line)
            avg_match = re.search(r"'avg_loss': ([\d.]+)", line)
            if loss_match:
                batch_losses.append(float(loss_match.group(1)))
            elif avg_match:
                # Also capture average losses
                batch_losses.append(float(avg_match.group(1)))
    else:
        # Parse regular training logs
        for line in lines:
            # Parse from progress bar: loss=0.0865, rmse=8.52e-5
            progress_match = re.search(r'loss=([\d.]+), rmse=([\d.e-]+)', line)
            if progress_match:
                batch_losses.append(float(progress_match.group(1)))
                try:
                    rmse_val = float(progress_match.group(2))
                    batch_rmse.append(rmse_val)
                except:
                    pass
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rmse': val_rmse,
        'learning_rates': learning_rates,
        'batch_losses': batch_losses,
        'batch_rmse': batch_rmse
    }

def plot_training_curves(metrics, output_dir='training_plots'):
    """Create training curve plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Plot 1: Batch losses (real-time)
    if metrics['batch_losses']:
        ax = axes[0, 0]
        ax.plot(metrics['batch_losses'], alpha=0.7, linewidth=0.5)
        # Add rolling average
        if len(metrics['batch_losses']) > 100:
            window = 100
            rolling_mean = np.convolve(metrics['batch_losses'], 
                                     np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(metrics['batch_losses'])), 
                   rolling_mean, 'r-', linewidth=2, label=f'{window}-batch avg')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (Per Batch)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 2: Epoch losses
    if metrics['train_losses'] and metrics['val_losses']:
        ax = axes[0, 1]
        ax.plot(metrics['epochs'], metrics['train_losses'], 'b-', label='Train', marker='o')
        ax.plot(metrics['epochs'][:len(metrics['val_losses'])], 
               metrics['val_losses'], 'r-', label='Validation', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: RMSE
    if metrics['val_rmse']:
        ax = axes[1, 0]
        ax.plot(metrics['val_rmse'], 'g-', marker='o')
        ax.set_xlabel('Validation Step')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    if metrics['learning_rates']:
        ax = axes[1, 1]
        ax.plot(metrics['learning_rates'], 'purple', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir / 'training_progress.png'}")
    
    # Also create a simple loss plot for quick viewing
    if metrics['batch_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['batch_losses'], alpha=0.5, linewidth=0.5)
        if len(metrics['batch_losses']) > 100:
            window = 100
            rolling_mean = np.convolve(metrics['batch_losses'], 
                                     np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(metrics['batch_losses'])), 
                    rolling_mean, 'r-', linewidth=2, label=f'{window}-batch avg')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        print(f"Saved loss curve to {output_dir / 'loss_curve.png'}")
    
    plt.close('all')
    
    # Print summary statistics
    if metrics['batch_losses']:
        print("\nTraining Statistics:")
        print(f"  Total batches processed: {len(metrics['batch_losses'])}")
        print(f"  Current loss: {metrics['batch_losses'][-1]:.4f}")
        print(f"  Average loss (last 100): {np.mean(metrics['batch_losses'][-100:]):.4f}")
        print(f"  Min loss: {min(metrics['batch_losses']):.4f}")
        print(f"  Max loss: {max(metrics['batch_losses']):.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot training progress')
    parser.add_argument('--log', type=str, default='training.log', help='Log file to parse')
    parser.add_argument('--diffusion', action='store_true', help='Parse diffusion training log')
    args = parser.parse_args()
    
    log_file = Path(args.log)
    
    if not log_file.exists():
        print(f"Error: {log_file} not found!")
        return
    
    print(f"Parsing {'diffusion' if args.diffusion else 'training'} log...")
    metrics = parse_training_log(log_file, is_diffusion=args.diffusion)
    
    print(f"Found {len(metrics['batch_losses'])} batch updates")
    print(f"Found {len(metrics['train_losses'])} epoch summaries")
    
    if metrics['batch_losses'] or metrics['train_losses']:
        plot_training_curves(metrics)
    else:
        print("No training data found in log file yet.")

if __name__ == '__main__':
    main()