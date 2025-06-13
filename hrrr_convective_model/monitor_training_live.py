#!/usr/bin/env python3
"""
Live monitoring of DDPM training with auto-updating plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import re
from pathlib import Path
from collections import deque
import matplotlib.animation as animation
from datetime import datetime


class TrainingMonitor:
    def __init__(self, log_path='logs/diffusion_fullres_final.log'):
        self.log_path = Path(log_path)
        self.losses = deque(maxlen=1000)  # Keep last 1000 losses
        self.epochs = []
        self.epoch_losses = []
        self.steps = 0
        self.last_position = 0
        
        # Set up the plot
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('DDPM Training Monitor - Live', fontsize=16, color='cyan')
        
    def parse_new_lines(self):
        """Parse new lines from log file."""
        if not self.log_path.exists():
            return
            
        with open(self.log_path, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()
            
        for line in new_lines:
            # Match batch losses
            match = re.search(r'loss=(\d+\.\d+), avg=(\d+\.\d+)', line)
            if match:
                loss = float(match.group(1))
                avg_loss = float(match.group(2))
                self.losses.append(loss)
                self.steps += 1
                
            # Match epoch completion
            epoch_match = re.search(r'Epoch (\d+): Average Loss = (\d+\.\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                avg_loss = float(epoch_match.group(2))
                self.epochs.append(epoch)
                self.epoch_losses.append(avg_loss)
    
    def update_plots(self, frame):
        """Update the plots with new data."""
        self.parse_new_lines()
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot batch losses
        if self.losses:
            self.ax1.plot(list(self.losses), 'g-', alpha=0.6, linewidth=1)
            
            # Add smoothed line
            if len(self.losses) > 20:
                window = min(20, len(self.losses))
                smoothed = np.convolve(list(self.losses), 
                                     np.ones(window)/window, mode='valid')
                self.ax1.plot(range(len(smoothed)), smoothed, 'y-', 
                            linewidth=2, label=f'Smoothed (w={window})')
            
            self.ax1.set_xlabel('Recent Batches')
            self.ax1.set_ylabel('Loss')
            self.ax1.set_title(f'Batch Losses (Last {len(self.losses)} batches)')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            
            # Add current loss annotation
            current_loss = self.losses[-1]
            self.ax1.text(0.98, 0.95, f'Current: {current_loss:.4f}', 
                         transform=self.ax1.transAxes,
                         ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                         color='white', fontsize=12)
        
        # Plot epoch losses
        if self.epoch_losses:
            self.ax2.plot(self.epochs, self.epoch_losses, 'b-o', 
                         linewidth=2, markersize=8, label='Epoch Average')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Average Loss')
            self.ax2.set_title('Epoch-Level Progress')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            
            # Add improvement percentage
            if len(self.epoch_losses) > 1:
                initial = self.epoch_losses[0]
                current = self.epoch_losses[-1]
                improvement = (initial - current) / initial * 100
                self.ax2.text(0.02, 0.95, f'Improvement: {improvement:.1f}%', 
                            transform=self.ax2.transAxes,
                            va='top',
                            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                            color='white', fontsize=12)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.fig.text(0.99, 0.01, f'Updated: {timestamp}', 
                     ha='right', va='bottom', fontsize=10, color='gray')
        
        # Add training info
        info_text = (f"Model: 81K params | Resolution: 1059x1799 | "
                    f"Data: Real HRRR (14 days)")
        self.fig.text(0.01, 0.01, info_text, 
                     ha='left', va='bottom', fontsize=10, color='gray')
        
        plt.tight_layout()
    
    def start(self):
        """Start the live monitoring."""
        # Create animation that updates every 5 seconds
        ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                    interval=5000, cache_frame_data=False)
        
        print("Live monitoring started. Press Ctrl+C to stop.")
        print(f"Monitoring: {self.log_path}")
        print("Plots update every 5 seconds...")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == '__main__':
    monitor = TrainingMonitor()
    monitor.start()