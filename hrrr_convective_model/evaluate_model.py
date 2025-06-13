#!/usr/bin/env python3
"""
Evaluation suite for trained DDPM model.
Generates ensemble forecasts and computes verification metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer
from utils.metrics import compute_crps, compute_spread_error, compute_reliability


class DDPMEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        self.model = UltraMinimalDDPM(
            in_channels=7,
            out_channels=7,
            base_dim=16
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load config
        self.config = checkpoint['config']
        
        # Create noise schedule
        self.noise_schedule = CosineBetaSchedule(
            timesteps=self.config['diffusion']['timesteps']
        )
        
        # Load normalizer
        self.normalizer = Normalizer(self.config['data']['stats'])
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['loss']:.4f}")
        
    def generate_ensemble(self, initial_state, num_members=50, num_steps=50):
        """Generate ensemble forecast using DDPM sampling."""
        B, C, H, W = initial_state.shape
        
        # Start from noise
        x = torch.randn(num_members, C, H, W, device=self.device)
        
        # Reverse diffusion process
        with torch.no_grad():
            for i in tqdm(range(num_steps, 0, -1), desc="Sampling"):
                t = torch.full((num_members,), i-1, device=self.device)
                
                # Predict noise
                noise_pred = self.model(x, t)
                
                # DDPM sampling step
                alpha = self.noise_schedule.alphas[i-1]
                alpha_bar = self.noise_schedule.alphas_cumprod[i-1]
                
                if i > 1:
                    alpha_bar_prev = self.noise_schedule.alphas_cumprod[i-2]
                    beta = 1 - alpha
                    
                    # Compute x_{t-1} from x_t
                    mean = (x - beta / torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
                    
                    # Add noise for next step
                    var = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta
                    noise = torch.randn_like(x) * torch.sqrt(var)
                    x = mean + noise
                else:
                    # Final step - no noise
                    x = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
        
        return x
    
    def evaluate_case(self, initial_state, target_state, case_name="test"):
        """Evaluate a single forecast case."""
        print(f"\nEvaluating case: {case_name}")
        
        # Generate ensemble
        ensemble = self.generate_ensemble(initial_state, num_members=50)
        
        # Denormalize for evaluation
        ensemble_denorm = torch.zeros_like(ensemble)
        target_denorm = torch.zeros_like(target_state)
        
        for i, var in enumerate(self.config['data']['variables']):
            for m in range(ensemble.shape[0]):
                ensemble_denorm[m, i] = self.normalizer.denormalize(
                    ensemble[m, i].cpu(), var
                )
            target_denorm[0, i] = self.normalizer.denormalize(
                target_state[0, i].cpu(), var
            )
        
        # Compute metrics
        metrics = {}
        
        # Ensemble mean
        ens_mean = ensemble_denorm.mean(dim=0, keepdim=True)
        
        # RMSE of ensemble mean
        rmse = torch.sqrt(((ens_mean - target_denorm) ** 2).mean()).item()
        metrics['rmse'] = rmse
        
        # Ensemble spread
        ens_spread = ensemble_denorm.std(dim=0).mean().item()
        metrics['spread'] = ens_spread
        
        # CRPS for each variable
        crps_scores = []
        for i, var in enumerate(self.config['data']['variables']):
            crps = compute_crps(
                ensemble_denorm[:, i].numpy(),
                target_denorm[0, i].numpy()
            )
            crps_scores.append(crps)
            metrics[f'crps_{var}'] = crps
        
        metrics['crps_mean'] = np.mean(crps_scores)
        
        # Spread-error ratio
        metrics['spread_error_ratio'] = ens_spread / rmse
        
        # Save results
        results = {
            'case_name': case_name,
            'metrics': metrics,
            'ensemble_shape': list(ensemble.shape),
            'timestamp': datetime.now().isoformat()
        }
        
        return results, ensemble_denorm
    
    def run_evaluation(self, dataset, num_cases=20, save_dir='evaluation_results'):
        """Run evaluation on multiple cases."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        for i in range(min(num_cases, len(dataset))):
            # Get random case
            idx = np.random.randint(len(dataset))
            initial_state, target_state = dataset[idx]
            
            # Move to device
            initial_state = initial_state.unsqueeze(0).to(self.device)
            target_state = target_state.unsqueeze(0).to(self.device)
            
            # Evaluate
            results, ensemble = self.evaluate_case(
                initial_state, target_state, 
                case_name=f"case_{i:03d}"
            )
            
            all_results.append(results)
            
            # Save ensemble sample for first few cases
            if i < 5:
                self.plot_ensemble(
                    ensemble, target_state,
                    save_path=save_dir / f"ensemble_case_{i:03d}.png"
                )
        
        # Aggregate metrics
        aggregated = self.aggregate_metrics(all_results)
        
        # Save results
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                'individual_results': all_results,
                'aggregated_metrics': aggregated
            }, f, indent=2)
        
        print("\nAggregated Results:")
        for key, value in aggregated.items():
            print(f"{key}: {value:.4f}")
        
        return aggregated
    
    def aggregate_metrics(self, results):
        """Aggregate metrics across all cases."""
        metrics = {}
        
        # Collect all metric values
        all_metrics = {}
        for r in results:
            for key, value in r['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Compute means
        for key, values in all_metrics.items():
            metrics[f'mean_{key}'] = np.mean(values)
            metrics[f'std_{key}'] = np.std(values)
        
        return metrics
    
    def plot_ensemble(self, ensemble, target, save_path):
        """Plot ensemble forecast visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Variables to plot
        var_indices = [0, 1, 5]  # REFC, T2M, CAPE
        var_names = ['REFC', 'T2M', 'CAPE']
        
        for i, (idx, name) in enumerate(zip(var_indices, var_names)):
            # Ensemble mean
            ens_mean = ensemble[:, idx].mean(dim=0).cpu().numpy()
            axes[i].imshow(ens_mean, cmap='viridis')
            axes[i].set_title(f'{name} - Ensemble Mean')
            axes[i].axis('off')
            
            # Ensemble spread
            ens_spread = ensemble[:, idx].std(dim=0).cpu().numpy()
            axes[i+3].imshow(ens_spread, cmap='plasma')
            axes[i+3].set_title(f'{name} - Ensemble Spread')
            axes[i+3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    # Find best checkpoint
    checkpoint_dir = Path('checkpoints/diffusion_fullres_final')
    
    # Use best_model.pt if it exists, otherwise latest
    if (checkpoint_dir / 'best_model.pt').exists():
        checkpoint_path = checkpoint_dir / 'best_model.pt'
    else:
        checkpoints = list(checkpoint_dir.glob('epoch_*.pt'))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        else:
            print("No checkpoints found!")
            return
    
    # Create evaluator
    evaluator = DDPMEvaluator(checkpoint_path)
    
    # Create test dataset
    dataset = HRRRDataset(
        zarr_path='data/zarr/training_14day/hrrr.zarr',
        variables=['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN'],
        lead_hours=1,
        stats_path='data/zarr/training_14day/stats.json'
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(dataset, num_cases=20)
    
    print("\nEvaluation complete! Check evaluation_results/ for outputs.")


if __name__ == '__main__':
    main()