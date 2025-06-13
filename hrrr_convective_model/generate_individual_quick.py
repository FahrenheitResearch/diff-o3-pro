#!/usr/bin/env python3
"""
Quick version - Generate individual PNG files with smaller ensemble.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.diffusion.ddpm_ultra_minimal import UltraMinimalDDPM, CosineBetaSchedule
from hrrr_dataset.hrrr_data import HRRRDataset
from utils.normalization import Normalizer


def save_individual_plot(data, filename, title, cmap='viridis', vmin=None, vmax=None, 
                        units='', cbar_label=None):
    """Save a single variable as an individual PNG."""
    plt.figure(figsize=(10, 8))
    
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    plt.title(title, fontsize=14, pad=10)
    plt.axis('off')
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=12)
    elif units:
        cbar.set_label(units, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_quick_individuals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint_path = Path('checkpoints/diffusion_fullres_final/best_model.pt')
    print(f"Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = UltraMinimalDDPM(in_channels=7, out_channels=7, base_dim=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    config = checkpoint['config']
    
    # Create noise schedule
    noise_schedule = CosineBetaSchedule(timesteps=1000, s=0.008)
    for attr in ['alphas_cumprod', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'betas', 'alphas']:
        setattr(noise_schedule, attr, getattr(noise_schedule, attr).to(device))
    
    # Load data
    normalizer = Normalizer(config['data']['stats'])
    dataset = HRRRDataset(
        zarr_path=config['data']['zarr'],
        variables=config['data']['variables'],
        lead_hours=1,
        stats_path=config['data']['stats']
    )
    
    # Get sample
    idx = 400  # Fixed index for reproducibility
    current_state, _ = dataset[idx]
    current_state = current_state.unsqueeze(0).to(device)
    
    # Generate small ensemble for speed
    num_members = 5
    print(f"\nGenerating {num_members}-member ensemble (quick version)...")
    ensemble_members = []
    
    # Use lower resolution for quick demo
    H, W = 265, 450  # 1/4 resolution
    current_small = torch.nn.functional.interpolate(current_state, size=(H, W), mode='bilinear')
    
    for i in tqdm(range(num_members), desc="Quick ensemble"):
        x = torch.randn(1, 7, H, W, device=device)
        
        with torch.no_grad():
            timesteps = np.linspace(999, 0, 30, dtype=int)  # Fewer steps
            
            for t_idx in timesteps:
                t = torch.tensor([t_idx], device=device)
                noise_pred = model(x, t)
                
                alpha = noise_schedule.alphas[t_idx]
                alpha_bar = noise_schedule.alphas_cumprod[t_idx]
                beta = noise_schedule.betas[t_idx]
                
                if t_idx > 0:
                    alpha_bar_prev = noise_schedule.alphas_cumprod[t_idx - 1]
                    x0_pred = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x0_pred = torch.clamp(x0_pred, -3, 3)
                    
                    posterior_mean = (alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)) * x0_pred + \
                                   (alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)) * x
                    posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    
                    noise = torch.randn_like(x) if t_idx > 1 else 0
                    x = posterior_mean + posterior_var.sqrt() * noise
                else:
                    x = (x - noise_pred * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x = torch.clamp(x, -3, 3)
        
        ensemble_members.append(x.cpu())
    
    ensemble = torch.cat(ensemble_members, dim=0)
    
    # Denormalize
    print("Processing ensemble...")
    var_names = config['data']['variables']
    var_info = {
        'REFC': {'units': 'dBZ', 'cmap': 'turbo', 'vmin': -10, 'vmax': 60},
        'T2M': {'units': 'K', 'cmap': 'RdBu_r', 'vmin': 270, 'vmax': 310},
        'D2M': {'units': 'K', 'cmap': 'RdBu_r', 'vmin': 260, 'vmax': 300},
        'U10': {'units': 'm/s', 'cmap': 'RdBu_r', 'vmin': -20, 'vmax': 20},
        'V10': {'units': 'm/s', 'cmap': 'RdBu_r', 'vmin': -20, 'vmax': 20},
        'CAPE': {'units': 'J/kg', 'cmap': 'hot_r', 'vmin': 0, 'vmax': 4000},
        'CIN': {'units': 'J/kg', 'cmap': 'Blues_r', 'vmin': -300, 'vmax': 0}
    }
    
    ensemble_denorm = torch.zeros_like(ensemble)
    current_denorm = torch.zeros_like(current_small.cpu())
    
    for i, var in enumerate(var_names):
        for m in range(num_members):
            ensemble_denorm[m, i] = torch.tensor(normalizer.decode(ensemble[m, i].numpy(), var))
        current_denorm[0, i] = torch.tensor(normalizer.decode(current_small.cpu()[0, i].numpy(), var))
    
    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path('forecasts') / f'individual_{timestamp}'
    
    # Create directories
    dirs = {
        'current': base_dir / '01_current_state',
        'mean': base_dir / '02_ensemble_mean',
        'spread': base_dir / '03_ensemble_spread',
        'probability': base_dir / '04_probability_maps',
        'members': base_dir / '05_sample_members'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving individual images to: {base_dir}")
    
    # Variables to focus on
    focus_vars = ['REFC', 'T2M', 'CAPE']  # Subset for quick demo
    focus_indices = [0, 1, 5]
    
    # 1. Save current state
    print("Saving current state...")
    for var, idx in zip(focus_vars, focus_indices):
        info = var_info[var]
        save_individual_plot(
            current_denorm[0, idx].numpy(),
            dirs['current'] / f'{var}_current.png',
            f'{var} - Current State',
            cmap=info['cmap'],
            vmin=info['vmin'],
            vmax=info['vmax'],
            units=info['units']
        )
    
    # 2. Save ensemble mean
    print("Saving ensemble mean...")
    ens_mean = ensemble_denorm.mean(dim=0)
    for var, idx in zip(focus_vars, focus_indices):
        info = var_info[var]
        save_individual_plot(
            ens_mean[idx].numpy(),
            dirs['mean'] / f'{var}_mean.png',
            f'{var} - Ensemble Mean (+1h)',
            cmap=info['cmap'],
            vmin=info['vmin'],
            vmax=info['vmax'],
            units=info['units']
        )
    
    # 3. Save ensemble spread
    print("Saving ensemble spread...")
    ens_std = ensemble_denorm.std(dim=0)
    for var, idx in zip(focus_vars, focus_indices):
        save_individual_plot(
            ens_std[idx].numpy(),
            dirs['spread'] / f'{var}_spread.png',
            f'{var} - Ensemble Spread',
            cmap='YlOrRd',
            vmin=0,
            units=var_info[var]['units'],
            cbar_label='Standard Deviation'
        )
    
    # 4. Save key probability maps
    print("Saving probability maps...")
    # CAPE > 1000
    prob = (ensemble_denorm[:, 5] > 1000).float().mean(dim=0)
    save_individual_plot(
        prob.numpy() * 100,
        dirs['probability'] / f'CAPE_prob_gt_1000.png',
        f'Probability CAPE > 1000 J/kg',
        cmap='Reds',
        vmin=0,
        vmax=100,
        cbar_label='Probability (%)'
    )
    
    # REFC > 35
    prob = (ensemble_denorm[:, 0] > 35).float().mean(dim=0)
    save_individual_plot(
        prob.numpy() * 100,
        dirs['probability'] / f'REFC_prob_gt_35.png',
        f'Probability REFC > 35 dBZ',
        cmap='Blues',
        vmin=0,
        vmax=100,
        cbar_label='Probability (%)'
    )
    
    # 5. Save one member
    print("Saving sample member...")
    for var, idx in zip(focus_vars, focus_indices):
        info = var_info[var]
        save_individual_plot(
            ensemble_denorm[0, idx].numpy(),
            dirs['members'] / f'{var}_member_000.png',
            f'{var} - Member 0',
            cmap=info['cmap'],
            vmin=info['vmin'],
            vmax=info['vmax'],
            units=info['units']
        )
    
    # Create README
    readme_content = f"""# Individual Forecast Images

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Ensemble Size: {num_members} members
Resolution: {H}x{W} (quarter resolution for demo)

## Directory Structure

- **01_current_state/**: Initial atmospheric conditions
- **02_ensemble_mean/**: Average of all ensemble members
- **03_ensemble_spread/**: Standard deviation (uncertainty)
- **04_probability_maps/**: Probability of exceeding thresholds
- **05_sample_members/**: Individual ensemble member examples

## Variables

- **REFC**: Composite Reflectivity (dBZ) - indicates precipitation
- **T2M**: 2-meter Temperature (K)
- **CAPE**: Convective Available Potential Energy (J/kg) - storm potential

## Usage

Each PNG file can be viewed individually or imported into other applications.
The probability maps show the likelihood of exceeding meteorologically significant thresholds.
"""
    
    with open(base_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ Generated individual PNG files!")
    print(f"✓ Directory: {base_dir}")
    print("\nContents:")
    for name, path in dirs.items():
        count = len(list(path.glob('*.png')))
        print(f"  {path.name}/: {count} images")
    
    print(f"\n✓ Total: {len(list(base_dir.rglob('*.png')))} PNG files")


if __name__ == '__main__':
    generate_quick_individuals()