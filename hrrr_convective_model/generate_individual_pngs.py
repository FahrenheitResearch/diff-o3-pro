#!/usr/bin/env python3
"""
Generate individual PNG files for each variable and metric.
Saves organized in folders rather than multi-panel images.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
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


def generate_individual_forecasts():
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
    idx = np.random.randint(len(dataset)//2, len(dataset))
    current_state, _ = dataset[idx]
    current_state = current_state.unsqueeze(0).to(device)
    
    # Generate ensemble
    num_members = 20
    print(f"\nGenerating {num_members}-member ensemble...")
    ensemble_members = []
    
    for i in tqdm(range(num_members), desc="Ensemble generation"):
        x = torch.randn_like(current_state)
        
        with torch.no_grad():
            timesteps = np.linspace(999, 0, 50, dtype=int)
            
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
    current_denorm = torch.zeros_like(current_state.cpu())
    
    for i, var in enumerate(var_names):
        for m in range(num_members):
            ensemble_denorm[m, i] = torch.tensor(normalizer.decode(ensemble[m, i].numpy(), var))
        current_denorm[0, i] = torch.tensor(normalizer.decode(current_state.cpu()[0, i].numpy(), var))
    
    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path('forecasts') / f'individual_{timestamp}'
    
    dirs = {
        'current': base_dir / 'current_state',
        'mean': base_dir / 'ensemble_mean',
        'spread': base_dir / 'ensemble_spread',
        'probability': base_dir / 'probability_maps',
        'percentiles': base_dir / 'percentiles',
        'members': base_dir / 'individual_members'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving individual images to: {base_dir}")
    
    # 1. Save current state
    print("Saving current state...")
    for i, var in enumerate(var_names):
        info = var_info[var]
        save_individual_plot(
            current_denorm[0, i].numpy(),
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
    for i, var in enumerate(var_names):
        info = var_info[var]
        save_individual_plot(
            ens_mean[i].numpy(),
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
    for i, var in enumerate(var_names):
        save_individual_plot(
            ens_std[i].numpy(),
            dirs['spread'] / f'{var}_spread.png',
            f'{var} - Ensemble Spread',
            cmap='YlOrRd',
            vmin=0,
            units=var_info[var]['units'],
            cbar_label='Standard Deviation'
        )
    
    # 4. Save probability maps
    print("Saving probability maps...")
    # CAPE probabilities
    for thresh in [500, 1000, 2000, 3000]:
        prob = (ensemble_denorm[:, 5] > thresh).float().mean(dim=0)
        save_individual_plot(
            prob.numpy() * 100,
            dirs['probability'] / f'CAPE_prob_gt_{thresh}.png',
            f'Probability CAPE > {thresh} J/kg',
            cmap='Reds',
            vmin=0,
            vmax=100,
            cbar_label='Probability (%)'
        )
    
    # Reflectivity probabilities
    for thresh in [20, 35, 50]:
        prob = (ensemble_denorm[:, 0] > thresh).float().mean(dim=0)
        save_individual_plot(
            prob.numpy() * 100,
            dirs['probability'] / f'REFC_prob_gt_{thresh}.png',
            f'Probability REFC > {thresh} dBZ',
            cmap='Blues',
            vmin=0,
            vmax=100,
            cbar_label='Probability (%)'
        )
    
    # 5. Save percentiles
    print("Saving percentiles...")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        p_dir = dirs['percentiles'] / f'p{p:02d}'
        p_dir.mkdir(exist_ok=True)
        
        for i, var in enumerate(var_names):
            info = var_info[var]
            percentile_data = torch.quantile(ensemble_denorm[:, i], p/100, dim=0)
            save_individual_plot(
                percentile_data.numpy(),
                p_dir / f'{var}_p{p:02d}.png',
                f'{var} - {p}th Percentile',
                cmap=info['cmap'],
                vmin=info['vmin'],
                vmax=info['vmax'],
                units=info['units']
            )
    
    # 6. Save a few individual members
    print("Saving sample members...")
    for m in [0, 4, 9, 14, 19]:  # Save 5 members
        m_dir = dirs['members'] / f'member_{m:03d}'
        m_dir.mkdir(exist_ok=True)
        
        for i, var in enumerate(var_names):
            info = var_info[var]
            save_individual_plot(
                ensemble_denorm[m, i].numpy(),
                m_dir / f'{var}_member_{m:03d}.png',
                f'{var} - Member {m}',
                cmap=info['cmap'],
                vmin=info['vmin'],
                vmax=info['vmax'],
                units=info['units']
            )
    
    # Create summary metadata
    metadata = {
        'timestamp': timestamp,
        'ensemble_size': num_members,
        'variables': var_names,
        'directory_structure': {
            'current_state': 'Initial conditions',
            'ensemble_mean': 'Mean of all ensemble members',
            'ensemble_spread': 'Standard deviation across ensemble',
            'probability_maps': 'Exceedance probabilities',
            'percentiles': 'Statistical percentiles (10, 25, 50, 75, 90)',
            'individual_members': 'Sample individual ensemble members'
        },
        'statistics': {}
    }
    
    for i, var in enumerate(var_names):
        metadata['statistics'][var] = {
            'mean': float(ens_mean[i].mean()),
            'max': float(ens_mean[i].max()),
            'min': float(ens_mean[i].min()),
            'avg_spread': float(ens_std[i].mean()),
            'max_spread': float(ens_std[i].max())
        }
    
    import json
    with open(base_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated {len(list(base_dir.rglob('*.png')))} individual PNG files!")
    print(f"✓ Organized in: {base_dir}")
    print("\nDirectory structure:")
    for name, path in dirs.items():
        count = len(list(path.glob('*.png'))) + len(list(path.glob('*/*.png')))
        print(f"  {name}/: {count} images")


if __name__ == '__main__':
    generate_individual_forecasts()