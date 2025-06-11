#!/usr/bin/env python
"""
Evaluate ensemble forecasts against observations.
Generates CSV files with metrics and spread-skill plots as in Figures 2-5 of the DEF paper.
"""
import argparse
import torch
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.normalization import Normalizer
import utils.metrics as metrics


def load_config(path: str = "configs/expanded.yaml") -> dict:
    """Load configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_forecast_and_obs(
    forecast_path: Path,
    obs_zarr_path: Path,
    variables: list,
    normalizer: Normalizer
) -> tuple:
    """
    Load forecast ensemble and corresponding observations.
    
    Returns:
        forecast_ds: xarray Dataset with ensemble forecasts
        obs_data: Dictionary with observation arrays
    """
    # Load forecast
    forecast_ds = xr.open_dataset(forecast_path)
    
    # Load observations from Zarr
    obs_store = zarr.open(str(obs_zarr_path), mode='r')
    
    # Get forecast valid times
    valid_times = pd.to_datetime(forecast_ds.time.values)
    init_time = pd.to_datetime(forecast_ds.attrs['init_time'])
    
    # Load observations for each valid time
    obs_data = {}
    obs_times = pd.to_datetime(obs_store['time'][:])
    
    for var in variables:
        if var not in obs_store:
            print(f"Warning: Variable {var} not found in observations")
            continue
            
        obs_var = []
        for valid_time in valid_times:
            # Find closest observation time
            time_idx = np.argmin(np.abs(obs_times - valid_time))
            if abs(obs_times[time_idx] - valid_time) > pd.Timedelta(hours=1):
                print(f"Warning: No observation found for {var} at {valid_time}")
                obs_var.append(np.nan * np.ones_like(obs_store[var][0]))
            else:
                # Load and normalize
                data = obs_store[var][time_idx].astype(np.float32)
                data = normalizer.encode(data, var)
                obs_var.append(data)
        
        obs_data[var] = np.stack(obs_var, axis=0)  # [T, H, W]
    
    return forecast_ds, obs_data


def evaluate_variable(
    forecast_ds: xr.Dataset,
    obs: np.ndarray,
    var_name: str,
    lead_times: list,
    output_dir: Path
) -> pd.DataFrame:
    """
    Evaluate a single variable across all lead times.
    
    Args:
        forecast_ds: Forecast dataset with ensemble members
        obs: Observation array [T, H, W]
        var_name: Variable name
        lead_times: List of lead times to evaluate
        output_dir: Output directory for plots
        
    Returns:
        DataFrame with metrics for each lead time
    """
    results = []
    
    # Get ensemble data
    if var_name in forecast_ds:
        ens_data = forecast_ds[var_name].values  # [B, T, H, W]
    else:
        print(f"Warning: {var_name} not found in forecast")
        return pd.DataFrame()
    
    # Get ensemble mean and spread
    ens_mean = forecast_ds[f"{var_name}_mean"].values if f"{var_name}_mean" in forecast_ds else ens_data.mean(axis=0)
    ens_spread = forecast_ds[f"{var_name}_spread"].values if f"{var_name}_spread" in forecast_ds else ens_data.std(axis=0)
    
    # Evaluate at each lead time
    for lead_idx, lead_hour in enumerate(lead_times):
        if lead_idx >= obs.shape[0] or lead_idx >= ens_data.shape[1]:
            continue
            
        # Get data for this lead time
        obs_t = torch.tensor(obs[lead_idx])
        ens_t = torch.tensor(ens_data[:, lead_idx])  # [B, H, W]
        ens_mean_t = torch.tensor(ens_mean[lead_idx])
        ens_spread_t = torch.tensor(ens_spread[lead_idx])
        
        # Skip if too many NaNs
        if torch.isnan(obs_t).sum() > 0.5 * obs_t.numel():
            continue
        
        # Compute metrics
        # Add dimensions for compatibility
        obs_t_exp = obs_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        ens_t_exp = ens_t.unsqueeze(0)  # [1, B, H, W]
        
        # RMSE of ensemble mean
        rmse_val = metrics.rmse(obs_t, ens_mean_t).item()
        
        # MAE
        mae_val = metrics.mae(obs_t, ens_mean_t).item()
        
        # Bias
        bias_val = metrics.bias(obs_t, ens_mean_t).item()
        
        # CRPS
        crps_val = metrics.crps_ensemble(obs_t_exp, ens_t_exp).item()
        
        # Mean spread
        spread_val = ens_spread_t.mean().item()
        
        # Spread-skill ratio
        ss_ratio = spread_val / (rmse_val + 1e-8)
        
        # Store results
        results.append({
            'variable': var_name,
            'lead_hour': lead_hour,
            'rmse': rmse_val,
            'mae': mae_val,
            'bias': bias_val,
            'crps': crps_val,
            'spread': spread_val,
            'spread_skill_ratio': ss_ratio,
            'ensemble_size': ens_data.shape[0]
        })
    
    return pd.DataFrame(results)


def plot_spread_skill(
    metrics_df: pd.DataFrame,
    variables: list,
    output_dir: Path
):
    """Create spread-skill plots similar to paper figures."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl", len(variables))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Plot 1: RMSE vs Lead Time
    ax = axes[0]
    for var in variables[:4]:  # Limit to 4 key variables
        var_data = metrics_df[metrics_df['variable'] == var]
        if not var_data.empty:
            ax.plot(var_data['lead_hour'], var_data['rmse'], 
                   marker='o', label=var, linewidth=2)
    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Square Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: CRPS vs Lead Time
    ax = axes[1]
    for var in variables[:4]:
        var_data = metrics_df[metrics_df['variable'] == var]
        if not var_data.empty:
            ax.plot(var_data['lead_hour'], var_data['crps'],
                   marker='s', label=var, linewidth=2)
    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel('CRPS')
    ax.set_title('Continuous Ranked Probability Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spread vs RMSE (Spread-Skill)
    ax = axes[2]
    for var in variables[:4]:
        var_data = metrics_df[metrics_df['variable'] == var]
        if not var_data.empty:
            ax.scatter(var_data['rmse'], var_data['spread'],
                      label=var, s=50, alpha=0.7)
    # Add diagonal line (perfect spread-skill)
    max_val = max(metrics_df['rmse'].max(), metrics_df['spread'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Ensemble Spread')
    ax.set_title('Spread-Skill Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Spread-Skill Ratio vs Lead Time
    ax = axes[3]
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect')
    for var in variables[:4]:
        var_data = metrics_df[metrics_df['variable'] == var]
        if not var_data.empty:
            ax.plot(var_data['lead_hour'], var_data['spread_skill_ratio'],
                   marker='^', label=var, linewidth=2)
    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel('Spread/Skill Ratio')
    ax.set_title('Spread-Skill Ratio')
    ax.set_ylim(0.5, 1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spread_skill_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric_name: str,
    variables: list,
    output_dir: Path
):
    """Create comparison plot for a specific metric."""
    plt.figure(figsize=(10, 6))
    
    for var in variables:
        var_data = metrics_df[metrics_df['variable'] == var]
        if not var_data.empty:
            plt.plot(var_data['lead_hour'], var_data[metric_name],
                    marker='o', label=var, linewidth=2)
    
    plt.xlabel('Lead Time (hours)')
    plt.ylabel(metric_name.upper())
    plt.title(f'{metric_name.upper()} vs Lead Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'{metric_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble forecasts')
    parser.add_argument('--forecast-dir', type=Path, required=True,
                        help='Directory containing forecast NetCDF files')
    parser.add_argument('--obs-zarr', type=Path, required=True,
                        help='Path to observation Zarr archive')
    parser.add_argument('--config', type=str, default='configs/expanded.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation'),
                        help='Output directory for results')
    parser.add_argument('--lead-times', nargs='+', type=int,
                        default=[6, 12, 24, 48, 72, 96, 120, 168, 240],
                        help='Lead times to evaluate (hours)')
    parser.add_argument('--variables', nargs='+', type=str, default=None,
                        help='Variables to evaluate (default: from config)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get variables to evaluate
    if args.variables:
        eval_variables = args.variables
    else:
        eval_variables = cfg['evaluation']['eval_variables']
    
    print(f"Evaluating variables: {eval_variables}")
    print(f"Lead times: {args.lead_times}")
    
    # Initialize normalizer
    normalizer = Normalizer(Path(cfg['data']['stats']))
    
    # Find all forecast files
    forecast_files = sorted(args.forecast_dir.glob('def_forecast_*.nc'))
    if not forecast_files:
        raise ValueError(f"No forecast files found in {args.forecast_dir}")
    
    print(f"Found {len(forecast_files)} forecast files")
    
    # Evaluate each forecast
    all_results = []
    
    for forecast_file in tqdm(forecast_files, desc="Evaluating forecasts"):
        # Load forecast and observations
        try:
            forecast_ds, obs_data = load_forecast_and_obs(
                forecast_file,
                args.obs_zarr,
                eval_variables,
                normalizer
            )
        except Exception as e:
            print(f"Error loading {forecast_file}: {e}")
            continue
        
        # Evaluate each variable
        for var in eval_variables:
            if var in obs_data:
                var_results = evaluate_variable(
                    forecast_ds,
                    obs_data[var],
                    var,
                    args.lead_times,
                    args.output_dir
                )
                
                # Add metadata
                var_results['forecast_file'] = forecast_file.name
                var_results['init_time'] = forecast_ds.attrs.get('init_time', '')
                
                all_results.append(var_results)
    
    # Combine all results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Save detailed results to CSV
        results_df.to_csv(args.output_dir / 'ensemble_evaluation_results.csv', index=False)
        print(f"Saved detailed results to {args.output_dir / 'ensemble_evaluation_results.csv'}")
        
        # Compute average metrics across all forecasts
        avg_metrics = results_df.groupby(['variable', 'lead_hour']).agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'bias': ['mean', 'std'],
            'crps': ['mean', 'std'],
            'spread': ['mean', 'std'],
            'spread_skill_ratio': ['mean', 'std']
        }).round(4)
        
        avg_metrics.to_csv(args.output_dir / 'average_metrics.csv')
        print(f"Saved average metrics to {args.output_dir / 'average_metrics.csv'}")
        
        # Create plots using average metrics
        avg_metrics_flat = results_df.groupby(['variable', 'lead_hour']).mean().reset_index()
        
        # Spread-skill plots
        plot_spread_skill(avg_metrics_flat, eval_variables, args.output_dir)
        
        # Individual metric plots
        for metric in ['rmse', 'crps', 'mae', 'bias']:
            plot_metric_comparison(avg_metrics_flat, metric, eval_variables, args.output_dir)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Summary Statistics (averaged across all forecasts)")
        print("="*60)
        
        for var in eval_variables:
            var_data = avg_metrics_flat[avg_metrics_flat['variable'] == var]
            if not var_data.empty:
                print(f"\n{var}:")
                for lead in [24, 72, 120, 240]:
                    lead_data = var_data[var_data['lead_hour'] == lead]
                    if not lead_data.empty:
                        rmse = lead_data['rmse'].iloc[0]
                        crps = lead_data['crps'].iloc[0]
                        spread = lead_data['spread'].iloc[0]
                        ss_ratio = lead_data['spread_skill_ratio'].iloc[0]
                        print(f"  +{lead:3d}h: RMSE={rmse:.3f}, CRPS={crps:.3f}, "
                              f"Spread={spread:.3f}, SS-Ratio={ss_ratio:.3f}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()