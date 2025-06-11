#!/usr/bin/env python3
"""Create forecast visualization maps."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import zarr
import argparse

# HRRR projection
HRRR_PROJ = ccrs.LambertConformal(
    central_longitude=-97.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5)
)

def get_reflectivity_cmap():
    """NOAA-style reflectivity colormap."""
    colors = [
        '#FFFFFF', '#00FFFF', '#0080FF', '#0000FF', '#00FF00',
        '#00C000', '#008000', '#FFFF00', '#FFC000', '#FF8000',
        '#FF0000', '#C00000', '#800000', '#FF00FF', '#8000FF'
    ]
    levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))
    return cmap, norm, levels

def get_temperature_cmap():
    """Temperature colormap (F)."""
    cmap = plt.cm.RdBu_r
    levels = np.arange(-20, 120, 10)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return cmap, norm, levels

def plot_forecast_map(forecast_file, zarr_path, output_dir):
    """Create forecast visualization."""
    # Load data
    ds = xr.open_dataset(forecast_file)
    store = zarr.open(str(zarr_path), 'r')
    lat = store['latitude'][:]
    lon = store['longitude'][:]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get timing info
    init_str = ds.attrs.get('initialization_time', '2025-01-01 00:00:00')
    
    # Plot each lead time
    for t_idx in range(len(ds.time)):
        lead_hr = t_idx  # Assuming hourly output
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Title
        fig.suptitle(f'DEF Ensemble Forecast - Init: {init_str} | Lead: {lead_hr}h',
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Reflectivity - Ensemble Mean
        ax1 = fig.add_subplot(2, 3, 1, projection=HRRR_PROJ)
        ax1.set_extent([-125, -66, 22, 50], ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        if 'REFC' in ds:
            refc_mean = ds.REFC.isel(time=t_idx).mean(dim='member').values
            cmap, norm, levels = get_reflectivity_cmap()
            cf = ax1.contourf(lon, lat, refc_mean, levels=levels,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax1, orientation='horizontal', pad=0.05, 
                        label='Reflectivity (dBZ)', shrink=0.8)
        ax1.set_title('Composite Reflectivity - Ensemble Mean', fontsize=12)
        
        # 2. Temperature - Ensemble Mean
        ax2 = fig.add_subplot(2, 3, 2, projection=HRRR_PROJ)
        ax2.set_extent([-125, -66, 22, 50], ccrs.PlateCarree())
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax2.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        if 'T2M' in ds:
            t2m_mean = ds.T2M.isel(time=t_idx).mean(dim='member').values
            # Convert K to F
            t2m_mean = (t2m_mean - 273.15) * 9/5 + 32
            cmap, norm, levels = get_temperature_cmap()
            cf = ax2.contourf(lon, lat, t2m_mean, levels=levels,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax2, orientation='horizontal', pad=0.05,
                        label='Temperature (°F)', shrink=0.8)
        ax2.set_title('2m Temperature - Ensemble Mean', fontsize=12)
        
        # 3. CAPE - Ensemble Mean
        ax3 = fig.add_subplot(2, 3, 3, projection=HRRR_PROJ)
        ax3.set_extent([-125, -66, 22, 50], ccrs.PlateCarree())
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax3.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        if 'CAPE' in ds:
            cape_mean = ds.CAPE.isel(time=t_idx).mean(dim='member').values
            levels = [0, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
            cmap = plt.cm.hot_r
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            cf = ax3.contourf(lon, lat, cape_mean, levels=levels,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax3, orientation='horizontal', pad=0.05,
                        label='CAPE (J/kg)', shrink=0.8)
        ax3.set_title('CAPE - Ensemble Mean', fontsize=12)
        
        # 4. Reflectivity - Ensemble Spread
        ax4 = fig.add_subplot(2, 3, 4, projection=HRRR_PROJ)
        ax4.set_extent([-125, -66, 22, 50], ccrs.PlateCarree())
        ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax4.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax4.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        if 'REFC' in ds:
            refc_spread = ds.REFC.isel(time=t_idx).std(dim='member').values
            levels = np.arange(0, 25, 2.5)
            cf = ax4.contourf(lon, lat, refc_spread, levels=levels,
                             cmap='viridis', transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax4, orientation='horizontal', pad=0.05,
                        label='Std Dev (dBZ)', shrink=0.8)
        ax4.set_title('Reflectivity - Ensemble Spread', fontsize=12)
        
        # 5. Temperature - Member 1 (Deterministic)
        ax5 = fig.add_subplot(2, 3, 5, projection=HRRR_PROJ)
        ax5.set_extent([-125, -66, 22, 50], ccrs.PlateCarree())
        ax5.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax5.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax5.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        if 'T2M' in ds:
            t2m_det = ds.T2M.isel(lead_time=t_idx, member=0).values
            # Convert K to F
            t2m_det = (t2m_det - 273.15) * 9/5 + 32
            cmap, norm, levels = get_temperature_cmap()
            cf = ax5.contourf(lon, lat, t2m_det, levels=levels,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax5, orientation='horizontal', pad=0.05,
                        label='Temperature (°F)', shrink=0.8)
        ax5.set_title('2m Temperature - Deterministic', fontsize=12)
        
        # 6. Wind Speed - Ensemble Mean
        ax6 = fig.add_subplot(2, 3, 6, projection=HRRR_PROJ)
        ax6.set_extent([-125, -66, 22, 50], ccrs.PlateCarree())
        ax6.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax6.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax6.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        if 'U10' in ds and 'V10' in ds:
            u10_mean = ds.U10.isel(time=t_idx).mean(dim='member').values
            v10_mean = ds.V10.isel(time=t_idx).mean(dim='member').values
            wspd = np.sqrt(u10_mean**2 + v10_mean**2) * 2.237  # m/s to mph
            
            levels = np.arange(0, 60, 5)
            cmap = plt.cm.viridis
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            cf = ax6.contourf(lon, lat, wspd, levels=levels,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax6, orientation='horizontal', pad=0.05,
                        label='Wind Speed (mph)', shrink=0.8)
            
            # Add wind barbs (subsample for clarity)
            skip = 50
            ax6.barbs(lon[::skip, ::skip], lat[::skip, ::skip],
                     u10_mean[::skip, ::skip]*2.237, v10_mean[::skip, ::skip]*2.237,
                     transform=ccrs.PlateCarree(), length=4, alpha=0.6)
        ax6.set_title('10m Wind Speed - Ensemble Mean', fontsize=12)
        
        # Save
        plt.tight_layout()
        output_file = output_dir / f'forecast_f{lead_hr:03d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
    
    print(f"\nAll maps saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast', type=str, 
                        default='forecasts/demo/def_forecast_20250101_00Z.nc')
    parser.add_argument('--zarr', type=str, 
                        default='data/zarr/test/hrrr.zarr')
    parser.add_argument('--output', type=str, 
                        default='forecasts/demo/maps')
    
    args = parser.parse_args()
    plot_forecast_map(args.forecast, args.zarr, args.output)