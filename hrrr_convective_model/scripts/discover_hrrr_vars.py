#!/usr/bin/env python
"""Discover all variables in HRRR GRIB2 files using wgrib2."""
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

def discover_with_wgrib2(grib_file):
    """Use wgrib2 to list all variables."""
    try:
        result = subprocess.run(
            ['wgrib2', str(grib_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error running wgrib2: {result.stderr}")
            return []
        
        # Parse wgrib2 output
        variables = defaultdict(list)
        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 4:
                    var_info = parts[3]  # Variable description
                    level_info = parts[4] if len(parts) > 4 else ""
                    variables[var_info.split()[0]].append(level_info)
        
        return variables
    except FileNotFoundError:
        print("wgrib2 not found. Please install it or use conda environment with it.")
        return {}

def discover_with_cfgrib(grib_file):
    """Use cfgrib to discover variables by trying different filters."""
    import xarray as xr
    
    print("\n" + "="*60)
    print(f"Discovering variables in {grib_file.name}")
    print("="*60)
    
    # Common HRRR variables to try
    test_patterns = [
        # Temperature variables
        ("t", "Temperature"),
        ("2t", "2m Temperature"),
        ("t2m", "2m Temperature"),
        ("tmax", "Max Temperature"),
        ("tmin", "Min Temperature"),
        
        # Moisture variables
        ("2d", "2m Dewpoint"),
        ("d2m", "2m Dewpoint"),
        ("2r", "2m Relative Humidity"),
        ("r2", "2m Relative Humidity"),
        ("q", "Specific Humidity"),
        ("pwat", "Precipitable Water"),
        
        # Wind variables
        ("10u", "10m U-wind"),
        ("10v", "10m V-wind"),
        ("u10", "10m U-wind"),
        ("v10", "10m V-wind"),
        ("gust", "Wind Gust"),
        ("wgust", "Wind Gust"),
        
        # Precipitation/Reflectivity
        ("refc", "Composite Reflectivity"),
        ("refd", "Reflectivity"),
        ("tp", "Total Precipitation"),
        ("cp", "Convective Precipitation"),
        ("prate", "Precipitation Rate"),
        ("apcp", "Accumulated Precipitation"),
        
        # Instability parameters
        ("cape", "CAPE"),
        ("cin", "CIN"),
        ("lftx", "Lifted Index"),
        ("4lftx", "Best 4-layer Lifted Index"),
        ("hlcy", "Storm Relative Helicity"),
        
        # Pressure/Height
        ("sp", "Surface Pressure"),
        ("prmsl", "Mean Sea Level Pressure"),
        ("gh", "Geopotential Height"),
        ("hgt", "Height"),
        
        # Other
        ("vis", "Visibility"),
        ("tcdc", "Total Cloud Cover"),
        ("dswrf", "Downward SW Radiation"),
        ("dlwrf", "Downward LW Radiation"),
        ("hpbl", "PBL Height"),
        ("fricv", "Friction Velocity"),
    ]
    
    found_vars = {}
    
    for short_name, description in test_patterns:
        try:
            ds = xr.open_dataset(
                grib_file,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": short_name}}
            )
            
            if ds.data_vars:
                var_names = list(ds.data_vars)
                dims = list(ds.dims)
                coords = list(ds.coords)
                
                found_vars[short_name] = {
                    'description': description,
                    'actual_names': var_names,
                    'dims': dims,
                    'coords': coords,
                    'shape': ds[var_names[0]].shape if var_names else None
                }
                
                print(f"\n✓ {short_name} ({description}):")
                print(f"  Variables: {var_names}")
                print(f"  Shape: {ds[var_names[0]].shape if var_names else 'N/A'}")
                
        except Exception as e:
            # Silently skip if not found
            pass
    
    # Try to find level-specific variables
    print("\n\nChecking level-specific variables...")
    
    level_types = [
        "surface",
        "heightAboveGround", 
        "isobaricInhPa",
        "atmosphereLayer",
        "entireAtmosphere",
    ]
    
    for level_type in level_types:
        try:
            ds = xr.open_dataset(
                grib_file,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": level_type}}
            )
            
            if ds.data_vars:
                print(f"\n✓ Level type '{level_type}':")
                print(f"  Variables: {list(ds.data_vars)[:5]}...")
                print(f"  Total: {len(ds.data_vars)} variables")
                
        except Exception as e:
            pass
    
    return found_vars

def main():
    if len(sys.argv) > 1:
        grib_file = Path(sys.argv[1])
    else:
        # Default to first file in data/raw
        grib_files = list(Path("data/raw").glob("*.grib2"))
        if not grib_files:
            print("No GRIB2 files found in data/raw/")
            return
        grib_file = grib_files[0]
    
    # Try wgrib2 first
    wgrib_vars = discover_with_wgrib2(grib_file)
    if wgrib_vars:
        print("\nVariables found with wgrib2:")
        for var, levels in sorted(wgrib_vars.items()):
            print(f"  {var}: {len(levels)} levels")
    
    # Then try cfgrib
    cfgrib_vars = discover_with_cfgrib(grib_file)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Variables accessible via cfgrib:")
    print("="*60)
    for short_name, info in sorted(cfgrib_vars.items()):
        print(f"\n{short_name}: {info['description']}")
        print(f"  Actual variable names: {info['actual_names']}")
        print(f"  Shape: {info['shape']}")

if __name__ == "__main__":
    main()