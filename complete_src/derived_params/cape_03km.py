from .common import *

def cape_03km(mlcape: np.ndarray, cape_profile: np.ndarray = None) -> np.ndarray:
    """
    Compute 0-3 km MLCAPE - Realistic Approximation
    
    0-3 km CAPE represents low-level buoyancy and is typically much smaller
    than total CAPE. Most values range 50-300 J/kg, rarely exceeding 500 J/kg.
    
    Args:
        mlcape: Mixed-Layer CAPE (J/kg)
        cape_profile: Full CAPE profile (if available)
        
    Returns:
        0-3 km MLCAPE approximation (J/kg)
        
    Note:
        This is an approximation. HRRR may provide direct 0-3km CAPE via 
        GRIB pressure layer access (300-0mb layer) which would be preferred.
    """
    # More realistic approximation based on meteorological studies:
    # 0-3km CAPE is typically 10-30% of total CAPE, with diminishing returns for high CAPE
    
    # Apply scaling with realistic caps
    cape_fraction = np.where(
        mlcape < 1000,     # Low CAPE: higher fraction (surface dominant)
        0.25,
        np.where(
            mlcape < 3000, # Moderate CAPE: decreasing fraction
            0.20,
            0.15           # High CAPE: lower fraction (more aloft)
        )
    )
    
    cape_03km = mlcape * cape_fraction
    
    # Apply realistic caps for 0-3km CAPE based on literature
    cape_03km = np.clip(cape_03km, 0, 600)  # Very rarely exceeds 600 J/kg
    
    # Quality control - warn if high values
    max_cape_03km = np.nanmax(cape_03km) if cape_03km.size > 0 else 0
    if max_cape_03km > 400:
        print(f"🔍 0-3km CAPE max: {max_cape_03km:.0f} J/kg (high for low-level CAPE)")
    
    return cape_03km
