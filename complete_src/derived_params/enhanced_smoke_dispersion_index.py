from .common import *

def enhanced_smoke_dispersion_index(wind_shear: np.ndarray, stability: np.ndarray,
                                  boundary_layer_height: np.ndarray,
                                  wind_speed: np.ndarray) -> np.ndarray:
    """
    Compute Enhanced Smoke Dispersion Index
    
    Args:
        wind_shear: Wind shear (s⁻¹)
        stability: Atmospheric stability parameter
        boundary_layer_height: Boundary layer height (m)
        wind_speed: Wind speed (m/s)
        
    Returns:
        Enhanced smoke dispersion index (dimensionless)
    """
    # Convert wind shear from m/s to s⁻¹ if needed (typical values are 0.001-0.1 s⁻¹)
    if np.mean(wind_shear) > 1.0:  # Likely in m/s, convert to s⁻¹
        # Assume height difference of 100m for conversion
        wind_shear_s = wind_shear / 100.0
    else:
        wind_shear_s = wind_shear
    
    # Mixing factor from wind shear (scale up since shear is small)
    shear_factor = np.minimum(wind_shear_s * 100, 2.0)  # Scale and cap
    
    # Stability factor (unstable = better mixing)
    stability_factor = np.where(stability < -0.1, 2.0,  # Unstable
                              np.where(stability > 0.1, 0.5, 1.0))  # Stable vs neutral
    
    # Boundary layer factor (typical PBL heights 500-3000m)
    bl_factor = np.clip(boundary_layer_height / 1500.0, 0.1, 2.0)
    
    # Wind factor (scale for typical 2-20 m/s winds)
    wind_factor = np.clip(wind_speed / 10.0, 0.1, 2.0)
    
    # Base dispersion index
    dispersion_index = shear_factor * stability_factor * bl_factor * wind_factor
    
    # Ensure minimum dispersion even in stable conditions
    dispersion_index = np.maximum(dispersion_index, 0.1)
    
    return np.clip(dispersion_index, 0.1, 10)
