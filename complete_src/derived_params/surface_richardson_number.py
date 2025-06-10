from .common import *

def surface_richardson_number(temp_gradient: np.ndarray, wind_shear: np.ndarray,
                            temp_surface: np.ndarray) -> np.ndarray:
    """
    Compute Surface Richardson Number
    
    Args:
        temp_gradient: Temperature gradient near surface (K/m)
        wind_shear: Wind shear near surface (s⁻¹)
        temp_surface: Surface temperature (K)
        
    Returns:
        Surface Richardson number (dimensionless)
    """
    # Richardson number = (g/T) * (dT/dz) / (du/dz)²
    g = 9.81  # gravity
    
    # Avoid division by zero
    shear_squared = np.maximum(wind_shear**2, 1e-6)
    
    ri = (g / temp_surface) * temp_gradient / shear_squared
    
    return np.clip(ri, -10, 10)  # Cap at reasonable values
