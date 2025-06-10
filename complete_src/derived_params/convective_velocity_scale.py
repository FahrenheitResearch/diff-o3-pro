from .common import *

def convective_velocity_scale(sensible_heat_flux: np.ndarray, 
                            boundary_layer_height: np.ndarray,
                            temp_surface: np.ndarray) -> np.ndarray:
    """
    Compute Convective Velocity Scale
    
    Args:
        sensible_heat_flux: Sensible heat flux (W/m²)
        boundary_layer_height: Boundary layer height (m)
        temp_surface: Surface temperature (K)
        
    Returns:
        Convective velocity scale (m/s)
    """
    # Constants
    g = 9.81
    cp = 1005
    rho = 1.225
    
    # Buoyancy flux
    buoyancy_flux = g * sensible_heat_flux / (temp_surface * rho * cp)
    
    # Convective velocity scale
    w_star = np.power(np.maximum(buoyancy_flux * boundary_layer_height, 0), 1.0/3.0)
    
    return np.clip(w_star, 0, 10)  # Cap at reasonable values
