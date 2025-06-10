from .common import *

def monin_obukhov_length(friction_velocity: np.ndarray, temp_surface: np.ndarray,
                       sensible_heat_flux: np.ndarray) -> np.ndarray:
    """
    Compute Monin-Obukhov Length (simplified)
    
    Args:
        friction_velocity: Friction velocity (m/s)
        temp_surface: Surface temperature (K)
        sensible_heat_flux: Sensible heat flux (W/m²)
        
    Returns:
        Monin-Obukhov length (m)
    """
    # Constants
    g = 9.81  # gravity
    cp = 1005  # specific heat of air
    rho = 1.225  # air density (approximate)
    k = 0.4  # von Karman constant
    
    # Calculate
    theta_star = -sensible_heat_flux / (rho * cp * friction_velocity)
    
    # Avoid division by zero
    theta_star = np.where(np.abs(theta_star) < 1e-6, 
                         np.sign(theta_star) * 1e-6, theta_star)
    
    L = temp_surface * friction_velocity**2 / (k * g * theta_star)
    
    return np.clip(L, -10000, 10000)  # Cap at reasonable values
