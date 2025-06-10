from .common import *

def mesocyclone_strength_parameter(updraft_helicity: np.ndarray, 
                                 vertical_velocity: np.ndarray = None,
                                 shear_magnitude: np.ndarray = None) -> np.ndarray:
    """
    Compute Mesocyclone Strength Parameter
    
    Args:
        updraft_helicity: Updraft helicity (m²/s²)
        vertical_velocity: Vertical velocity (m/s) - optional
        shear_magnitude: Wind shear magnitude (m/s) - optional
        
    Returns:
        Mesocyclone strength parameter (dimensionless)
    """
    # Base strength from updraft helicity
    base_strength = np.minimum(updraft_helicity / 100.0, 3.0)
    
    # Enhancement factors if additional data available
    strength = base_strength
    
    if vertical_velocity is not None:
        w_factor = np.minimum(np.maximum(vertical_velocity, 0) / 20.0, 1.5)
        strength *= (1.0 + 0.5 * w_factor)
    
    if shear_magnitude is not None:
        shear_factor = np.minimum(shear_magnitude / 25.0, 1.2)
        strength *= shear_factor
    
    return np.maximum(strength, 0)
