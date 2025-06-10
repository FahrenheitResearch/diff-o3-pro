from .common import *

def ventilation_rate(wind_speed: np.ndarray, boundary_layer_height: np.ndarray) -> np.ndarray:
    """
    Compute Ventilation Rate for smoke dispersion
    
    Args:
        wind_speed: Wind speed (m/s)
        boundary_layer_height: Boundary layer height (m)
        
    Returns:
        Ventilation rate (m²/s)
    """
    return wind_speed * boundary_layer_height
