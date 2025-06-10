from .common import *
from .ventilation_rate import ventilation_rate

def ventilation_rate_from_components(u_wind: np.ndarray, v_wind: np.ndarray,
                                   boundary_layer_height: np.ndarray) -> np.ndarray:
    """
    Compute Ventilation Rate from wind components
    """
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    return ventilation_rate(wind_speed, boundary_layer_height)
