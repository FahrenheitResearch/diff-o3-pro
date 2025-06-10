from .common import *

def wind_speed_10m(u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
    """
    Compute 10m wind speed from U and V components
    
    Args:
        u_wind: U wind component (m/s)
        v_wind: V wind component (m/s)
        
    Returns:
        Wind speed (m/s)
    """
    return np.sqrt(u_wind**2 + v_wind**2)
