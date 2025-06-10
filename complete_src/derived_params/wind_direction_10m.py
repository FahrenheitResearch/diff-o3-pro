from .common import *

def wind_direction_10m(u_wind: np.ndarray, v_wind: np.ndarray) -> np.ndarray:
    """
    Compute 10m wind direction from U and V components
    
    Args:
        u_wind: U wind component (m/s)
        v_wind: V wind component (m/s)
        
    Returns:
        Wind direction (degrees from north)
    """
    # Calculate direction in mathematical convention (east = 0°)
    dir_math = np.arctan2(v_wind, u_wind) * 180.0 / np.pi
    
    # Convert to meteorological convention (north = 0°, clockwise)
    dir_met = (270.0 - dir_math) % 360.0
    
    return dir_met
