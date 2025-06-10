from .common import *

def crosswind_component(u_wind: np.ndarray, v_wind: np.ndarray, 
                       reference_direction: float = 0.0) -> np.ndarray:
    """
    Compute crosswind component relative to reference direction
    
    Args:
        u_wind: U wind component (m/s)
        v_wind: V wind component (m/s)
        reference_direction: Reference direction in degrees (default: north)
        
    Returns:
        Crosswind component (m/s, positive = right of reference)
    """
    # Convert reference direction to radians
    ref_rad = np.radians(reference_direction)
    
    # Calculate crosswind component
    crosswind = -u_wind * np.sin(ref_rad) + v_wind * np.cos(ref_rad)
    
    return crosswind
