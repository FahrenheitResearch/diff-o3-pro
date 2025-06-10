from .common import *

def wind_shear_magnitude(u_component: np.ndarray, 
                       v_component: np.ndarray) -> np.ndarray:
    """
    Compute wind shear magnitude from U and V components
    
    Args:
        u_component: U component of bulk shear (m/s)
        v_component: V component of bulk shear (m/s)
        
    Returns:
        Shear magnitude (m/s)
    """
    return np.sqrt(u_component**2 + v_component**2)
