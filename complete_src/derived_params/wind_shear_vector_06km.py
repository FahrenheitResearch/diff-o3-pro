from .common import *

def wind_shear_vector_06km(u_shear: np.ndarray, v_shear: np.ndarray) -> np.ndarray:
    """
    Compute 0-6km wind shear vector magnitude
    
    Args:
        u_shear: U-component wind shear 0-6km (m/s)
        v_shear: V-component wind shear 0-6km (m/s)
        
    Returns:
        Wind shear vector magnitude (m/s)
    """
    return np.sqrt(u_shear**2 + v_shear**2)
