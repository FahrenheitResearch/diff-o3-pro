from .common import *

def shear_vector_magnitude_ratio(shear_01km: np.ndarray, shear_06km: np.ndarray) -> np.ndarray:
    """
    Compute ratio of 0-1km to 0-6km wind shear vector magnitudes
    
    Args:
        shear_01km: 0-1km wind shear magnitude (m/s)
        shear_06km: 0-6km wind shear magnitude (m/s)
        
    Returns:
        Shear ratio (dimensionless)
    """
    # Avoid division by zero
    ratio = np.where(shear_06km > 0.1, shear_01km / shear_06km, 0.0)
    return np.clip(ratio, 0.0, 2.0)  # Cap at reasonable values
