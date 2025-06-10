from .common import *

def shear_vector_magnitude_ratio_from_components(u_shear_01km: np.ndarray, v_shear_01km: np.ndarray,
                                               u_shear_06km: np.ndarray, v_shear_06km: np.ndarray) -> np.ndarray:
    """
    Compute Shear Vector Magnitude Ratio from wind shear components
    """
    # Calculate magnitudes from components
    shear_01km = np.sqrt(u_shear_01km**2 + v_shear_01km**2)
    shear_06km = np.sqrt(u_shear_06km**2 + v_shear_06km**2)
    
    return shear_vector_magnitude_ratio(shear_01km, shear_06km)
