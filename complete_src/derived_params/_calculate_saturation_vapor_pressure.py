from .common import *

def _calculate_saturation_vapor_pressure(temp_k: np.ndarray) -> np.ndarray:
    """Calculate saturation vapor pressure using Bolton (1980) formula"""
    temp_c = temp_k - 273.15
    return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))  # mb
