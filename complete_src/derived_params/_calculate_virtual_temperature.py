from .common import *

def _calculate_virtual_temperature(temp_k: np.ndarray, mixing_ratio: np.ndarray) -> np.ndarray:
    """Calculate virtual temperature accounting for moisture effects"""
    # Virtual temperature correction factor (1 + 0.61 * r)
    virt_factor = 1.0 + 0.61 * mixing_ratio
    return temp_k * virt_factor
