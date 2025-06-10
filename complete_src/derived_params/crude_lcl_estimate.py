from .common import *

def crude_lcl_estimate(temp_2m: np.ndarray, dewpoint_2m: np.ndarray) -> np.ndarray:
    """
    Crude LCL height estimate using Bolton (1980) approximation
    
    Args:
        temp_2m: 2m temperature (°C)
        dewpoint_2m: 2m dewpoint temperature (°C)
        
    Returns:
        LCL height (m)
    """
    # Bolton (1980) approximation: LCL ≈ (T - Td) × 125 meters
    # Assumes ~8°C/km moist adiabatic lapse rate
    temp_diff = temp_2m - dewpoint_2m
    lcl_height = temp_diff * 125.0  # meters
    
    # Ensure reasonable bounds (100m to 5000m)
    lcl_height = np.clip(lcl_height, 100, 5000)
    
    return lcl_height
