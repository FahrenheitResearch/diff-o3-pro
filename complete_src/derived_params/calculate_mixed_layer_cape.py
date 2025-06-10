from .common import *

def calculate_mixed_layer_cape(temp_2m: np.ndarray, dewpoint_2m: np.ndarray,
                             pressure_surface: np.ndarray) -> np.ndarray:
    """
    Calculate Mixed-Layer CAPE estimate for derived parameter system
    
    Args:
        temp_2m: 2m temperature (K)
        dewpoint_2m: 2m dewpoint (K)
        pressure_surface: Surface pressure (Pa)
        
    Returns:
        MLCAPE estimate (J/kg)
    """
    # Mixed layer CAPE is typically slightly lower than surface CAPE
    # due to averaging effects in the boundary layer
    sbcape = calculate_surface_based_cape(
        temp_2m, dewpoint_2m, pressure_surface)
    
    # Apply mixed-layer reduction factor (typically 0.8-0.9 of SBCAPE)
    mlcape = sbcape * 0.85
    
    return mlcape
