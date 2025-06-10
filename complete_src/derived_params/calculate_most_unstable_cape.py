from .common import *

def calculate_most_unstable_cape(temp_2m: np.ndarray, dewpoint_2m: np.ndarray,
                               pressure_surface: np.ndarray) -> np.ndarray:
    """
    Calculate Most-Unstable CAPE estimate for derived parameter system
    
    Args:
        temp_2m: 2m temperature (K)
        dewpoint_2m: 2m dewpoint (K)
        pressure_surface: Surface pressure (Pa)
        
    Returns:
        MUCAPE estimate (J/kg)
    """
    # MUCAPE is typically equal to or slightly higher than SBCAPE
    # In well-mixed conditions, they are often similar
    sbcape = calculate_surface_based_cape(
        temp_2m, dewpoint_2m, pressure_surface)
    
    # Apply modest boost for most unstable (typically 1.0-1.1 of SBCAPE)
    mucape = sbcape * 1.05
    
    return mucape
