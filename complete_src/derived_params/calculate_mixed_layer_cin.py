from .common import *

def calculate_mixed_layer_cin(temp_2m: np.ndarray, dewpoint_2m: np.ndarray,
                            pressure_surface: np.ndarray) -> np.ndarray:
    """
    Calculate Mixed-Layer CIN estimate for derived parameter system
    
    Args:
        temp_2m: 2m temperature (K)
        dewpoint_2m: 2m dewpoint (K)
        pressure_surface: Surface pressure (Pa)
        
    Returns:
        MLCIN estimate (J/kg, negative values)
    """
    # Mixed layer CIN is typically weaker than surface CIN
    # due to boundary layer mixing reducing sharp inversions
    sbcin = calculate_surface_based_cin(
        temp_2m, dewpoint_2m, pressure_surface)
    
    # Apply mixed-layer reduction factor (weaker cap)
    mlcin = sbcin * 0.7
    
    return mlcin
