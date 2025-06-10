from .common import *

def lifted_index(temp_surface: np.ndarray, dewpoint_surface: np.ndarray, 
                temp_500mb: np.ndarray) -> np.ndarray:
    """
    Compute Lifted Index (LI)
    
    LI = T_env(500mb) - T_parcel(500mb)
    
    Formula: LI = Environmental temperature at 500mb minus temperature of 
    surface parcel lifted to 500mb. Negative values indicate instability.
    
    Args:
        temp_surface: Surface temperature (K)
        dewpoint_surface: Surface dewpoint (K) 
        temp_500mb: Environmental temperature at 500mb (K)
        
    Returns:
        Lifted Index (°C), negative values indicate instability
        
    Interpretation:
        LI > 0: Stable
        0 to -3: Marginal instability
        -4 to -6: Very unstable
        < -6: Extremely unstable
    """
    # Convert to Celsius for calculation
    temp_surface_c = temp_surface - 273.15
    dewpoint_surface_c = dewpoint_surface - 273.15
    temp_500mb_c = temp_500mb - 273.15
    
    # Simplified parcel temperature at 500mb using moist adiabatic lapse rate
    # This is an approximation - full calculation would require lifting the parcel
    temp_parcel_500mb_c = temp_surface_c - 6.5 * (5.5)  # Approximate 5.5 km lift to 500mb
    
    # Add moisture adjustment for lifted parcel
    moisture_adjustment = (temp_surface_c - dewpoint_surface_c) * 0.1
    temp_parcel_500mb_c += moisture_adjustment
    
    # Calculate LI
    li = temp_500mb_c - temp_parcel_500mb_c
    
    return li
