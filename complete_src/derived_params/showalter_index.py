from .common import *

def showalter_index(temp_850mb: np.ndarray, dewpoint_850mb: np.ndarray,
                   temp_500mb: np.ndarray) -> np.ndarray:
    """
    Compute Showalter Index (SI)
    
    SI = T_env(500mb) - T_parcel_850→500(500mb)
    
    Similar to LI but uses 850mb parcel instead of surface parcel.
    
    Args:
        temp_850mb: Temperature at 850mb (K)
        dewpoint_850mb: Dewpoint at 850mb (K)
        temp_500mb: Environmental temperature at 500mb (K)
        
    Returns:
        Showalter Index (°C), negative values indicate instability
        
    Interpretation:
        SI > 0: Stable
        0 to -3: Moderately unstable  
        < -6: Extremely unstable
    """
    # Convert to Celsius
    temp_850mb_c = temp_850mb - 273.15
    dewpoint_850mb_c = dewpoint_850mb - 273.15  
    temp_500mb_c = temp_500mb - 273.15
    
    # Simplified parcel temperature at 500mb from 850mb
    temp_parcel_500mb_c = temp_850mb_c - 6.5 * (3.5)  # Approximate 3.5 km lift
    
    # Add moisture adjustment
    moisture_adjustment = (temp_850mb_c - dewpoint_850mb_c) * 0.1
    temp_parcel_500mb_c += moisture_adjustment
    
    # Calculate SI
    si = temp_500mb_c - temp_parcel_500mb_c
    
    return si
