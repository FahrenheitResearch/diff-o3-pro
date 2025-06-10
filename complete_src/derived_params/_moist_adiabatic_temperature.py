from .common import *

def _moist_adiabatic_temperature(temp_k: np.ndarray, pressure_start_pa: np.ndarray,
                               pressure_end_pa: np.ndarray) -> np.ndarray:
    """
    Calculate temperature following moist adiabatic lapse rate
    Uses pseudoadiabatic approximation
    """
    # Moist adiabatic lapse rate approximation (6.5 K/km average)
    # Convert pressure difference to approximate height difference
    height_diff = -7000.0 * np.log(pressure_end_pa / pressure_start_pa)  # m
    
    # Apply moist adiabatic lapse rate
    temp_end_k = temp_k - 0.0065 * height_diff  # K
    
    # Ensure physical temperatures
    temp_end_k = np.maximum(temp_end_k, 180.0)  # No colder than -93°C
    
    return temp_end_k
