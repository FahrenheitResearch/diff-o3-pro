from .common import *

def _mixing_ratio_approximation(dewpoint: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """
    Calculate mixing ratio using basic formula
    """
    # Convert pressure to mb if in Pa
    if np.mean(pressure) > 2000:
        pressure_mb = pressure / 100.0
    else:
        pressure_mb = pressure
    
    # Saturation vapor pressure at dewpoint (Tetens formula)
    es = 6.112 * np.exp(17.67 * dewpoint / (dewpoint + 243.5))  # mb
    
    # Mixing ratio formula: w = 621.97 * es / (p - es) 
    mixing_ratio = 621.97 * es / (pressure_mb - es)  # g/kg
    
    return np.maximum(mixing_ratio, 0)  # Ensure non-negative
