from .common import *

def _wet_bulb_approximation(temp: np.ndarray, dewpoint: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """
    Stull (2011) approximation for wet bulb temperature
    """
    # Convert pressure to kPa if in Pa
    if np.mean(pressure) > 2000:
        pressure_kpa = pressure / 1000.0
    else:
        pressure_kpa = pressure / 10.0  # mb to kPa
    
    # Relative humidity from temp and dewpoint
    es_temp = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
    es_dwpt = 6.112 * np.exp(17.67 * dewpoint / (dewpoint + 243.5))
    rh = 100 * es_dwpt / es_temp
    
    # Stull approximation
    wb = temp * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + \
         np.arctan(temp + rh) - np.arctan(rh - 1.676331) + \
         0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh) - 4.686035
    
    return wb
