from .common import *
from ._mixing_ratio_approximation import _mixing_ratio_approximation

def mixing_ratio_2m(dewpoint_2m: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """
    Compute 2m mixing ratio
    
    Args:
        dewpoint_2m: 2m dewpoint temperature (°C)
        pressure: Surface pressure (Pa)
        
    Returns:
        Mixing ratio (g/kg)
    """
    if METPY_AVAILABLE:
        try:
            dwpt = dewpoint_2m * units.celsius
            
            # Convert pressure from mb to Pa if needed
            if np.mean(pressure) < 2000:  # Likely in mb
                pres = pressure * 100 * units.pascal
            else:
                pres = pressure * units.pascal
            
            # Calculate saturation vapor pressure at dewpoint
            es = 6.112 * np.exp(17.67 * dewpoint_2m / (dewpoint_2m + 243.5)) * units.hectopascal
            
            mix_ratio = mixing_ratio(saturation_vapor_pressure=es, pressure=pres)
            
            return mix_ratio.to('gram/kilogram').magnitude
        except Exception as e:
            print(f"MetPy mixing ratio failed: {e}, using fallback")
    
    # Fallback calculation
    return _mixing_ratio_approximation(dewpoint_2m, pressure)
