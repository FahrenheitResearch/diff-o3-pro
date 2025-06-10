from .common import *
from ._wet_bulb_approximation import _wet_bulb_approximation

def wet_bulb_temperature_metpy(temp_2m: np.ndarray, dewpoint_2m: np.ndarray, 
                              pressure: np.ndarray) -> np.ndarray:
    """
    Compute wet bulb temperature using MetPy or fallback method
    
    Args:
        temp_2m: 2m temperature (°C)
        dewpoint_2m: 2m dewpoint (°C) 
        pressure: Surface pressure (Pa)
        
    Returns:
        Wet bulb temperature (°C)
    """
    if METPY_AVAILABLE:
        try:
            # Convert to MetPy units - ensure pressure is in Pa
            temp = temp_2m * units.celsius
            dwpt = dewpoint_2m * units.celsius  
            
            # Convert pressure from mb to Pa if needed
            if np.mean(pressure) < 2000:  # Likely in mb
                pres = pressure * 100 * units.pascal
            else:
                pres = pressure * units.pascal
            
            # Calculate wet bulb temperature
            wb_temp = wet_bulb_temperature(pres, temp, dwpt)
            
            return wb_temp.to('celsius').magnitude
        except Exception as e:
            print(f"MetPy wet bulb failed: {e}, using fallback")
    
    # Fallback: Stull approximation for wet bulb temperature
    return _wet_bulb_approximation(temp_2m, dewpoint_2m, pressure)
