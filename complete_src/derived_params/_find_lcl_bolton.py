from .common import *

def _find_lcl_bolton(temp_surface_k: np.ndarray, dewpoint_surface_k: np.ndarray, 
                    pressure_surface_pa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find Lifted Condensation Level using Bolton (1980) approximation
    
    Returns:
        Tuple of (LCL pressure in Pa, LCL temperature in K)
    """
    temp_c = temp_surface_k - 273.15
    dewpoint_c = dewpoint_surface_k - 273.15
    
    # Bolton (1980) LCL temperature
    lcl_temp_k = (1.0 / (1.0 / (dewpoint_c + 273.15) - 
                        np.log(temp_surface_k / dewpoint_surface_k) / 2840.0))
    
    # LCL pressure using Poisson equation
    lcl_pressure_pa = pressure_surface_pa * (lcl_temp_k / temp_surface_k) ** (1000.0 / 287.0)
    
    return lcl_pressure_pa, lcl_temp_k
