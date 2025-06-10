from .common import *

def vorticity_generation_parameter(cape: np.ndarray, 
                                 wind_shear_01km: np.ndarray) -> np.ndarray:
    """
    Compute Vorticity Generation Parameter (VGP)
    
    VGP estimates the rate at which updrafts can generate vertical vorticity
    from horizontal vorticity through tilting and stretching.
    
    Args:
        cape: CAPE (J/kg)
        wind_shear_01km: 0-1km wind shear magnitude (m/s)
        
    Returns:
        VGP values (m/s²), >0.2 indicates tornado potential
        
    Interpretation:
        > 0.2 m/s²: Increased tornado potential
        > 0.5 m/s²: High tornado potential
    """
    # Convert shear to vorticity (approximate)
    vorticity = wind_shear_01km / 1000.0  # shear over 1km depth
    
    # VGP proportional to CAPE and low-level vorticity
    vgp = (cape / 1000.0) * vorticity
    
    # Scale to get meaningful values
    vgp = vgp * 0.1
    
    # Mask invalid data
    vgp = np.where((cape < 0) | (np.isnan(cape)) | 
                  (wind_shear_01km < 0) | (np.isnan(wind_shear_01km)), 
                  np.nan, vgp)
    
    return vgp
