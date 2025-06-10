from .common import *

def energy_helicity_index_01km(cape: np.ndarray, srh_01km: np.ndarray) -> np.ndarray:
    """
    Compute 0-1 km Energy-Helicity Index 
    
    EHI_01 = (CAPE/1600) × (0-1km SRH/50)
    
    0-1 km EHI is more directly linked to tornado potential near the surface.
    
    Args:
        cape: CAPE (J/kg) - surface-based or mixed-layer
        srh_01km: 0-1 km Storm Relative Helicity (m²/s²)
        
    Returns:
        0-1 km EHI values (dimensionless)
        
    Interpretation:
        EHI_01 > 1: Notable tornado potential
        EHI_01 > 2: High tornado potential
    """
    ehi_01 = (cape / 1600.0) * (srh_01km / 50.0)
    
    # Only positive values for tornado potential (0-1 km layer)
    ehi_01 = np.maximum(ehi_01, 0)
    
    # Mask invalid input data
    ehi_01 = np.where((cape < 0) | (np.isnan(cape)) | (np.isnan(srh_01km)), 
                     np.nan, ehi_01)
    
    return ehi_01
