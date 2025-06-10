from .common import *

def energy_helicity_index_03km_cape(cape_03km: np.ndarray, srh_03km: np.ndarray, 
                                   mlcin: np.ndarray = None) -> np.ndarray:
    """
    Compute Energy-Helicity Index using 0-3km CAPE diagnostic
    
    EHI = (0-3km CAPE × SRH) / 160000
    
    CRITICAL FIX: HRRR's "90–0 mb CAPE" diagnostic runs 300–2500 J/kg everywhere 
    there's moist surface air, creating huge "red sea" when used directly.
    Scale by /300 rather than /50 as done for VTP low-level term.
    
    Args:
        cape_03km: 0-3km CAPE diagnostic from HRRR (J/kg) 
        srh_03km: 0-3 km Storm Relative Helicity (m²/s²)
        mlcin: Mixed-layer CIN (J/kg) for carpet suppression [optional]
        
    Returns:
        EHI values (dimensionless)
        
    Interpretation:
        EHI > 1: Notable for supercells
        EHI > 2: Significant tornado potential  
    """
    # CRITICAL FIX: Scale HRRR diagnostic by /300 to prevent red sea
    cape_03km_scaled = cape_03km / 300.0  # not /50
    
    # Standard EHI formula with scaled 0-3km CAPE
    ehi = (cape_03km_scaled * srh_03km) / 160000.0
    
    # Apply CIN gate to knock out carpets (optional but recommended)
    if mlcin is not None:
        ehi = ehi * cin_gate(mlcin)
    
    # Only positive values for tornado potential
    ehi = np.maximum(ehi, 0.0)
    
    # Mask invalid input data
    ehi = np.where((cape_03km < 0) | (np.isnan(cape_03km)) | (np.isnan(srh_03km)), 
                   np.nan, ehi)
    
    return ehi