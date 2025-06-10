from .common import *

def effective_shear(wind_shear_06km: np.ndarray, mlcape: np.ndarray, mlcin: np.ndarray) -> np.ndarray:
    """
    Compute Effective Bulk Wind Difference (EBWD)
    
    Uses standard 0-6km shear as proxy for effective shear.
    In full implementation would calculate based on effective inflow layer.
    
    Args:
        wind_shear_06km: 0-6km bulk wind shear magnitude (m/s)
        mlcape: Mixed-layer CAPE (J/kg) - for future effective layer calculation
        mlcin: Mixed-layer CIN (J/kg) - for future effective layer calculation
        
    Returns:
        Effective bulk wind difference (m/s)
    """
    # For now, use 0-6km shear as effective shear proxy
    # In full implementation, would calculate actual effective inflow layer
    # based on CAPE/CIN profiles and use shear over that layer
    effective_shear = wind_shear_06km
    
    # Add small CIN gate to reduce noise (per feedback)
    # Effective shear should be near zero where CIN < -100 J/kg
    cin_weight = cin_gate(mlcin, hi=-50.0, lo=-100.0)
    effective_shear = effective_shear * cin_weight
    
    # Ensure positive values
    effective_shear = np.maximum(effective_shear, 0.0)
    
    # Mask invalid data
    effective_shear = np.where((mlcape < 0) | (np.isnan(mlcape)) | 
                              (np.isnan(wind_shear_06km)), np.nan, effective_shear)
    
    return effective_shear
