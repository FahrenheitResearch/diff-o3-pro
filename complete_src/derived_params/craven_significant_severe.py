from .common import *

def craven_significant_severe(mlcape: np.ndarray, bulk_shear_06km: np.ndarray) -> np.ndarray:
    """
    Compute Craven Significant Severe Parameter
    
    Craven SigSvr = MLCAPE × 0-6km Bulk Shear
    
    Simple product of mixed-layer CAPE and deep-layer shear.
    
    Args:
        mlcape: Mixed-Layer CAPE (J/kg)
        bulk_shear_06km: 0-6km bulk shear magnitude (m/s)
        
    Returns:
        Craven parameter (m³/s³), >20,000 indicates significant severe potential
        
    Interpretation:
        > 20,000: Significant severe weather potential
        > 50,000: Very high severe potential
    """
    craven = mlcape * bulk_shear_06km
    
    # Mask invalid data
    craven = np.where((mlcape < 0) | (np.isnan(mlcape)) | 
                     (bulk_shear_06km < 0) | (np.isnan(bulk_shear_06km)), 
                     np.nan, craven)
    
    return craven
