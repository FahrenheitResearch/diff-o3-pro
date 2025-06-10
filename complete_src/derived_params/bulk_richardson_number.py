from .common import *

def bulk_richardson_number(cape: np.ndarray, wind_shear: np.ndarray) -> np.ndarray:
    """
    Compute Bulk Richardson Number (BRN)
    
    BRN = CAPE / (0.5 × ΔV²)
    
    BRN compares instability (CAPE) to vertical wind shear, indicating storm organization
    potential. Moderate BRN values (10-50) are associated with supercell thunderstorms.
    Low BRN (<10) indicates extreme shear that may disrupt updrafts, while high BRN 
    (>50) indicates weak shear favoring pulse or multicell storms.
    
    Args:
        cape: CAPE (J/kg)
        wind_shear: Bulk wind shear magnitude (m/s) - typically 0-6 km layer
        
    Returns:
        BRN values (dimensionless)
        
    Interpretation:
        BRN < 10: Extreme shear (storms may struggle)
        BRN 10-45: Optimal balance for supercells  
        BRN > 50: Weak shear (pulse/multicell storms)
    """
    # BRN formula: CAPE / (0.5 * ΔV²)
    # Where ΔV is the bulk wind shear vector magnitude
    
    # Avoid division by zero - set minimum shear of 1 m/s
    # (Below 1 m/s shear is essentially no shear environment)
    shear_term = 0.5 * np.maximum(wind_shear**2, 1.0**2)
    
    brn = cape / shear_term
    
    # Handle edge cases
    brn = np.where(cape <= 0, 0, brn)  # No CAPE = no BRN
    
    # For display purposes, cap at 999 (beyond 100+ all means "weak shear")
    # This prevents near-zero shear from creating unrealistic BRN values
    brn = np.minimum(brn, 999.0)
    
    # Mask invalid input data
    brn = np.where((np.isnan(cape)) | (np.isnan(wind_shear)) | (wind_shear < 0), 
                  np.nan, brn)
    
    return brn
