from .common import *

def significant_tornado_parameter_cin(mlcape: np.ndarray, mlcin: np.ndarray,
                                     effective_srh: np.ndarray, effective_shear: np.ndarray,
                                     lcl_height: np.ndarray) -> np.ndarray:
    """
    Compute STP CIN Version (Thompson et al. 2004)
    
    STP CIN = (MLCAPE/1500) × (ESRH/150) × (EBWD/12) × ((2000-MLLCL)/1000) × ((MLCIN+200)/150)
    
    This is the effective-layer version with CIN term.
    Uses mixed-layer CAPE, effective SRH/shear, and includes CIN constraint.
    
    Args:
        mlcape: Mixed-Layer CAPE (J/kg) - 100mb mean parcel
        mlcin: Mixed-Layer CIN (J/kg, negative values)
        effective_srh: Effective Storm Relative Helicity (m²/s²)
        effective_shear: Effective Bulk Wind Difference (m/s)
        lcl_height: Mixed-Layer LCL height (m AGL)
        
    Returns:
        STP CIN values (dimensionless, always ≥ 0)
    """
    # 1. CAPE term: MLCAPE/1500
    cape_term = np.maximum(mlcape / 1500.0, 0)
    
    # 2. Effective SRH term: ESRH/150 (only positive values)
    srh_term = np.maximum(effective_srh / 150.0, 0)
    
    # 3. Effective Shear term: EBWD/12 m/s with SPC constraints
    # Minimum value raised to 12 m/s, capped at 1.5 when > 30 m/s
    shear_term = np.where(effective_shear < 12.0, 0.0,
                         np.where(effective_shear > 30.0, 1.5,
                                 effective_shear / 12.0))
    
    # 4. LCL term: (2000-MLLCL)/1000 with clipping
    # LCL < 1000m → 1.0, LCL > 2000m → 0.0
    lcl_term = np.where(lcl_height < 1000, 1.0,
                       np.where(lcl_height > 2000, 0.0,
                               (2000 - lcl_height) / 1000.0))
    
    # 5. CIN term: (MLCIN + 200)/150
    # MLCIN > -50 J/kg → 1.0, MLCIN < -200 J/kg → 0.0
    cin_term = np.where(mlcin > -50, 1.0,
                       np.where(mlcin < -200, 0.0,
                               (mlcin + 200) / 150.0))
    
    # STP CIN calculation
    stp_cin = cape_term * srh_term * shear_term * lcl_term * cin_term
    
    # Zero out where CAPE is too low for convection
    stp_cin = np.where(mlcape < 100, 0, stp_cin)
    
    # Ensure non-negative
    stp_cin = np.maximum(stp_cin, 0.0)
    
    # Mask invalid input data
    stp_cin = np.where((mlcape < 0) | (np.isnan(mlcape)) | (np.isnan(mlcin)) |
                      (np.isnan(effective_srh)) | (np.isnan(effective_shear)) |
                      (np.isnan(lcl_height)), np.nan, stp_cin)
    
    return stp_cin
