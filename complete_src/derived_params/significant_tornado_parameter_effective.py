from .common import *

def significant_tornado_parameter_effective(mlcape: np.ndarray, mlcin: np.ndarray,
                                          effective_srh: np.ndarray, effective_shear: np.ndarray,
                                          mllcl_height: np.ndarray) -> np.ndarray:
    """
    Compute Enhanced STP using Effective Layers - Latest SPC Definition
    
    STP_eff = (MLCAPE/1500) × (Effective_SRH/150) × (Effective_Shear/20) × ((2000-MLLCL)/1000) × ((MLCIN+200)/150)
    
    Uses effective shear and effective SRH instead of fixed layers for improved accuracy.
    Based on Thompson et al. (2004) updated methodology with effective layer approach.
    
    Args:
        mlcape: Mixed Layer CAPE (J/kg)
        mlcin: Mixed Layer CIN (J/kg, negative values)
        effective_srh: Effective Storm Relative Helicity (m²/s²)
        effective_shear: Effective bulk shear magnitude (m/s)
        mllcl_height: Mixed Layer LCL height (m AGL)
        
    Returns:
        Enhanced STP values (dimensionless, always ≥ 0)
        
    Interpretation:
        STP_eff > 1: Heightened significant tornado risk
        STP_eff > 4: Extreme tornado potential
        STP_eff > 8: Historic outbreak-level environment
    """
    # 1. CAPE term: MLCAPE/1500 (no arbitrary cap)
    cape_term = np.maximum(mlcape / 1500.0, 0)
    
    # 2. LCL term: (2000-MLLCL)/1000 with proper clipping
    # MLLCL < 1000m → 1.0 (extremely favorable low cloud base)
    # MLLCL > 2000m → 0.0 (unfavorable high cloud base)
    lcl_term = np.where(mllcl_height < 1000, 1.0,
                       np.where(mllcl_height > 2000, 0.0,
                               (2000 - mllcl_height) / 1000.0))
    
    # 3. Effective SRH term: Effective_SRH/150 (preserve sign but cap negative)
    srh_term = np.maximum(effective_srh / 150.0, 0)
    
    # 4. Effective Shear term: Effective_Shear/20 m/s with SPC clipping
    # < 10 m/s → 0 (insufficient shear)
    # > 20 m/s → cap at 1.0 (normalization factor)
    shear_term = np.where(effective_shear < 10, 0,
                         np.where(effective_shear >= 20, 1.0, 
                                 effective_shear / 20.0))
    
    # 5. CIN term: (MLCIN + 200)/150 - Official SPC formula  
    # MLCIN > -50 J/kg → 1.0 (weak/no cap)
    # MLCIN < -200 J/kg → 0.0 (strong cap suppresses tornadoes)
    cin_term = np.where(mlcin > -50, 1.0,
                       np.where(mlcin < -200, 0.0,
                               np.maximum((mlcin + 200) / 150.0, 0.0)))
    
    # Enhanced STP calculation (multiplicative)
    stp_eff = cape_term * lcl_term * srh_term * shear_term * cin_term
    
    # Zero out where CAPE is too low for convection
    stp_eff = np.where(mlcape < 100, 0, stp_eff)
    
    # Ensure STP is never negative
    stp_eff = np.maximum(stp_eff, 0.0)
    
    # Mask invalid input data
    stp_eff = np.where((mlcape < 0) | (np.isnan(mlcape)) | (np.isnan(mlcin)) |
                      (np.isnan(effective_srh)) | (np.isnan(effective_shear)) | 
                      (mllcl_height < 0) | (np.isnan(mllcl_height)), 
                      np.nan, stp_eff)
    
    return stp_eff
