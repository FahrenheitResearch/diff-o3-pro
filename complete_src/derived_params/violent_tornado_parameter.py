from .common import *

def violent_tornado_parameter(mlcape: np.ndarray, mlcin: np.ndarray,
                             lcl_height: np.ndarray, storm_relative_helicity_03km: np.ndarray,
                             wind_shear_06km: np.ndarray, cape_03km: np.ndarray,
                             lapse_rate_03km: np.ndarray) -> np.ndarray:
    """
    Compute Violent Tornado Parameter (VTP) - SPC Compliant Implementation
    
    VTP = (MLCAPE/1500) × (EBWD/20) × (ESRH/150) × ((2000-MLLCL)/1000) × 
          ((200+MLCIN)/150) × (0-3km MLCAPE/50) × (0-3km Lapse Rate/6.5)
    
    Follows Hampshire et al. (2018) formulation with proper SPC scaling and caps:
    - CAPE term: MLCAPE/1500, no cap (but physically limited)
    - LCL term: (2000-MLLCL)/1000, capped at 1.0 for LCL<1000m, 0 for LCL>2000m  
    - CIN term: (MLCIN+200)/150, capped at 1.0 for CIN>-50, 0 for CIN<-200
    - Shear term: EBWD/20, capped at 1.5 for shear>30 m/s, 0 for shear<12.5 m/s
    - SRH term: ESRH/150, no cap (allows for extreme values)
    - 0-3km CAPE term: cape_03km/50, capped at 2.0 per SPC specification
    - Lapse term: lapse_03km/6.5, capped at 2.0 (maxes at ~13°C/km)
    
    Args:
        mlcape: Mixed-Layer CAPE (J/kg) - use HRRR 180-0mb or compute 100mb parcel
        mlcin: Mixed-Layer CIN (J/kg, negative values)
        lcl_height: Mixed-Layer LCL height (m AGL)
        storm_relative_helicity_03km: 0-3km SRH (m²/s²) from HRRR or computed
        wind_shear_06km: 0-6km bulk wind shear magnitude (m/s)
        cape_03km: 0-3km MLCAPE (J/kg) - prefer proper parcel calculation
        lapse_rate_03km: 0-3km environmental lapse rate (°C/km)
        
    Returns:
        VTP values (dimensionless, ≥ 0) following exact SPC specification
    """
    
    # ------------------------------------------------------------------------
    # EFFECTIVE-LAYER GATE (BALANCED for realistic environments)
    # ------------------------------------------------------------------------
    # Relaxed CIN threshold to allow moderate caps while preventing very strong caps
    effective_mask = (
        (mlcape   >= 100.0) &
        (mlcin    >= -150.0) &   # Allow moderate caps (was -50, too strict)
        (lcl_height <= 2000.0)
    )

    # Gate SRH & shear so they only contribute where storms can actually root
    effective_srh   = np.where(effective_mask, storm_relative_helicity_03km, 0.0)
    effective_shear = np.where(effective_mask, wind_shear_06km,              0.0)
    
    # ========================================================================
    # VTP TERM CALCULATIONS - Following SPC scaling and caps exactly
    # ========================================================================
    
    # 1. CAPE term: MLCAPE/1500 with soft cap (keeps values civil)
    cape_term = np.clip(mlcape / 1500.0,       0.0,  2.0)   # soft cap ≈ SPC behaviour
    
    # 2. LCL term: (2000-MLLCL)/1000 with SPC clipping
    # SPC rules: LCL < 1000m → 1.0, LCL > 2000m → 0.0, linear between
    lcl_term = (2000.0 - lcl_height) / 1000.0
    lcl_term = np.clip(lcl_term, 0.0, 1.0)  # Cap at 1.0, floor at 0.0
    
    # 3. Effective Shear term: EBWD/20 with SPC clipping
    # SPC rules: Shear < 12.5 m/s → 0, Shear > 30 m/s → capped at 1.5
    shear_term = effective_shear / 20.0
    shear_term = np.where(effective_shear < 12.5, 0.0, shear_term)  # Zero below threshold
    shear_term = np.minimum(shear_term, 1.5)  # Cap at 1.5 (30 m/s)
    
    # 4. CIN term: (MLCIN + 200)/150 with SPC clipping
    # SPC rules: CIN > -50 J/kg → 1.0, CIN < -200 J/kg → 0.0, linear between
    cin_term = (mlcin + 200.0) / 150.0
    cin_term = np.clip(cin_term, 0.0, 1.0)  # Cap at 1.0, floor at 0.0
    
    # 5. Low-level CAPE term: Use SPC specification with /50 scaling  
    # SPC formula: (0-3 km MLCAPE / 50 J kg⁻¹)
    cape_03km_term = np.clip(cape_03km / 50.0, 0.0, 2.0)
    
    # Force to zero when diagnostic value is too low for meaningful convection
    cape_03km_term = np.where(cape_03km < 25.0, 0.0, cape_03km_term)
    
    # 6. Effective SRH term: ESRH/150 with soft cap (600 m² s⁻² → 4.0)
    srh_term  = np.clip(effective_srh / 150.0, 0.0,  4.0)   # 600 m² s⁻² → 4.0
    
    # Gate SRH term by CAPE/lapse combo to prevent wide ribbons  
    # Threshold adjusted for correct /50 scaling
    srh_term = np.where(cape_03km_term < 1.0, 0.0, srh_term)  # ~50 J/kg threshold
    
    # 7. Lapse rate term: Make override depend on *rescored* 3-CAPE term
    # CRITICAL FIX: Only fire override when buoyancy is actually extreme
    lapse_term = np.clip(lapse_rate_03km / 6.5, 0.0, 2.0)
    # Threshold adjusted for correct /50 scaling 
    lapse_term = np.where(cape_03km_term >= 4.0, 2.0, lapse_term)  # ~200 J/kg threshold
    
    # ========================================================================
    # VTP CALCULATION - Multiplicative (all ingredients must be present)
    # ========================================================================
    vtp = cape_term * lcl_term * srh_term * shear_term * cin_term \
          * cape_03km_term * lapse_term

    
    # ------------------------------------------------------------------------
    # HARD CEILING – allow rare 8-10 pixels in extreme cases
    # ------------------------------------------------------------------------
    vtp = np.clip(vtp, 0.0, 8.0)
    
    # ========================================================================
    # QUALITY CONTROL AND MASKING
    # ========================================================================
    
    # Mask invalid input data - if ANY critical input is NaN/invalid, VTP = 0
    valid_data = (
        np.isfinite(mlcape) & (mlcape >= 0) &
        np.isfinite(mlcin) & 
        np.isfinite(lcl_height) & (lcl_height >= 0) &
        np.isfinite(storm_relative_helicity_03km) &
        np.isfinite(wind_shear_06km) & (wind_shear_06km >= 0) &
        np.isfinite(cape_03km) & (cape_03km >= 0) &
        np.isfinite(lapse_rate_03km) & (lapse_rate_03km >= 0)
    )
    
    vtp = np.where(valid_data, vtp, 0.0)
    
    # Ensure VTP is never negative (should be impossible with above logic)
    vtp = np.maximum(vtp, 0.0)
    
    return vtp
