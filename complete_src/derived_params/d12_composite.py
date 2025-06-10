from .common import *

def d12_composite(mlcape: np.ndarray, wind_shear_06km: np.ndarray,
                  srh_01km: np.ndarray, mlcin: np.ndarray,
                  lcl_height: np.ndarray, pwat: np.ndarray) -> np.ndarray:
    """
    D12 Severe Weather Composite Index - "Dream Conditions" Detector
    
    An extremely selective composite index designed to identify only the rarest,
    most ideal severe weather environments where ALL critical parameters are
    simultaneously near-optimal. Uses exponential penalties to harshly punish
    any deviation from perfect conditions.
    
    Philosophy: Perfect severe weather environments are exceptionally rare and
    require simultaneous optimization of instability, shear, helicity, minimal
    inhibition, low LCL, and optimal moisture. This index is intentionally strict
    to highlight only "textbook perfect" supercell environments.
    
    Ideal Thresholds (for maximum values ~1.0):
    - CAPE: ≥4000 J/kg (perfect conditions), ≥2500 J/kg (highly favorable)
    - 0-6km Shear: ≥30 m/s (optimal), ≥20 m/s (highly favorable)  
    - 0-1km SRH: ≥400 m²/s² (optimal), ≥300 m²/s² (strong rotation)
    - CIN: ≤-25 J/kg (minimal inhibition), rapidly deteriorates with stronger CIN
    - LCL: ≤750m (ideal), ≤1000m (highly favorable)
    - PWAT: 30-45mm optimal range, exponential drop-off outside this range
    
    Args:
        mlcape: Mixed Layer CAPE (J/kg) - conditional instability
        wind_shear_06km: 0-6km bulk wind shear magnitude (m/s) - storm organization
        srh_01km: 0-1km Storm Relative Helicity (m²/s²) - low-level rotation
        mlcin: Mixed Layer CIN (J/kg, negative) - convective inhibition  
        lcl_height: Lifting Condensation Level height (m) - boundary layer moisture
        pwat: Precipitable water (mm) - atmospheric moisture content
        
    Returns:
        D12 values (0-1 scale). Values approaching 1.0 indicate exceptionally 
        rare "dream conditions". Most areas will show very low values (<0.3).
    """
    
    # ========================================================================
    # STRICT PARAMETER NORMALIZATION WITH EXPONENTIAL PENALTIES
    # ========================================================================
    
    # CAPE: Exponential reward above 2500, optimal at 4000+ J/kg
    # Harsh drop-off below 2500 J/kg using squared penalty
    cape_norm = np.clip((mlcape - 2000) / 2000, 0, 1)**2
    
    # SHEAR: Exponential reward above 20 m/s, optimal at 30+ m/s
    # Strong penalty below 15 m/s threshold
    shear_norm = np.clip((wind_shear_06km - 15) / 15, 0, 1)**2
    
    # SRH: Exponential reward above 300 m²/s², optimal at 400+ m²/s²
    # Harsh penalty below 150 m²/s² threshold  
    srh_norm = np.clip((srh_01km - 150) / 150, 0, 1)**2
    
    # CIN: Exponential penalty for CIN stronger than -25 J/kg
    # Near-zero CIN gets maximum score, rapidly deteriorates with stronger inhibition
    cin_norm = np.exp(np.clip(mlcin / 25, None, 0))  # Exponential penalty for CIN < -25
    
    # LCL: Exponential penalty above 1000m, optimal below 750m
    # Very harsh punishment for high LCL heights  
    lcl_norm = np.exp(-np.clip((lcl_height - 750) / 250, 0, None))
    
    # PWAT: Optimal around 35mm, exponential drop-off outside 30-45mm range
    # Too dry or too moist both hurt severe weather potential
    pwat_norm = np.exp(-((pwat - 35) / 8)**2)
    
    # ========================================================================
    # MULTIPLICATIVE COMPOSITE WITH HARSH ALL-OR-NOTHING PHILOSOPHY
    # ========================================================================
    
    # All parameters must be simultaneously excellent
    # Sixth root ensures no single parameter can compensate for poor others
    d12 = np.power(cape_norm * shear_norm * srh_norm * 
                   cin_norm * lcl_norm * pwat_norm, 1.0/6.0)
    
    # Additional quality control: zero out unrealistic combinations
    # Even small deficiencies should prevent high composite values
    d12 = np.where((mlcape < 1000) | (wind_shear_06km < 10) | 
                   (srh_01km < 50) | (lcl_height > 2500), 0.0, d12)
    
    return d12