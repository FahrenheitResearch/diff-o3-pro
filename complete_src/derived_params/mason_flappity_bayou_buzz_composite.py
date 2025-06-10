from .common import *

def mason_flappity_bayou_buzz_composite(pwat: np.ndarray, mlcape: np.ndarray,
                                      wind_shear_06km: np.ndarray, 
                                      u_850mb: np.ndarray, v_850mb: np.ndarray,
                                      q_850mb: np.ndarray, absv_850mb: np.ndarray) -> np.ndarray:
    """
    Compute Mason-Flappity Bayou Buzz Composite (MF-Buzz)
    
    A mesoscale diagnostic for synoptic-to-convective "interest" in SW Louisiana.
    Weights five independent processes that drive high-impact weather along the
    NW Gulf Coast: moisture anomaly, convective potential, storm organization,
    mesoscale forcing, and low-level vorticity.
    
    Formula: MF-Buzz = M + J + S + C + V (0-5 scale)
    Where:
    - M = min(PWAT_anom/15mm, 1) - Column moisture anomaly
    - J = min(MLCAPE/2500, 1) - Convective potential  
    - S = min(|ΔV_0-6|/25, 1) - Storm-scale organization (0-6km shear)
    - C = min(4×10⁴ |∇·(qV)|_850, 1) - Mesoscale forcing (moisture flux convergence)
    - V = min(2×10⁵ |ζ|_850, 1) - Low-level vorticity
    
    Args:
        pwat: Precipitable water (mm or kg/m²)
        mlcape: Mixed Layer CAPE (J/kg)
        wind_shear_06km: 0-6km bulk wind shear magnitude (m/s)
        u_850mb: 850mb u-wind component (m/s)
        v_850mb: 850mb v-wind component (m/s)
        q_850mb: 850mb specific humidity (kg/kg)
        absv_850mb: 850mb absolute vorticity (s⁻¹)
        
    Returns:
        MF-Buzz values (0-5 scale, dimensionless)
    """
    # 1. Moisture anomaly term: M = min(PWAT_anom/15mm, 1)
    # Using rough Gulf Coast climatology estimate (warm season ~35mm)
    pwat_climo_est = 35.0  # mm (MAMJJAS average for SW Louisiana)
    pwat_anom = pwat - pwat_climo_est
    M = np.clip(pwat_anom / 15.0, 0.0, 1.0)
    
    # 2. Convective potential term: J = min(MLCAPE/2500, 1)
    J = np.clip(mlcape / 2500.0, 0.0, 1.0)
    
    # 3. Storm organization term: S = min(|ΔV_0-6|/25, 1)
    # Using 0-6km bulk shear as proxy for storm organization
    S = np.clip(wind_shear_06km / 25.0, 0.0, 1.0)
    
    # 4. Mesoscale forcing term: C = min(4×10⁴ |∇·(qV)|_850, 1)
    # Moisture flux convergence at 850mb
    # Simple finite difference approximation
    # Note: This is a simplified calculation - ideally would use proper grid spacing
    qu_850 = q_850mb * u_850mb
    qv_850 = q_850mb * v_850mb
    
    # Estimate gradients (this is approximate without proper grid spacing)
    # Using central differences where possible
    du_dx = np.gradient(qu_850, axis=1)  # d(qu)/dx
    dv_dy = np.gradient(qv_850, axis=0)  # d(qv)/dy
    moisture_flux_div = du_dx + dv_dy
    
    C = np.clip(4e4 * np.abs(moisture_flux_div), 0.0, 1.0)
    
    # 5. Low-level vorticity term: V = min(2×10⁵ |ζ|_850, 1)
    # Using 850mb absolute vorticity
    V = np.clip(2e5 * np.abs(absv_850mb), 0.0, 1.0)
    
    # Mason-Flappity Bayou Buzz Composite (additive!)
    # Each term contributes 0-1, total range 0-5
    mf_buzz = M + J + S + C + V
    
    # Ensure 0-5 bounds
    mf_buzz = np.clip(mf_buzz, 0.0, 5.0)
    
    # Mask invalid data
    mf_buzz = np.where(
        (np.isnan(pwat)) | (np.isnan(mlcape)) | (np.isnan(wind_shear_06km)) |
        (np.isnan(u_850mb)) | (np.isnan(v_850mb)) | (np.isnan(q_850mb)) |
        (np.isnan(absv_850mb)),
        np.nan, mf_buzz
    )
    
    return mf_buzz
