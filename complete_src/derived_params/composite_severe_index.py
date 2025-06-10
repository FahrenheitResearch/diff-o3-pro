from .common import *

def composite_severe_index(scp: np.ndarray, stp: np.ndarray, 
                         uh: np.ndarray, weights: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    Create composite severe weather index combining multiple parameters
    
    Args:
        scp: Supercell Composite Parameter
        stp: Significant Tornado Parameter
        uh: Updraft Helicity (m²/s²)
        weights: Tuple of weights for (SCP, STP, UH). Default: (0.4, 0.4, 0.2)
        
    Returns:
        Composite index (dimensionless)
    """
    if weights is None:
        weights = (0.4, 0.4, 0.2)
    
    # Normalize UH to 0-1 scale (using 200 as max)
    uh_norm = np.minimum(uh / 200.0, 1.0)
    
    # Normalize SCP and STP (cap at reasonable values)
    scp_norm = np.minimum(scp / 5.0, 1.0)
    stp_norm = np.minimum(stp / 3.0, 1.0)
    
    composite = (weights[0] * scp_norm + 
                weights[1] * stp_norm + 
                weights[2] * uh_norm)
    
    return composite
