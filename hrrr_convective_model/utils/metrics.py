"""
Evaluation metrics for deterministic and ensemble weather forecasts.
Implements RMSE, CRPS (Eq. 22), and Energy Score (Eq. 23) from the DEF paper.
"""
import torch
import numpy as np
from typing import Optional, Dict, Tuple


def rmse(y: torch.Tensor, yhat: torch.Tensor, 
         dim: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Root Mean Square Error.
    
    Args:
        y: Ground truth [B, C, H, W] or [B, T, C, H, W]
        yhat: Predictions (same shape as y)
        dim: Dimensions to reduce over. If None, reduces over all dims.
    
    Returns:
        RMSE value(s)
    """
    if dim is None:
        return torch.sqrt(torch.mean((y - yhat) ** 2))
    else:
        return torch.sqrt(torch.mean((y - yhat) ** 2, dim=dim))


def mae(y: torch.Tensor, yhat: torch.Tensor,
        dim: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Mean Absolute Error."""
    if dim is None:
        return torch.mean(torch.abs(y - yhat))
    else:
        return torch.mean(torch.abs(y - yhat), dim=dim)


def bias(y: torch.Tensor, yhat: torch.Tensor,
         dim: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Mean bias (yhat - y)."""
    if dim is None:
        return torch.mean(yhat - y)
    else:
        return torch.mean(yhat - y, dim=dim)


def crps_ensemble(obs: torch.Tensor, ens: torch.Tensor,
                  dim: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Continuous Ranked Probability Score for ensemble forecasts.
    
    From Eq. 22 in the DEF paper:
    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    
    Args:
        obs: Observations [B, C, H, W] or [B, T, C, H, W]
        ens: Ensemble forecasts [B, N, C, H, W] or [B, T, N, C, H, W]
             where N is ensemble size
        dim: Dimensions to average over (spatial + channel dims)
    
    Returns:
        CRPS values [B] or [B, T] depending on input
    """
    # Handle different input shapes
    if obs.dim() == 4 and ens.dim() == 5:  # [B,C,H,W] and [B,N,C,H,W]
        obs = obs.unsqueeze(1)  # [B,1,C,H,W]
        reduce_dims = (1, 2, 3, 4) if dim is None else dim
    elif obs.dim() == 5 and ens.dim() == 6:  # [B,T,C,H,W] and [B,T,N,C,H,W]
        obs = obs.unsqueeze(2)  # [B,T,1,C,H,W]
        reduce_dims = (2, 3, 4, 5) if dim is None else dim
    else:
        raise ValueError(f"Unexpected shapes: obs {obs.shape}, ens {ens.shape}")
    
    # First term: E[|X - y|]
    term1 = torch.mean(torch.abs(ens - obs), dim=reduce_dims)
    
    # Second term: E[|X - X'|]
    # We need to compute pairwise differences between ensemble members
    N = ens.shape[1] if obs.dim() == 5 else ens.shape[2]
    
    # Efficient computation without expanding memory too much
    term2 = 0
    for i in range(N):
        for j in range(i+1, N):
            if obs.dim() == 5:  # [B,N,C,H,W] case
                diff = torch.abs(ens[:, i] - ens[:, j])
                # For [B,N,C,H,W], reduce_dims=(1,2,3,4), so we want dims (1,2,3) for diff
                term2 = term2 + torch.mean(diff, dim=(1, 2, 3))
            else:  # [B,T,N,C,H,W] case  
                diff = torch.abs(ens[:, :, i] - ens[:, :, j])
                # For [B,T,N,C,H,W], reduce_dims=(2,3,4,5), so we want dims (2,3,4) for diff 
                term2 = term2 + torch.mean(diff, dim=(2, 3, 4))
    
    # Normalize by number of pairs
    term2 = term2 * 2 / (N * (N - 1))
    
    return term1 - 0.5 * term2


def energy_score(obs: torch.Tensor, ens: torch.Tensor) -> torch.Tensor:
    """Energy Score for multivariate ensemble forecasts.
    
    From Eq. 23 in the DEF paper:
    ES = E[||X - y||] - 0.5 * E[||X - X'||]
    
    Args:
        obs: Observations [B, C, H, W]
        ens: Ensemble forecasts [B, N, C, H, W] where N is ensemble size
    
    Returns:
        Energy score values [B]
    """
    B, N, C, H, W = ens.shape
    
    # First term: E[||X - y||_2]
    # Compute L2 norm over spatial dimensions for each channel
    obs_expanded = obs.unsqueeze(1)  # [B, 1, C, H, W]
    diff_obs = ens - obs_expanded  # [B, N, C, H, W]
    
    # Flatten spatial dimensions
    diff_obs_flat = diff_obs.view(B, N, C, -1)  # [B, N, C, H*W]
    
    # L2 norm over spatial dimensions for each channel, then sum over channels
    norm_obs = torch.sqrt(torch.sum(diff_obs_flat ** 2, dim=-1))  # [B, N, C]
    norm_obs = torch.sum(norm_obs, dim=-1)  # [B, N]
    term1 = torch.mean(norm_obs, dim=1)  # [B]
    
    # Second term: E[||X - X'||_2] 
    # Compute pairwise differences between ensemble members
    term2 = torch.zeros(B, device=ens.device)
    
    for i in range(N):
        for j in range(i+1, N):
            diff_ens = ens[:, i] - ens[:, j]  # [B, C, H, W]
            diff_ens_flat = diff_ens.view(B, C, -1)  # [B, C, H*W]
            norm_ens = torch.sqrt(torch.sum(diff_ens_flat ** 2, dim=-1))  # [B, C]
            norm_ens = torch.sum(norm_ens, dim=-1)  # [B]
            term2 = term2 + norm_ens
    
    # Normalize by number of pairs
    term2 = term2 * 2 / (N * (N - 1))
    
    return term1 - 0.5 * term2


def ensemble_spread(ens: torch.Tensor, dim: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Compute ensemble spread (standard deviation).
    
    Args:
        ens: Ensemble forecasts [B, N, C, H, W]
        dim: Dimensions to compute spread over
    
    Returns:
        Spread values
    """
    if dim is None:
        # Spread across ensemble members
        return torch.std(ens, dim=1)
    else:
        return torch.std(ens, dim=dim)


def ensemble_mean(ens: torch.Tensor) -> torch.Tensor:
    """Compute ensemble mean.
    
    Args:
        ens: Ensemble forecasts [B, N, C, H, W]
    
    Returns:
        Ensemble mean [B, C, H, W]
    """
    return torch.mean(ens, dim=1)


def rank_histogram(obs: torch.Tensor, ens: torch.Tensor, 
                   n_bins: int = 10) -> torch.Tensor:
    """Compute rank histogram for ensemble calibration assessment.
    
    Args:
        obs: Observations [B, C, H, W]
        ens: Ensemble forecasts [B, N, C, H, W]
        n_bins: Number of bins (should be N+1 for N ensemble members)
    
    Returns:
        Rank histogram counts [n_bins]
    """
    B, N, C, H, W = ens.shape
    
    # Flatten spatial and channel dimensions
    obs_flat = obs.view(-1)  # [B*C*H*W]
    ens_flat = ens.view(B*C*H*W, N)  # [B*C*H*W, N]
    
    # For each observation, find its rank among ensemble members
    ranks = torch.zeros_like(obs_flat, dtype=torch.long)
    
    for i in range(obs_flat.shape[0]):
        # Count how many ensemble members are less than observation
        rank = torch.sum(ens_flat[i] < obs_flat[i]).item()
        ranks[i] = rank
    
    # Create histogram
    hist = torch.histc(ranks.float(), bins=n_bins, min=0, max=N)
    
    return hist


def reliability_diagram(obs: torch.Tensor, ens_prob: torch.Tensor, 
                        threshold: float, n_bins: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reliability diagram data for probabilistic forecasts.
    
    Args:
        obs: Binary observations [B, H, W] (1 if event occurred, 0 otherwise)
        ens_prob: Ensemble-based probabilities [B, H, W] (fraction of members predicting event)
        threshold: Not used in this version (probabilities already computed)
        n_bins: Number of probability bins
    
    Returns:
        forecast_probs: Binned forecast probabilities
        observed_freq: Observed frequencies in each bin
    """
    # Flatten
    obs_flat = obs.view(-1)
    prob_flat = ens_prob.view(-1)
    
    # Create bins
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    forecast_probs = torch.zeros(n_bins)
    observed_freq = torch.zeros(n_bins)
    
    for i in range(n_bins):
        # Find points in this bin
        mask = (prob_flat >= bin_edges[i]) & (prob_flat < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = mask | (prob_flat == bin_edges[i+1])
        
        if mask.sum() > 0:
            forecast_probs[i] = prob_flat[mask].mean()
            observed_freq[i] = obs_flat[mask].float().mean()
        else:
            forecast_probs[i] = (bin_edges[i] + bin_edges[i+1]) / 2
            observed_freq[i] = float('nan')
    
    return forecast_probs, observed_freq


def spread_error_ratio(ensemble_preds: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Compute spread-error ratio for ensemble calibration.
    
    Well-calibrated ensembles should have ratio ≈ 1.
    
    Args:
        ensemble_preds: Ensemble predictions [M, B, C, H, W]
        truth: Ground truth [B, C, H, W]
    
    Returns:
        Spread-error ratio
    """
    # Compute ensemble spread
    spread = ensemble_preds.std(dim=0).mean()
    
    # Compute ensemble mean
    ens_mean = ensemble_preds.mean(dim=0)
    
    # Compute RMSE of ensemble mean
    error = torch.sqrt(((ens_mean - truth) ** 2).mean())
    
    return spread / (error + 1e-8)


def spread_skill_ratio(spread: torch.Tensor, rmse: torch.Tensor) -> torch.Tensor:
    """Compute spread-skill ratio.
    
    A well-calibrated ensemble should have ratio ≈ 1.
    
    Args:
        spread: Ensemble spread values
        rmse: RMSE of ensemble mean
    
    Returns:
        Spread-skill ratio
    """
    return spread / (rmse + 1e-8)  # Add small epsilon to avoid division by zero


def compute_all_metrics(obs: torch.Tensor, ens: torch.Tensor,
                       compute_energy: bool = True) -> Dict[str, torch.Tensor]:
    """Compute all relevant metrics for ensemble evaluation.
    
    Args:
        obs: Observations [B, C, H, W]
        ens: Ensemble forecasts [B, N, C, H, W]
        compute_energy: Whether to compute energy score (expensive)
    
    Returns:
        Dictionary of metric values
    """
    ens_mean = ensemble_mean(ens)
    spread = ensemble_spread(ens)
    
    metrics = {
        'rmse': rmse(obs, ens_mean),
        'mae': mae(obs, ens_mean),
        'bias': bias(obs, ens_mean),
        'crps': crps_ensemble(obs, ens),
        'spread': torch.mean(spread),
        'spread_skill_ratio': spread_skill_ratio(torch.mean(spread), rmse(obs, ens_mean))
    }
    
    if compute_energy:
        metrics['energy_score'] = torch.mean(energy_score(obs, ens))
    
    return metrics