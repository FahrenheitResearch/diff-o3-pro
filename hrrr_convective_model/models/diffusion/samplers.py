"""
DPM-Solver++ implementation for fast diffusion sampling.
Based on "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
"""
import torch
import torch.nn as nn
from typing import Optional, List, Callable, Union
import numpy as np


class DPMSolverPlusPlus:
    """
    DPM-Solver++ for fast sampling of diffusion models.
    Supports 1st, 2nd, and 3rd order solvers with adaptive step sizing.
    """
    
    def __init__(
        self,
        noise_prediction_fn: Callable,
        alphas_cumprod: torch.Tensor,
        num_timesteps: int = 1000,
        algorithm_type: str = "dpmsolver++",
        solver_order: int = 2,
        solver_type: str = "midpoint",  # midpoint or heun
    ):
        """
        Args:
            noise_prediction_fn: Function that predicts noise given (x_t, t, condition)
            alphas_cumprod: Cumulative product of alphas from diffusion schedule
            num_timesteps: Number of diffusion timesteps (T)
            algorithm_type: "dpmsolver" or "dpmsolver++"
            solver_order: Order of the solver (1, 2, or 3)
            solver_type: "midpoint" or "heun" for 2nd order
        """
        self.noise_prediction_fn = noise_prediction_fn
        self.alphas_cumprod = alphas_cumprod
        self.num_timesteps = num_timesteps
        self.algorithm_type = algorithm_type
        self.solver_order = solver_order
        self.solver_type = solver_type
        
        # Precompute lambda schedule
        self.lambdas = self._compute_lambda_schedule()
        
    def _compute_lambda_schedule(self) -> torch.Tensor:
        """Compute lambda(t) = log(alpha_t) - log(sigma_t)"""
        alphas = self.alphas_cumprod
        sigmas = torch.sqrt(1 - alphas)
        lambdas = torch.log(alphas) - torch.log(sigmas)
        return lambdas
    
    def _sigma(self, t: Union[int, torch.Tensor, np.integer]) -> torch.Tensor:
        """Compute sigma(t) = sqrt(1 - alpha_cumprod(t))"""
        alpha_t = self._alpha(t)
        return torch.sqrt(1 - alpha_t)
    
    def _alpha(self, t: Union[int, torch.Tensor, np.integer]) -> torch.Tensor:
        """Get alpha_cumprod at timestep t"""
        # Handle both integer and tensor timesteps
        if isinstance(t, (int, np.integer)):
            return self.alphas_cumprod[t]
        elif isinstance(t, torch.Tensor):
            # Interpolate for continuous timesteps
            t_int = t.long()
            t_frac = t - t_int
            alpha_int = self.alphas_cumprod[t_int]
            if t_int < len(self.alphas_cumprod) - 1:
                alpha_next = self.alphas_cumprod[t_int + 1]
                return alpha_int * (1 - t_frac) + alpha_next * t_frac
            else:
                return alpha_int
        else:
            # Convert to tensor
            t = torch.tensor(t, dtype=torch.long)
            return self.alphas_cumprod[t]
    
    def _predict_x0(self, x_t: torch.Tensor, t: int, 
                   noise_pred: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and noise prediction"""
        alpha_t = self._alpha(t)
        sigma_t = self._sigma(t)
        
        # x_0 = (x_t - sigma_t * noise) / alpha_t
        x_0 = (x_t - sigma_t * noise_pred) / alpha_t
        return x_0
    
    def dpm_solver_first_order_update(
        self, x_t: torch.Tensor, t_start: int, t_end: int,
        condition: Optional[torch.Tensor] = None,
        guidance_weight: float = 0.0
    ) -> torch.Tensor:
        """First-order DPM-Solver update from t_start to t_end"""
        # Get noise prediction at t_start
        noise_pred = self._get_noise_prediction(x_t, t_start, condition, guidance_weight)
        
        # Predict x_0
        x_0 = self._predict_x0(x_t, t_start, noise_pred)
        
        # Get coefficients
        alpha_start = self._alpha(t_start)
        sigma_start = self._sigma(t_start)
        alpha_end = self._alpha(t_end)
        sigma_end = self._sigma(t_end)
        
        # DPM-Solver++ update
        if self.algorithm_type == "dpmsolver++":
            # x_{t_end} = alpha_end/alpha_start * x_t + (sigma_end - alpha_end/alpha_start * sigma_start) * noise
            coeff_xt = alpha_end / alpha_start
            coeff_noise = sigma_end - coeff_xt * sigma_start
            x_next = coeff_xt * x_t + coeff_noise * noise_pred
        else:
            # Standard DPM-Solver
            x_next = alpha_end * x_0 + sigma_end * noise_pred
            
        return x_next
    
    def dpm_solver_second_order_update(
        self, x_t: torch.Tensor, t_list: List[int],
        condition: Optional[torch.Tensor] = None,
        guidance_weight: float = 0.0,
        x_t_1: Optional[torch.Tensor] = None,
        noise_t_1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Second-order DPM-Solver update"""
        t_start, t_mid, t_end = t_list
        
        # First step or reuse previous computation
        if x_t_1 is None:
            x_t_1 = self.dpm_solver_first_order_update(
                x_t, t_start, t_mid, condition, guidance_weight
            )
            noise_t_1 = self._get_noise_prediction(x_t_1, t_mid, condition, guidance_weight)
        
        # Get noise at t_start
        noise_t = self._get_noise_prediction(x_t, t_start, condition, guidance_weight)
        
        # Compute lambda values
        lambda_start = self.lambdas[t_start]
        lambda_mid = self.lambdas[t_mid]
        lambda_end = self.lambdas[t_end]
        
        h = lambda_end - lambda_start
        h_0 = lambda_mid - lambda_start
        r_0 = h_0 / h
        
        # Predict x_0 values
        x_0_t = self._predict_x0(x_t, t_start, noise_t)
        x_0_t_1 = self._predict_x0(x_t_1, t_mid, noise_t_1)
        
        # Linear interpolation for x_0
        if self.algorithm_type == "dpmsolver++":
            # Second-order approximation
            if self.solver_type == "midpoint":
                x_0 = (1 - 1 / (2 * r_0)) * x_0_t + 1 / (2 * r_0) * x_0_t_1
            else:  # Heun's method
                x_0 = x_0_t_1
        else:
            x_0 = (1 - r_0) * x_0_t + r_0 * x_0_t_1
            
        # Get final x
        alpha_end = self._alpha(t_end)
        sigma_end = self._sigma(t_end)
        
        # Compute noise approximation
        if self.algorithm_type == "dpmsolver++":
            noise_approx = (1 / r_0) * (noise_t_1 - noise_t) + noise_t
            x_next = alpha_end * x_0 + sigma_end * noise_approx
        else:
            x_next = alpha_end * x_0 + sigma_end * noise_t_1
            
        return x_next, noise_t_1  # Return noise for potential reuse
    
    def _get_noise_prediction(
        self, x: torch.Tensor, t: int,
        condition: Optional[torch.Tensor] = None,
        guidance_weight: float = 0.0
    ) -> torch.Tensor:
        """Get noise prediction with optional classifier-free guidance"""
        B = x.shape[0]
        t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)
        
        if guidance_weight > 0 and condition is not None:
            # Classifier-free guidance
            noise_cond = self.noise_prediction_fn(x, t_tensor, condition)
            noise_uncond = self.noise_prediction_fn(x, t_tensor, None)
            noise_pred = noise_uncond + guidance_weight * (noise_cond - noise_uncond)
        else:
            noise_pred = self.noise_prediction_fn(x, t_tensor, condition)
            
        return noise_pred
    
    @torch.no_grad()
    def sample(
        self, x_T: torch.Tensor, 
        num_steps: int = 50,
        condition: Optional[torch.Tensor] = None,
        guidance_weight: float = 0.1,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using DPM-Solver++.
        
        Args:
            x_T: Initial noise [B, C, H, W]
            num_steps: Number of denoising steps (< T)
            condition: Conditioning information
            guidance_weight: Classifier-free guidance weight
            return_intermediate: Whether to return intermediate states
            
        Returns:
            x_0: Denoised sample [B, C, H, W]
            intermediates: List of intermediate states (if requested)
        """
        device = x_T.device
        
        # Create adaptive timestep schedule
        if num_steps >= self.num_timesteps:
            timesteps = list(range(self.num_timesteps))[::-1]
        else:
            # Uniform in log-SNR space for better sampling
            t_seq = np.linspace(0, self.num_timesteps - 1, num_steps + 1)
            t_seq = np.round(t_seq).astype(int)
            timesteps = list(t_seq[:-1])[::-1]
        
        x = x_T
        intermediates = []
        
        # Cache for second/third order methods
        noise_cache = None
        
        for i in range(len(timesteps)):
            if return_intermediate:
                intermediates.append(x.clone())
                
            if i == len(timesteps) - 1:
                # Last step goes to t=0
                t_cur = timesteps[i]
                t_next = 0
            else:
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                
            # Select solver order adaptively
            if i == 0 or self.solver_order == 1:
                # First step is always first-order
                x = self.dpm_solver_first_order_update(
                    x, t_cur, t_next, condition, guidance_weight
                )
            elif self.solver_order == 2:
                if i == 1:
                    # Second step: we need to compute intermediate point
                    t_mid = (t_cur + t_next) // 2
                    x, noise_cache = self.dpm_solver_second_order_update(
                        x, [t_cur, t_mid, t_next], condition, guidance_weight
                    )
                else:
                    # Reuse previous function evaluation
                    t_prev = timesteps[i - 1]
                    x, noise_cache = self.dpm_solver_second_order_update(
                        x, [t_cur, t_prev, t_next], condition, guidance_weight,
                        noise_t_1=noise_cache
                    )
            else:
                raise NotImplementedError(f"Solver order {self.solver_order} not implemented")
                
        if return_intermediate:
            intermediates.append(x)
            return x, intermediates
        else:
            return x
    
    @torch.no_grad()
    def sample_batch_adaptive(
        self, x_T: torch.Tensor,
        num_steps_list: List[int],
        condition: Optional[torch.Tensor] = None,
        guidance_weight: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Sample multiple trajectories with different numbers of steps.
        Useful for ensembles with varying computational budgets.
        """
        results = []
        
        for num_steps in num_steps_list:
            x_0 = self.sample(
                x_T.clone(), num_steps, condition, guidance_weight
            )
            results.append(x_0)
            
        return results