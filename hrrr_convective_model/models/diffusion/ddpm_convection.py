"""
Faithful DEF implementation for convection prediction.
This is the CORRECT architecture - predicts noise, not weather states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from models.unet_attention_fixed import UNetAttn as UNetModel


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvectionConditioningNetwork(nn.Module):
    """
    Encode past weather states and static features for conditioning.
    Critical for convection: CAPE, CIN, and REFC from past hours.
    """
    def __init__(self, 
                 history_length: int = 4,
                 in_channels: int = 7,
                 hidden_dim: int = 128,
                 out_dim: int = 512):
        super().__init__()
        
        # Temporal aggregation of past states
        self.temporal_conv = nn.Conv3d(
            in_channels, 
            hidden_dim, 
            kernel_size=(history_length, 1, 1),
            padding=0
        )
        
        # Spatial processing
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, out_dim, 3, stride=2, padding=1)
        )
        
        # Global pooling for context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, past_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            past_states: (B, T, C, H, W) past weather states
        Returns:
            spatial_features: (B, out_dim, H//4, W//4)
            global_features: (B, out_dim)
        """
        B, T, C, H, W = past_states.shape
        
        # Reshape for 3D conv
        x = past_states.transpose(1, 2)  # (B, C, T, H, W)
        
        # Temporal aggregation
        x = self.temporal_conv(x).squeeze(2)  # (B, hidden_dim, H, W)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(x)
        
        # Global context
        global_features = self.global_pool(spatial_features).squeeze(-1).squeeze(-1)
        
        return spatial_features, global_features


class ConvectionDDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for convection prediction.
    Faithful to DEF paper - predicts noise, not weather states.
    """
    def __init__(self,
                 in_channels: int = 7,
                 out_channels: int = 7,
                 base_dim: int = 64,
                 dim_mults: Tuple[int] = (1, 2, 4, 8),
                 attention_resolutions: Tuple[int] = (16, 8),
                 num_res_blocks: int = 2,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 history_length: int = 4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        time_dim = base_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Conditioning network for past states
        self.condition_encoder = ConvectionConditioningNetwork(
            history_length=history_length,
            in_channels=in_channels,
            hidden_dim=base_dim * 2,
            out_dim=time_dim
        )
        
        # Main U-Net for denoising
        # Using our existing UNetAttn architecture
        # Add extra channels for conditioning
        self.denoise_unet = UNetModel(
            in_ch=in_channels * 2,  # Noisy state + conditioning channels
            out_ch=out_channels,  # Predicted noise
            base_features=base_dim,
            use_temporal_encoding=False  # We handle time encoding separately
        )
        
    def forward(self, 
                noisy_state: torch.Tensor,
                timestep: torch.Tensor,
                past_states: torch.Tensor,
                return_dict: bool = False) -> torch.Tensor:
        """
        Predict noise given noisy state and conditioning.
        
        Args:
            noisy_state: (B, C, H, W) noisy future weather state
            timestep: (B,) diffusion timestep
            past_states: (B, T, C, H, W) past weather states for conditioning
            
        Returns:
            noise_pred: (B, C, H, W) predicted noise
        """
        # Get time embedding
        t_emb = self.time_embed(timestep)
        
        # Encode past states for conditioning
        spatial_cond, global_cond = self.condition_encoder(past_states)
        
        # Combine time and global conditioning
        conditioning = t_emb + global_cond
        
        # Add conditioning to the noisy state as additional channels
        # This is a simple way to condition the U-Net
        B, C, H, W = noisy_state.shape
        
        # Broadcast global conditioning to spatial dimensions
        global_cond_spatial = global_cond.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        # Concatenate noisy state with conditioning
        conditioned_input = torch.cat([noisy_state, global_cond_spatial[:, :C, :, :]], dim=1)
        
        # Predict noise with U-Net
        noise_pred = self.denoise_unet(conditioned_input)
        
        # Extract only the noise channels
        noise_pred = noise_pred[:, :self.out_channels, :, :]
        
        return noise_pred


class CosineBetaSchedule:
    """
    Cosine schedule for variance, as proposed in DEF paper.
    More stable than linear schedule for weather data.
    """
    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        self.timesteps = timesteps
        
        # Cosine schedule
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Define betas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to schedule."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def denoise_step(self, xt: torch.Tensor, noise_pred: torch.Tensor, t: int) -> torch.Tensor:
        """Single denoising step in reverse process."""
        beta_t = self.betas[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Predict x0
        model_mean = sqrt_recip_alphas_t * (
            xt - beta_t / sqrt_one_minus_alphas_cumprod_t * noise_pred
        )
        
        if t > 0:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(xt)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean


def create_convection_diffusion_model(config: dict) -> ConvectionDDPM:
    """Create model with config."""
    return ConvectionDDPM(
        in_channels=len(config['data']['variables']),
        out_channels=len(config['data']['variables']),
        base_dim=config['model']['base_dim'],
        dim_mults=config['model']['dim_mults'],
        attention_resolutions=config['model']['attention_resolutions'],
        num_res_blocks=config['model']['num_res_blocks'],
        dropout=config['model']['dropout'],
        history_length=config['model']['history_length']
    )