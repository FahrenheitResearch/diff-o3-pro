"""
Minimal faithful DDPM for RTX 4090 - 100% faithful to DEF paper.
Predicts NOISE, not weather states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings - CRITICAL for diffusion."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """t: (B,) tensor of timesteps"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add timestep info
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.skip(x)


class MinimalDDPM(nn.Module):
    """
    Minimal but FAITHFUL diffusion model for convection.
    Key: Predicts NOISE, includes timestep embedding.
    """
    def __init__(self, in_channels=7, out_channels=7, base_dim=16):
        super().__init__()
        
        # Time embedding - CRITICAL
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.down1 = nn.Sequential(
            ResBlock(base_dim, base_dim, time_dim),
            ResBlock(base_dim, base_dim, time_dim),
            nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1)
        )
        
        self.down2 = nn.Sequential(
            ResBlock(base_dim, base_dim * 2, time_dim),
            ResBlock(base_dim * 2, base_dim * 2, time_dim),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, stride=2, padding=1)
        )
        
        # Middle
        self.mid = nn.Sequential(
            ResBlock(base_dim * 2, base_dim * 2, time_dim),
            ResBlock(base_dim * 2, base_dim * 2, time_dim)
        )
        
        # Decoder - using Conv2d with Upsample to avoid size issues
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1),
            ResBlock(base_dim * 4, base_dim * 2, time_dim),
            ResBlock(base_dim * 2, base_dim * 2, time_dim)
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),
            ResBlock(base_dim * 2, base_dim, time_dim),
            ResBlock(base_dim, base_dim, time_dim)
        )
        
        # Output - predicts NOISE
        self.conv_out = nn.Conv2d(base_dim, out_channels, 3, padding=1)
        
    def forward(self, x, t):
        """
        Args:
            x: (B, C, H, W) noisy weather state
            t: (B,) diffusion timesteps
        Returns:
            (B, C, H, W) predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Encoder with skip connections
        h1 = self.conv_in(x)
        h2 = self.down1[0](h1, t_emb)
        h2 = self.down1[1](h2, t_emb)
        h2_down = self.down1[2](h2)
        
        h3 = self.down2[0](h2_down, t_emb)
        h3 = self.down2[1](h3, t_emb)
        h3_down = self.down2[2](h3)
        
        # Middle
        h = self.mid[0](h3_down, t_emb)
        h = self.mid[1](h, t_emb)
        
        # Decoder with skip connections
        h = self.up2[0](h)  # Upsample
        h = self.up2[1](h)  # Conv
        h = torch.cat([h, h3], dim=1)
        h = self.up2[2](h, t_emb)
        h = self.up2[3](h, t_emb)
        
        h = self.up1[0](h)  # Upsample
        h = self.up1[1](h)  # Conv
        h = torch.cat([h, h2], dim=1)
        h = self.up1[2](h, t_emb)
        h = self.up1[3](h, t_emb)
        
        # Output noise
        return self.conv_out(h)


class CosineBetaSchedule:
    """Cosine beta schedule - more stable than linear."""
    def __init__(self, timesteps=100, s=0.008):
        self.timesteps = timesteps
        
        # Cosine schedule
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Compute betas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.02)
        
        # Store what we need
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, x0, noise, t):
        """Add noise to data for timestep t."""
        device = x0.device
        
        # Move to device if needed
        if self.sqrt_alphas_cumprod.device != device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        # Get coefficients
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
            
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise