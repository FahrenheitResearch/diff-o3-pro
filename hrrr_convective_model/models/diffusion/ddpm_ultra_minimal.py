"""
Ultra-minimal DDPM that handles odd dimensions properly.
100% faithful to DEF - predicts NOISE with timestep conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Timestep embeddings - CRITICAL for diffusion."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ConvBlock(nn.Module):
    """Basic conv block with time conditioning."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h


class UltraMinimalDDPM(nn.Module):
    """
    Ultra-minimal DDPM that works with any input size.
    Faithful to DEF: predicts noise with timestep conditioning.
    """
    def __init__(self, in_channels=7, out_channels=7, base_dim=16):
        super().__init__()
        
        # Time embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Simple encoder-decoder without downsampling
        # This avoids all dimension issues
        self.enc1 = ConvBlock(in_channels, base_dim, time_dim)
        self.enc2 = ConvBlock(base_dim, base_dim * 2, time_dim)
        self.enc3 = ConvBlock(base_dim * 2, base_dim * 2, time_dim)
        
        self.dec3 = ConvBlock(base_dim * 2, base_dim * 2, time_dim)
        self.dec2 = ConvBlock(base_dim * 2, base_dim, time_dim)
        self.dec1 = ConvBlock(base_dim, base_dim, time_dim)
        
        # Output - predicts NOISE
        self.out_conv = nn.Conv2d(base_dim, out_channels, 1)
        
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
        
        # Simple forward pass - no downsampling
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(h1, t_emb)
        h3 = self.enc3(h2, t_emb)
        
        h = self.dec3(h3, t_emb)
        h = self.dec2(h, t_emb)
        h = self.dec1(h, t_emb)
        
        # Output noise
        return self.out_conv(h)


class CosineBetaSchedule:
    """Cosine schedule - faithful to DEF."""
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
        """Add noise for timestep t."""
        device = x0.device
        
        # Move to device
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