#!/usr/bin/env python3
"""
100% FAITHFUL Conditional DDPM for DEF (Diffusion-augmented Ensemble Forecasting).
This model takes BOTH the noisy future state AND current conditions as input.
CRITICAL: Without conditioning on current state, we can't forecast!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Timestep embeddings for diffusion."""
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


class ConditionalConvBlock(nn.Module):
    """Conv block with time conditioning - handles concatenated inputs."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # Ensure num_groups divides out_ch
        num_groups = min(8, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h


class ConditionalDDPM(nn.Module):
    """
    FAITHFUL Conditional DDPM for weather forecasting.
    Takes both noisy future state AND current conditions.
    This is what DEF actually requires!
    """
    def __init__(self, channels=7, base_dim=24):
        super().__init__()
        
        # Time embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial conv to handle concatenated input
        # Input: [noisy_future (7ch) + current_state (7ch)] = 14 channels
        self.init_conv = nn.Conv2d(channels * 2, base_dim, 3, padding=1)
        
        # Simple U-Net without downsampling (to handle odd dimensions)
        # Encoder
        self.enc1 = ConditionalConvBlock(base_dim, base_dim, time_dim)
        self.enc2 = ConditionalConvBlock(base_dim, base_dim * 2, time_dim)
        self.enc3 = ConditionalConvBlock(base_dim * 2, base_dim * 3, time_dim)
        
        # Middle
        self.mid1 = ConditionalConvBlock(base_dim * 3, base_dim * 3, time_dim)
        self.mid2 = ConditionalConvBlock(base_dim * 3, base_dim * 3, time_dim)
        
        # Decoder with skip connections
        self.dec3 = ConditionalConvBlock(base_dim * 6, base_dim * 3, time_dim)  # Skip from enc3
        self.dec2 = ConditionalConvBlock(base_dim * 5, base_dim * 2, time_dim)  # Skip from enc2
        self.dec1 = ConditionalConvBlock(base_dim * 3, base_dim, time_dim)      # Skip from enc1
        
        # Output - predicts NOISE (not the denoised state!)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_dim, channels, 1)  # Output same channels as weather vars
        )
        
        print(f"Conditional DDPM initialized:")
        print(f"  - Input: {channels * 2} channels (noisy + conditions)")
        print(f"  - Base dim: {base_dim}")
        print(f"  - Parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, noisy_state, timestep, condition):
        """
        100% FAITHFUL forward pass.
        
        Args:
            noisy_state: (B, C, H, W) - the noisy future state
            timestep: (B,) - diffusion timestep
            condition: (B, C, H, W) - the CURRENT atmospheric state
            
        Returns:
            (B, C, H, W) - predicted NOISE (not the clean state!)
        """
        # CRITICAL: Concatenate noisy state with current conditions
        # This is what makes it a conditional model!
        x = torch.cat([noisy_state, condition], dim=1)
        
        # Time embedding
        t_emb = self.time_mlp(timestep.float())
        
        # Initial conv
        x = self.init_conv(x)
        
        # U-Net forward with skips
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(h1, t_emb)
        h3 = self.enc3(h2, t_emb)
        
        h = self.mid1(h3, t_emb)
        h = self.mid2(h, t_emb)
        
        # Decoder with skip connections
        h = torch.cat([h, h3], dim=1)
        h = self.dec3(h, t_emb)
        
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, t_emb)
        
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)
        
        # Predict noise
        return self.out_conv(h)


class CosineBetaSchedule:
    """Cosine schedule as used in DEF paper."""
    def __init__(self, timesteps=1000, s=0.008):
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
        """Add noise to clean data for timestep t."""
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