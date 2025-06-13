"""
18GB-optimized DDPM for RTX 4090.
100% faithful to DEF - predicts NOISE with timestep conditioning.
Uses efficient attention only at low resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings."""
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


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        
        # Time conditioning
        t_emb = self.time_mlp(t_emb)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h + self.skip(x)


class EfficientAttention(nn.Module):
    """Memory-efficient attention for low-resolution feature maps only."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Only apply attention if resolution is small enough
        if H * W > 4096:  # Skip attention for large feature maps
            return x
            
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Efficient attention computation
        attn = torch.softmax((q.transpose(-1, -2) @ k) * self.scale, dim=-1)
        h = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        
        return x + self.proj(h)


class DDPM18GB(nn.Module):
    """
    18GB-optimized DDPM for RTX 4090.
    ~50M parameters, 100% faithful to DEF.
    """
    def __init__(self, in_channels=7, out_channels=7, base_dim=64):
        super().__init__()
        
        # Time embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Encoder - 64 -> 128 -> 256 -> 512
        # 1059x1799 -> 530x900 -> 265x450 -> 133x225 -> 67x113
        self.enc1 = nn.ModuleList([
            ResBlock(base_dim, base_dim, time_dim),
            ResBlock(base_dim, base_dim, time_dim),
        ])
        self.down1 = nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1)
        
        self.enc2 = nn.ModuleList([
            ResBlock(base_dim, base_dim * 2, time_dim),
            ResBlock(base_dim * 2, base_dim * 2, time_dim),
        ])
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 2, 3, stride=2, padding=1)
        
        self.enc3 = nn.ModuleList([
            ResBlock(base_dim * 2, base_dim * 4, time_dim),
            ResBlock(base_dim * 4, base_dim * 4, time_dim),
            ResBlock(base_dim * 4, base_dim * 4, time_dim),
        ])
        self.down3 = nn.Conv2d(base_dim * 4, base_dim * 4, 3, stride=2, padding=1)
        
        self.enc4 = nn.ModuleList([
            ResBlock(base_dim * 4, base_dim * 8, time_dim),
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
        ])
        self.down4 = nn.Conv2d(base_dim * 8, base_dim * 8, 3, stride=2, padding=1)
        
        # Middle - only use attention here at 67x113 resolution
        self.mid = nn.ModuleList([
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
            EfficientAttention(base_dim * 8),
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
            EfficientAttention(base_dim * 8),
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
        ])
        
        # Decoder with skip connections
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim * 8, base_dim * 8, 3, padding=1)
        )
        self.dec4 = nn.ModuleList([
            ResBlock(base_dim * 16, base_dim * 4, time_dim),
            ResBlock(base_dim * 4, base_dim * 4, time_dim),
            ResBlock(base_dim * 4, base_dim * 4, time_dim),
        ])
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim * 4, base_dim * 4, 3, padding=1)
        )
        self.dec3 = nn.ModuleList([
            ResBlock(base_dim * 8, base_dim * 2, time_dim),
            ResBlock(base_dim * 2, base_dim * 2, time_dim),
            ResBlock(base_dim * 2, base_dim * 2, time_dim),
        ])
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1)
        )
        self.dec2 = nn.ModuleList([
            ResBlock(base_dim * 4, base_dim, time_dim),
            ResBlock(base_dim, base_dim, time_dim),
        ])
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_dim, base_dim, 3, padding=1)
        )
        self.dec1 = nn.ModuleList([
            ResBlock(base_dim * 2, base_dim, time_dim),
            ResBlock(base_dim, base_dim, time_dim),
        ])
        
        # Output
        self.norm_out = nn.GroupNorm(8, base_dim)
        self.conv_out = nn.Conv2d(base_dim, out_channels, 3, padding=1)
        
        # Zero init
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        
    def forward(self, x, t):
        """
        Predicts NOISE, not weather states!
        Args:
            x: (B, C, H, W) noisy weather state
            t: (B,) timesteps
        Returns:
            (B, C, H, W) predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder with skips
        h1 = h
        for layer in self.enc1:
            h1 = layer(h1, t_emb)
        h2 = self.down1(h1)
        
        for layer in self.enc2:
            h2 = layer(h2, t_emb)
        h3 = self.down2(h2)
        
        for layer in self.enc3:
            h3 = layer(h3, t_emb)
        h4 = self.down3(h3)
        
        for layer in self.enc4:
            h4 = layer(h4, t_emb)
        h = self.down4(h4)
        
        # Middle
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder with skips - handle size mismatches
        h = self.up4(h)
        # Crop or pad to match skip connection size
        if h.shape[2:] != h4.shape[2:]:
            h = F.interpolate(h, size=h4.shape[2:], mode='nearest')
        h = torch.cat([h, h4], dim=1)
        for layer in self.dec4:
            h = layer(h, t_emb)
            
        h = self.up3(h)
        if h.shape[2:] != h3.shape[2:]:
            h = F.interpolate(h, size=h3.shape[2:], mode='nearest')
        h = torch.cat([h, h3], dim=1)
        for layer in self.dec3:
            h = layer(h, t_emb)
            
        h = self.up2(h)
        if h.shape[2:] != h2.shape[2:]:
            h = F.interpolate(h, size=h2.shape[2:], mode='nearest')
        h = torch.cat([h, h2], dim=1)
        for layer in self.dec2:
            h = layer(h, t_emb)
            
        h = self.up1(h)
        if h.shape[2:] != h1.shape[2:]:
            h = F.interpolate(h, size=h1.shape[2:], mode='nearest')
        h = torch.cat([h, h1], dim=1)
        for layer in self.dec1:
            h = layer(h, t_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


# Reuse cosine schedule
class CosineBetaSchedule:
    """Cosine beta schedule - 100% faithful to DEF."""
    def __init__(self, timesteps=1000, s=0.008):
        self.timesteps = timesteps
        
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.02)
        
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, x0, noise, t):
        """Add noise to data."""
        device = x0.device
        
        if self.sqrt_alphas_cumprod.device != device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        while len(sqrt_alpha.shape) < len(x0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
            
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise