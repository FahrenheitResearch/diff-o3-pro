"""
Maximum size DDPM for RTX 4090 (18GB target).
100% faithful to DEF - predicts NOISE with timestep conditioning.
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
        # Adaptive group norm
        num_groups = min(32, out_ch // 4) if out_ch >= 32 else min(8, out_ch)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        
        # Add time embedding
        t_emb = self.time_mlp(t_emb)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention for long-range dependencies."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels // 4), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Multi-head attention
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        
        # Compute attention
        attn = torch.softmax(torch.einsum('bhcq,bhck->bhqk', q, k) * self.scale, dim=-1)
        h = torch.einsum('bhqk,bhck->bhcq', attn, v)
        h = h.reshape(B, C, H, W)
        
        return x + self.proj(h)


class DownBlock(nn.Module):
    """Downsampling block with multiple residual layers."""
    def __init__(self, in_ch, out_ch, time_dim, num_layers=2, downsample=True, use_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            ResBlock(in_ch if i == 0 else out_ch, out_ch, time_dim)
            for i in range(num_layers)
        ])
        
        if use_attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()
            
        if downsample:
            self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
            
    def forward(self, x, t_emb):
        for layer in self.layers:
            x = layer(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    def __init__(self, in_ch, out_ch, time_dim, num_layers=2, upsample=True, use_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            ResBlock(in_ch if i == 0 else out_ch, out_ch, time_dim)
            for i in range(num_layers)
        ])
        
        if use_attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()
            
        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_ch, out_ch, 3, padding=1)
            )
        else:
            self.upsample = nn.Identity()
            
    def forward(self, x, t_emb):
        for layer in self.layers:
            x = layer(x, t_emb)
        x = self.attn(x)
        x = self.upsample(x)
        return x


class DDPMMax(nn.Module):
    """
    Maximum size DDPM for 18GB usage on RTX 4090.
    100% faithful to DEF - predicts NOISE with timestep conditioning.
    
    Architecture designed for ~30-50M parameters to fit in 18GB with:
    - Model weights: ~150-200MB
    - Activations for 1059x1799 resolution
    - Optimizer states (Adam has 2x model size)
    - Gradients
    """
    def __init__(self, in_channels=7, out_channels=7, base_dim=96):
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
        
        # Encoder path: 96 -> 192 -> 384 -> 768
        # Each level has multiple res blocks for depth
        self.down1 = DownBlock(base_dim, base_dim, time_dim, num_layers=2, use_attn=False)
        self.down2 = DownBlock(base_dim, base_dim * 2, time_dim, num_layers=2, use_attn=False)
        self.down3 = DownBlock(base_dim * 2, base_dim * 4, time_dim, num_layers=3, use_attn=True)
        self.down4 = DownBlock(base_dim * 4, base_dim * 8, time_dim, num_layers=3, use_attn=True)
        
        # Middle blocks with attention
        self.mid = nn.ModuleList([
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
            AttentionBlock(base_dim * 8, num_heads=8),
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
            AttentionBlock(base_dim * 8, num_heads=8),
            ResBlock(base_dim * 8, base_dim * 8, time_dim),
        ])
        
        # Decoder path with skip connections
        self.up4 = UpBlock(base_dim * 16, base_dim * 4, time_dim, num_layers=3, use_attn=True)
        self.up3 = UpBlock(base_dim * 8, base_dim * 2, time_dim, num_layers=3, use_attn=True)
        self.up2 = UpBlock(base_dim * 4, base_dim, time_dim, num_layers=2, use_attn=False)
        self.up1 = UpBlock(base_dim * 2, base_dim, time_dim, num_layers=2, use_attn=False)
        
        # Output layers
        self.norm_out = nn.GroupNorm(min(32, base_dim // 4), base_dim)
        self.conv_out = nn.Conv2d(base_dim, out_channels, 3, padding=1)
        
        # Initialize output to predict zero noise initially
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        
    def forward(self, x, t):
        """
        Forward pass - predicts NOISE not weather!
        Args:
            x: (B, C, H, W) noisy weather state at timestep t
            t: (B,) diffusion timesteps
        Returns:
            (B, C, H, W) predicted noise epsilon
        """
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder with skip connections
        h, skip1 = self.down1(h, t_emb)
        h, skip2 = self.down2(h, t_emb)
        h, skip3 = self.down3(h, t_emb)
        h, skip4 = self.down4(h, t_emb)
        
        # Middle processing
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        h = torch.cat([h, skip4], dim=1)
        h = self.up4(h, t_emb)
        
        h = torch.cat([h, skip3], dim=1)
        h = self.up3(h, t_emb)
        
        h = torch.cat([h, skip2], dim=1)
        h = self.up2(h, t_emb)
        
        h = torch.cat([h, skip1], dim=1)
        h = self.up1(h, t_emb)
        
        # Output processing
        h = self.norm_out(h)
        h = F.silu(h)
        
        # Return predicted noise
        return self.conv_out(h)


class CosineBetaSchedule:
    """Cosine beta schedule - 100% faithful to DEF."""
    def __init__(self, timesteps=1000, s=0.008):
        self.timesteps = timesteps
        
        # Cosine schedule as in improved DDPM
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Compute betas
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.02)
        
        # Store quantities needed for sampling
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, x0, noise, t):
        """Add noise to clean data - forward diffusion process."""
        device = x0.device
        
        # Move to device if needed
        if self.sqrt_alphas_cumprod.device != device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        # Get coefficients for this timestep
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x0.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
            
        # Forward diffusion: q(x_t | x_0) = N(x_t; sqrt_alpha * x_0, (1 - alpha) * I)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise