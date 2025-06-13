"""Ultra-minimal U-Net for memory-constrained training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single conv block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )
        
    def forward(self, x):
        return self.conv(x)


class UNetMinimal(nn.Module):
    """Ultra-minimal U-Net for faithful diffusion with low memory."""
    
    def __init__(self, in_channels=7, out_channels=7, base_dim=16):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_dim)
        self.down1 = nn.Conv2d(base_dim, base_dim, 2, stride=2)
        
        self.enc2 = ConvBlock(base_dim, base_dim * 2)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 2, 2, stride=2)
        
        # Middle
        self.middle = ConvBlock(base_dim * 2, base_dim * 2)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_dim * 4, base_dim * 2)
        
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.dec1 = ConvBlock(base_dim * 2, base_dim)
        
        # Output
        self.out_conv = nn.Conv2d(base_dim, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        # Middle
        m = self.middle(d2)
        
        # Decoder
        u2 = self.up2(m)
        u2 = torch.cat([u2, e2], dim=1)
        d2_out = self.dec2(u2)
        
        u1 = self.up1(d2_out)
        u1 = torch.cat([u1, e1], dim=1)
        d1_out = self.dec1(u1)
        
        # Output
        out = self.out_conv(d1_out)
        
        return out