import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class AttentionBlock(nn.Module):
    """Attention mechanism for U-Net skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_c, out_c),
            ConvBNReLU(out_c, out_c)
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_c + skip_c, out_c),
            ConvBNReLU(out_c, out_c)
        )
    
    def forward(self, x, skip, attn):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TemporalEncoding(nn.Module):
    """Fixed temporal encoding that uses ACTUAL timestamps"""
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        # Hour of day (0-23) and day of year (0-365)
        self.time_proj = nn.Linear(4, d_model)  # sin/cos for hour and day
        
    def forward(self, x, timestamps):
        """
        x: [B, C, H, W] input features
        timestamps: [B] hours since epoch (real timestamps!)
        """
        B, C, H, W = x.shape
        
        # Convert hours since epoch to hour-of-day and day-of-year
        hours_in_day = 24
        hours_in_year = 24 * 365
        
        # Extract hour of day and day of year
        hour_of_day = (timestamps % hours_in_day) / hours_in_day  # Normalize to [0, 1]
        day_of_year = ((timestamps % hours_in_year) / hours_in_day) / 365  # Normalize to [0, 1]
        
        # Create sinusoidal features
        time_features = torch.stack([
            torch.sin(2 * np.pi * hour_of_day),
            torch.cos(2 * np.pi * hour_of_day),
            torch.sin(2 * np.pi * day_of_year),
            torch.cos(2 * np.pi * day_of_year)
        ], dim=1).float()  # [B, 4]
        
        # Project to d_model dimensions
        time_emb = self.time_proj(time_features)  # [B, d_model]
        
        # Reshape and expand to spatial dimensions
        time_emb = time_emb.view(B, self.d_model, 1, 1)
        time_emb = time_emb.expand(B, self.d_model, H, W)
        
        return time_emb

class UNetResidual(nn.Module):
    """U-Net with residual connection for weather prediction.
    
    CRITICAL: This model predicts CHANGES (deltas) not absolute values!
    output = input + model(input)
    """
    def __init__(self, in_ch, out_ch, base_features=64, use_temporal_encoding=True):
        super().__init__()
        assert in_ch == out_ch, "Residual UNet requires same input/output channels"
        
        self.use_temporal_encoding = use_temporal_encoding
        nf = base_features
        
        # Temporal encoding
        if use_temporal_encoding:
            self.temporal_encoder = TemporalEncoding(d_model=nf)
            in_ch_adjusted = in_ch + nf
        else:
            in_ch_adjusted = in_ch
        
        # Encoder
        self.inc = nn.Sequential(ConvBNReLU(in_ch_adjusted, nf), ConvBNReLU(nf, nf))
        self.down1 = DownBlock(nf, nf * 2)
        self.down2 = DownBlock(nf * 2, nf * 4)
        self.down3 = DownBlock(nf * 4, nf * 8)
        
        # Bridge
        self.bridge = nn.Sequential(
            ConvBNReLU(nf * 8, nf * 16),
            ConvBNReLU(nf * 16, nf * 16)
        )
        
        # Attention blocks
        self.a3 = AttentionBlock(F_g=nf * 16, F_l=nf * 8, F_int=nf * 4)
        self.a2 = AttentionBlock(F_g=nf * 8, F_l=nf * 4, F_int=nf * 2)
        self.a1 = AttentionBlock(F_g=nf * 4, F_l=nf * 2, F_int=nf)
        self.a0 = AttentionBlock(F_g=nf * 2, F_l=nf, F_int=nf // 2)
        
        # Decoder
        self.up3 = UpBlock(nf * 16, nf * 8, nf * 8)
        self.up2 = UpBlock(nf * 8, nf * 4, nf * 4)
        self.up1 = UpBlock(nf * 4, nf * 2, nf * 2)
        self.up0 = UpBlock(nf * 2, nf, nf)
        
        # Output - predict residuals with small random initialization
        self.outc = nn.Conv2d(nf, out_ch, 1)
        # Use small random initialization instead of zeros
        nn.init.normal_(self.outc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.outc.bias, 0.0)
        
        # Scaling factor for residuals (learnable, start much larger)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, timestamps=None, add_noise=False, noise_std=0.01):
        # Store input for residual connection
        identity = x
        
        # Add noise during training for symmetry breaking
        if add_noise and self.training:
            x = x + torch.randn_like(x) * noise_std
        
        # Add temporal encoding if enabled
        if self.use_temporal_encoding and timestamps is not None:
            temporal_features = self.temporal_encoder(x, timestamps)
            x = torch.cat([x, temporal_features], dim=1)
        
        # Encoder
        x0 = self.inc(x)
        x1, skip1 = self.down1(x0)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        # Bridge
        bridge = self.bridge(x3)
        
        # Decoder with attention
        up3 = self.up3(bridge, skip3, self.a3)
        up2 = self.up2(up3, skip2, self.a2)
        up1 = self.up1(up2, skip1, self.a1)
        up0 = self.up0(up1, x0, self.a0)
        
        # Output residuals
        residuals = self.outc(up0)
        
        # Scale residuals and add to input
        # This is the KEY change - we predict deltas!
        output = identity + self.residual_scale * residuals
        
        return output

class UNetResidualSmall(UNetResidual):
    """Smaller version for limited GPU memory"""
    def __init__(self, in_ch, out_ch, use_temporal_encoding=True):
        super().__init__(in_ch, out_ch, base_features=32, use_temporal_encoding=use_temporal_encoding)