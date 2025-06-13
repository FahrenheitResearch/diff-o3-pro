import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    """Spectral loss in Fourier domain to preserve high-frequency details."""
    def __init__(self, weight_high_freq=2.0):
        super().__init__()
        self.weight_high_freq = weight_high_freq
    
    def forward(self, pred, target):
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Create frequency weighting (emphasize high frequencies)
        B, C, H, W = pred_mag.shape
        # Create radial frequency grid
        fy = torch.fft.fftfreq(pred.shape[-2], device=pred.device).view(-1, 1)
        fx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).view(1, -1)
        freq_grid = torch.sqrt(fy**2 + fx**2)
        
        # Weight high frequencies more
        freq_weight = 1.0 + (self.weight_high_freq - 1.0) * freq_grid
        freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Weighted spectral loss
        spectral_diff = freq_weight * (pred_mag - target_mag).abs()
        loss = spectral_diff.mean()
        
        return loss

class GradientLoss(nn.Module):
    """Loss on spatial gradients to preserve edges."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Sobel filters for gradients
        def gradient(x):
            # Spatial gradients
            dx = x[..., :, 1:] - x[..., :, :-1]
            dy = x[..., 1:, :] - x[..., :-1, :]
            return dx, dy
        
        pred_dx, pred_dy = gradient(pred)
        target_dx, target_dy = gradient(target)
        
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        
        return loss_dx + loss_dy

class WeatherLoss(nn.Module):
    """Combined loss for weather prediction.
    
    Combines:
    - L1 loss for sharp features
    - Spectral loss for high-frequency preservation  
    - MSE loss for overall structure
    - Gradient loss for edges
    """
    def __init__(self, l1_weight=0.4, spectral_weight=0.3, mse_weight=0.2, gradient_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralLoss(weight_high_freq=2.0)
        self.gradient_loss = GradientLoss()
        
    def forward(self, pred, target):
        # Individual losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        spectral = self.spectral_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        
        # Combined loss
        total_loss = (self.l1_weight * l1 + 
                     self.spectral_weight * spectral + 
                     self.mse_weight * mse +
                     self.gradient_weight * gradient)
        
        # Return total and components for logging
        return total_loss, {
            'l1': l1.item(),
            'mse': mse.item(), 
            'spectral': spectral.item(),
            'gradient': gradient.item(),
            'total': total_loss.item()
        }

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG features (optional, requires pretrained VGG)."""
    def __init__(self):
        super().__init__()
        # For now, just use pixel loss
        # In production, would load VGG and extract features
        pass
    
    def forward(self, pred, target):
        # Placeholder - would compute VGG feature differences
        return F.l1_loss(pred, target)