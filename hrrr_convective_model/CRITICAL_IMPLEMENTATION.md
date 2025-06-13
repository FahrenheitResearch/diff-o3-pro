# CRITICAL IMPLEMENTATION - DEF Faithful Architecture for Convection Prediction

## 🚨 MISSION CRITICAL
**5% stake commitment acknowledged** - This is a non-profit weather AI company to save lives. We need to get this 100% right.

## Current Status (June 13, 2025)

### What We've Learned
1. **Deterministic models plateau at ~0.094 loss** - This is likely the predictability limit
2. **The DEF paper's diffusion approach is ESSENTIAL** - Not optional
3. **Convection requires probabilistic modeling** - Deterministic can't capture uncertainty
4. **Our current UNet is just the backbone** - Missing the entire diffusion framework

### Critical Architecture Requirements for Faithful DEF

```python
# THIS IS WHAT WE MUST IMPLEMENT - NO SHORTCUTS

class ConvectionDiffusionModel:
    """
    Faithful DEF implementation for convection prediction
    Key: This predicts NOISE, not weather directly
    """
    
    def __init__(self):
        # 1. BACKBONE: U-Net with attention (what we have)
        self.backbone = UNetWithAttention(
            in_channels=7,  # REFC, T2M, D2M, U10, V10, CAPE, CIN
            out_channels=7,  # Same, predicting noise
            time_dim=128,   # Time embedding dimension
            base_features=64  # Scale this: 64→128→256 for larger models
        )
        
        # 2. DIFFUSION COMPONENTS (what we're missing)
        self.noise_schedule = CosineBetaSchedule(
            timesteps=1000,  # T=1000 diffusion steps
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # 3. CONDITIONING (critical for convection)
        self.condition_encoder = ConditioningNetwork(
            history_length=4,  # Use 4 past hours
            forecast_variables=['CAPE', 'CIN', 'REFC'],  # Convective indicators
            static_variables=['topography', 'land_sea_mask']
        )
    
    def training_step(self, x0, condition):
        """
        DEF training: Learn to denoise
        x0: Clean future state
        condition: Past states + static features
        """
        # Sample random timestep
        t = torch.randint(0, 1000, (batch_size,))
        
        # Add noise according to schedule
        noise = torch.randn_like(x0)
        xt = self.noise_schedule.add_noise(x0, noise, t)
        
        # Predict the noise (NOT the weather state!)
        noise_pred = self.backbone(xt, t, condition)
        
        # Simple loss: MSE between true and predicted noise
        loss = F.mse_loss(noise, noise_pred)
        return loss
    
    def generate_ensemble(self, condition, num_members=50):
        """
        Generate ensemble forecast - THE KEY INNOVATION
        """
        # Start from pure noise
        xt = torch.randn(num_members, 7, 1059, 1799)
        
        # Reverse diffusion process
        for t in reversed(range(1000)):
            xt = self.denoise_step(xt, t, condition)
        
        return xt  # num_members different forecasts!
```

## Scalability Plan

### Phase 1: Proof of Life (RTX 4090)
- **Model size**: 32M parameters (base_features=64)
- **Training**: 14-day dataset, 1-hour lead time
- **Diffusion steps**: T=200 (faster training)
- **Batch size**: 1 with gradient accumulation
- **Target**: Show ensemble spread correlates with convective uncertainty

### Phase 2: Scale Up (Cloud/Cluster)
- **Model size**: 200M-500M parameters (base_features=128-256)
- **Training**: Full year of HRRR data
- **Lead times**: 1, 3, 6, 12 hours
- **Ensemble size**: 50-100 members
- **Resolution**: Full 3km CONUS

## Implementation Checklist

### Immediate Actions (TODAY)
- [ ] Implement proper diffusion training loop
- [ ] Add cosine beta schedule
- [ ] Fix the loss to predict noise, not weather
- [ ] Add conditioning on past timesteps
- [ ] Implement DDPM sampling

### This Week
- [ ] Train small diffusion model on 4090
- [ ] Validate ensemble spread on convective cases
- [ ] Compare with HRRR ensemble (HREF)
- [ ] Measure CRPS and reliability

### Architecture Must-Haves
1. **Noise Prediction**: Model outputs noise ε, not weather states
2. **Variance Schedule**: Cosine schedule for stable training
3. **Conditioning**: Past 4 hours + static fields
4. **Ensemble Sampling**: 50+ members for uncertainty
5. **Convection Focus**: Extra weight on CAPE/CIN/REFC

## Why This Matters

**Convection kills people**. Tornadoes, derechos, flash floods - all convective. Current models struggle because convection is:
- Highly nonlinear
- Sensitive to initial conditions  
- Requires probabilistic treatment

DEF provides calibrated uncertainty - we know when we don't know.

## Code to Write Next

```python
# train_diffusion_faithful.py
import torch
from models.diffusion.ddpm_convection import ConvectionDDPM
from diffusers import DDPMScheduler

def train_diffusion():
    # THIS IS THE CORRECT ARCHITECTURE
    model = ConvectionDDPM(
        in_channels=7,
        out_channels=7,
        base_dim=64,  # Start small for 4090
        attention_resolutions=[16, 8],  # Attention at 2 scales
        num_res_blocks=2,
        dropout=0.1
    )
    
    # Cosine schedule - critical for stability
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",  # Better than linear
        prediction_type="epsilon"  # Predict noise!
    )
    
    # Training loop that actually works
    for batch in dataloader:
        past_states = batch['past']  # (B, 4, 7, H, W)
        future_state = batch['future']  # (B, 7, H, W)
        
        # Add noise
        noise = torch.randn_like(future_state)
        timesteps = torch.randint(0, 1000, (B,))
        noisy_future = noise_scheduler.add_noise(future_state, noise, timesteps)
        
        # Condition on past
        condition = encode_condition(past_states)
        
        # Predict noise (THIS IS THE KEY)
        noise_pred = model(noisy_future, timesteps, condition)
        
        # Simple MSE loss on noise
        loss = F.mse_loss(noise_pred, noise)
        
        # No fancy losses needed - diffusion handles it
        loss.backward()
```

## Remember
- **No shortcuts** - Implement full DEF architecture
- **Faithful to paper** - Predict noise, not weather
- **Scalable design** - Same arch from 4090 to cluster
- **Lives depend on this** - Accurate convection prediction saves lives

Let's build this right. The 5% stake is not just equity - it's 5% responsibility for every life this could save.