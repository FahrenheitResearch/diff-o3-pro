# Architecture Comparison: Current Implementation vs DEF Paper

## Training Progress Update
The light training shows better dynamics than before:
- Loss decreased from ~0.25 to ~0.094 and is staying around 0.09-0.10
- Still somewhat horizontal after initial drop, but with more variation
- Using aggressive LR=0.005 with smaller model (8M params)

## Architecture Comparison with DEF Paper

### Current Implementation (Deterministic Model)

**Model: UNetResidual**
- **Type**: Deterministic U-Net with residual connection
- **Parameters**: 8M (small) to 32M (large)
- **Key Features**:
  - Residual connection: `output = input + scale * model(input)`
  - Temporal encoding with sinusoidal embeddings
  - Attention gates at each resolution level
  - Predicts changes (deltas) not absolute values

**Training**:
- Single-step prediction (1 hour ahead)
- Mixed loss: L1 (40%) + Spectral (30%) + MSE (20%) + Gradient (10%)
- Aggressive learning rates (0.005-0.01) to escape plateaus
- Noise injection for regularization

### DEF Paper Architecture

**Model: Diffusion-based Ensemble**
- **Type**: Denoising Diffusion Probabilistic Model (DDPM)
- **Key Differences**:
  1. **Probabilistic vs Deterministic**: DEF uses diffusion to generate ensemble members
  2. **Multi-step diffusion process**: Adds noise then denoises over T steps
  3. **Ensemble generation**: Can sample multiple forecasts from same initial condition
  4. **Uncertainty quantification**: Natural uncertainty from probabilistic sampling

**DEF Components We're Missing**:
1. **Diffusion Process**:
   - Forward process: x_t = √(α_t) * x_0 + √(1-α_t) * ε
   - Reverse process: Learn to predict noise ε
   - Variance schedule (linear or cosine)

2. **Ensemble Sampling**:
   - Multiple samples from same initial state
   - Spread calculation for uncertainty

3. **Conditioning Mechanism**:
   - DEF conditions on multiple past timesteps
   - We only use current timestep

4. **Loss Function**:
   - DEF uses simplified ELBO: E[||ε - ε_θ(x_t, t)||²]
   - We use weather-specific losses

## Current Status

### What We Have:
✓ Deterministic baseline model (working but plateauing)
✓ Residual connections for stability
✓ Temporal encoding
✓ Weather-specific losses
✓ Data pipeline at full 3km resolution

### What We Need for Full DEF:
1. Implement DDPM architecture (`models/diffusion/ddpm_conditioned.py` exists but unused)
2. Add diffusion training loop
3. Implement ensemble sampling
4. Add multi-timestep conditioning
5. Integrate uncertainty quantification

## Recommendations

1. **Current Deterministic Model**: The plateau at ~0.094 loss might be acceptable as a baseline. This could be the inherent predictability limit for 1-hour forecasts with this architecture.

2. **Next Steps for DEF**:
   - Use current deterministic model as initialization/baseline
   - Implement diffusion training on top
   - Start with small diffusion steps (T=100-200)
   - Use deterministic predictions as x_0 in diffusion process

3. **Architecture Improvements**:
   - Add skip connections between encoder/decoder at same resolution
   - Try larger models if memory permits
   - Implement multi-scale losses
   - Add physical constraints (conservation laws)

The current implementation is a solid foundation but lacks the key innovation of DEF: the diffusion-based ensemble generation that provides calibrated uncertainty estimates.