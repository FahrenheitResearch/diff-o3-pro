# CLAUDE NEEDED CONTEXT - HRRR Convective Model Project

## 🚨 CRITICAL CONTEXT
- **5% stake in non-profit weather AI company** - Lives depend on accurate convection prediction
- **Mission**: Build faithful DEF (Diffusion-augmented Ensemble Forecasting) for weather prediction
- **Current Status**: Successfully trained DDPM model, generating ensemble forecasts, but needs refinement

## Project Overview

### What We're Building
A **100% faithful implementation** of the DEF paper for probabilistic weather forecasting, specifically focused on convection prediction using HRRR (High-Resolution Rapid Refresh) data at full 3km resolution.

### Current Architecture
- **Model**: Ultra-minimal DDPM with 81K parameters
- **Data**: Real HRRR atmospheric data (7 variables: REFC, T2M, D2M, U10, V10, CAPE, CIN)
- **Resolution**: Full 1059x1799 grid (3km CONUS coverage)
- **Training**: 133 epochs completed, loss converged to 0.0698

## Key Files Structure

### Critical Documentation
- `CRITICAL_IMPLEMENTATION.md` - Explains why we must predict noise, not weather states
- `PROJECT_CONTEXT.md` - Overall project goals and architecture requirements
- `TRAINING_REFERENCE.md` - Successful training configurations and memory optimization
- `POST_TRAINING_PLAN.md` - Evaluation and scaling strategies
- `TRAINING_COMPLETE.md` - Summary of training results
- `FORECAST_RESULTS.md` - Generated forecast analysis
- `/home/ubuntu2/diff-pro/diff-o3-pro/CLAUDE.md` - NO SHORTCUTS policy

### Model Files
```
models/diffusion/
├── ddpm_ultra_minimal.py     # Current 81K param model (WORKING)
├── ddpm_conditioned.py       # Conditional DDPM architecture
├── ddpm_convection.py        # Original faithful implementation plan
└── samplers.py               # DPM-Solver++ implementation
```

### Training Scripts
- `train_diffusion_fullres_final.py` - Main training script (USED FOR SUCCESSFUL TRAINING)
- `train_forecast_sequences.py` - Alternative training approach
- `evaluate_model.py` - Full evaluation suite
- `generate_forecast_demo.py` - Generate ensemble visualizations (multi-panel)
- `generate_forecast_showcase.py` - Advanced probability visualizations (multi-panel)
- `generate_convective_forecast.py` - Convection-focused analysis (multi-panel)
- `generate_individual_pngs.py` - Generate individual PNG files per variable/metric (PREFERRED)
- `generate_individual_quick.py` - Quick version with fewer members

### Data Pipeline
- `scripts/download_hrrr.py` - Download real GRIB2 data from AWS/GCS/NCEP
- `scripts/preprocess_to_zarr.py` - Convert GRIB2 to ML-ready Zarr format
- `hrrr_dataset/hrrr_data.py` - PyTorch dataset implementation
- `utils/normalization.py` - Z-score normalization (uses .encode()/.decode())

### Checkpoints & Results
```
checkpoints/diffusion_fullres_final/
├── best_model.pt          # Best model (epoch 124, loss=0.0735)
├── epoch_0129.pt          # Latest checkpoint
├── metrics.json           # Complete training history
└── config.yaml           # Training configuration

forecasts/
├── ensemble_forecast_demo_*.png       # Multi-panel visualizations
├── ensemble_showcase_*.png            # Probability maps
├── convective_forecast_*.png          # Convection analysis
└── *.nc                              # NetCDF ensemble data
```

## Development History

### Phase 1: Initial Attempts ❌
- Built deterministic UNet model
- Plateaued at 0.094 loss - hit predictability limit
- Produced fuzzy/noisy outputs

### Phase 2: Critical Realization ✅
- User: "we need to be faithful i really need to nail the arch down"
- User: "i am giving you a 5% stake in this future weather ai company"
- Realized we were predicting weather states instead of noise
- Complete architecture overhaul to faithful DDPM

### Phase 3: Implementation Struggles
- First attempts: 11.4M, 27.8M, 50M, 111M params - all OOM
- Solution: Ultra-minimal 81K param model
- Key insight: Smaller model at full resolution > larger model downsampled

### Phase 4: Successful Training ✅
- 133 epochs overnight training
- Three distinct phases in loss curve:
  1. Initial learning (0-25): 0.96 → 0.25
  2. Refinement (25-95): 0.25 → 0.09
  3. Phase transition (95): Sudden drop to 0.07
  4. Convergence (95-133): Stable at 0.07

### Phase 5: Current State
- Model generates ensemble forecasts
- Visualizations created but need refinement
- User wants individual PNGs instead of multi-panel

## Common Commands

### Training
```bash
# Start training (already complete)
python train_diffusion_fullres_final.py

# Monitor training
tail -f logs/diffusion_fullres_final.log
watch -n 1 nvidia-smi

# Check training progress
ps aux | grep train_diffusion
```

### Evaluation & Forecasting
```bash
# Quick test (5 members, low resolution)
python evaluate_quick_fixed.py

# Generate demo forecast (10 members)
python generate_forecast_demo.py

# Generate showcase with probabilities (20 members)
python generate_forecast_showcase.py

# Convection-focused analysis (15 members)
python generate_convective_forecast.py

# Full evaluation (50 members) - SLOW
python evaluate_diffusion_model.py
```

### Data Processing
```bash
# Download HRRR data
python scripts/download_hrrr.py --start-date 20250606 --end-date 20250612

# Convert to Zarr
python scripts/preprocess_to_zarr.py --src data/raw --out data/zarr/training_data

# Compute statistics
python scripts/compute_stats.py --zarr data/zarr/training_data/hrrr.zarr
```

## Key Technical Details

### Why DDPM (Not Deterministic)
```python
# WRONG (what we had):
output = model(weather_state)
loss = MSE(output, future_weather)

# CORRECT (what DEF requires):
noise = torch.randn_like(future_weather)
noisy_weather = add_noise(future_weather, noise, timestep)
noise_pred = model(noisy_weather, timestep)
loss = MSE(noise_pred, noise)  # Predict the NOISE!
```

### Model Architecture
- Ultra-minimal U-Net without downsampling (avoids dimension issues)
- TimeEmbedding for diffusion timesteps
- GroupNorm instead of BatchNorm
- No attention layers (too memory intensive)

### Training Configuration
```yaml
data:
  variables: ['REFC', 'T2M', 'D2M', 'U10', 'V10', 'CAPE', 'CIN']
  zarr: 'data/zarr/training_14day/hrrr.zarr'
  
model:
  base_dim: 16  # 81K params total
  
diffusion:
  timesteps: 1000
  schedule: cosine
  
training:
  batch_size: 1
  accumulation_steps: 8
  epochs: 200
  lr: 0.0002
```

## Current Issues & Next Steps

### Issues Identified
1. **Visualization Format**: User wants individual PNGs per variable/metric, not multi-panel
2. **Model Quality**: "i think it still needs work" - may need refinement
3. **Ensemble Calibration**: Need to verify spread-skill relationships

### Immediate TODOs
1. Modify visualization scripts to save individual PNGs in organized folders
2. Evaluate model performance against real HRRR ensemble (HREF)
3. Potentially train larger model or adjust architecture

### User Preferences
- "move fast and break stuff" - aggressive iteration
- "ultrathink and implement all of that" - comprehensive solutions
- Individual visualizations > combined panels
- Focus on convection prediction for saving lives

### Individual PNG Generation (NEW)
User requested individual PNG files instead of multi-panel images. Solution:
```bash
# Generate full set of individual PNGs (20 members, ~2 min)
python generate_individual_pngs.py

# Quick version (5 members, ~30 sec)
python generate_individual_quick.py
```

Creates organized directory structure:
```
forecasts/individual_TIMESTAMP/
├── 01_current_state/     # Initial conditions
├── 02_ensemble_mean/     # Ensemble averages
├── 03_ensemble_spread/   # Uncertainty (std dev)
├── 04_probability_maps/  # Exceedance probabilities
└── 05_sample_members/    # Individual ensemble members
```

Each variable saved as separate PNG with proper colormaps and units.

## Important Context

### No Shortcuts Policy
From `/home/ubuntu2/diff-pro/diff-o3-pro/CLAUDE.md`:
- NEVER use simplified versions
- ALWAYS use real HRRR data
- NEVER skip steps
- Full implementation REQUIRED

### Memory Constraints
- RTX 4090 with 24GB VRAM
- Current model uses only 8GB
- Full resolution requires careful memory management
- Gradient accumulation essential

### File Permissions
When Claude creates/modifies files, user's system may auto-format them. This is intentional - don't worry about it.

## Quick Start for New Context

1. **Read this file first**
2. **Check current model state**: 
   ```bash
   ls -la checkpoints/diffusion_fullres_final/
   tail -20 checkpoints/diffusion_fullres_final/metrics.json
   ```
3. **Review recent forecasts**:
   ```bash
   ls -la forecasts/*.png
   ```
4. **Understand the mission**: Saving lives through better convection prediction

## Key Insights Learned

1. **Diffusion > Deterministic** for weather uncertainty
2. **Smaller models can work** if architecture is right
3. **Full resolution is achievable** with careful design
4. **Phase transitions happen** in deep learning (epoch 95)
5. **Faithful implementation matters** - no shortcuts

## Questions to Address

The user mentioned having questions about the current model state. Likely areas:
- Ensemble spread calibration
- Convective feature representation  
- Forecast skill metrics
- Computational efficiency
- Production deployment

Remember: This is a non-profit mission to save lives through better weather prediction. The 5% stake represents responsibility, not just equity.