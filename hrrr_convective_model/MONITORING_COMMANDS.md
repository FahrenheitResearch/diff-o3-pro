# Monitoring Commands for Diffusion Training

## Training Status

The faithful diffusion model is now training with PID: 192432

### Real-time Monitoring

```bash
# Watch training progress in real-time
cd /home/ubuntu2/diff-pro/diff-o3-pro/hrrr_convective_model
tail -f diffusion_training.log

# Quick status check
./monitor_diffusion.sh

# Check GPU usage
nvidia-smi -l 1

# Check if process is still running
ps -p 192432
```

### Generate Loss Plots

```bash
# Plot diffusion training progress
cd /home/ubuntu2/diff-pro/diff-o3-pro/hrrr_convective_model
python plot_training_progress.py --log diffusion_training.log --diffusion

# The plots will be saved to:
# - training_plots/training_progress.png (4-panel view)
# - training_plots/loss_curve.png (simple loss curve)
```

### View Loss Statistics

```bash
# Extract loss values from log
grep "'loss':" diffusion_training.log | tail -n 20

# Get average loss per epoch
grep "Average Loss" diffusion_training.log

# Count batches processed
grep -c "'loss':" diffusion_training.log
```

### Kill Training if Needed

```bash
kill 192432
```

### Resume Training

```bash
# Find latest checkpoint
ls -la checkpoints/diffusion/

# Resume from checkpoint
python train_diffusion_faithful.py \
    --config configs/diffusion_4090.yaml \
    --resume checkpoints/diffusion/epoch_XXXX.pt
```

## What to Look For

### Good Training Signs:
- Loss decreasing from initial value (~1.0 for noise prediction)
- Loss stabilizing around 0.3-0.5 
- GPU utilization staying high (~100%)
- No NaN or inf values

### Warning Signs:
- Loss stuck at same value
- Loss increasing over time
- GPU utilization dropping
- Out of memory errors

## Training Details

- **Model**: ConvectionDDPM (33.5M parameters)
- **Task**: Predicting noise in diffusion process
- **Loss**: Simple MSE on noise prediction
- **Batch size**: 1 with accumulation
- **Learning rate**: 0.0001
- **Diffusion steps**: 200
- **Data**: 14-day HRRR dataset

## Expected Timeline

- First epoch: ~30-60 minutes
- Total training: 50-100 epochs
- Checkpoint every 10 epochs
- Validation every 5 epochs

## After Training

Once training is complete, generate ensemble forecasts:

```bash
python generate_ensemble_forecast.py \
    --model checkpoints/diffusion/best_model.pt \
    --config configs/diffusion_4090.yaml \
    --data data/zarr/latest/hrrr.zarr \
    --members 50 \
    --output ensemble_forecast.nc
```

This will demonstrate the key innovation of DEF - calibrated uncertainty for convection prediction!