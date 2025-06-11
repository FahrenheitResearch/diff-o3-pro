# HRRR Convection Surrogate Model (3 km)

A deep learning model for high-resolution weather prediction using HRRR (High-Resolution Rapid Refresh) data at native 3km resolution. This implementation uses an Attention U-Net architecture to predict future atmospheric states from current conditions.

## Features

- **Full 3km CONUS Resolution**: No downsampling - trains on full resolution HRRR data
- **Real Data Pipeline**: Downloads and processes actual HRRR GRIB2 files from multiple sources
- **Attention U-Net Architecture**: State-of-the-art architecture with attention mechanisms
- **Efficient Zarr Storage**: Converts GRIB2 to Zarr for fast, parallel data access
- **Production Ready**: Includes checkpointing, metrics, and distributed training support

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate hrrr_convective_env
```

### 2. Download REAL HRRR Data

Download actual HRRR data from AWS, Google Cloud, or NCEP:

```bash
# Download one week of data (4 cycles per day)
python scripts/download_hrrr.py --start-date 20250101 --end-date 20250107 --hours 0 6 12 18

# Or use the shell script
./scripts/download_hrrr.sh 20250101 20250107
```

### 3. Convert GRIB2 to Zarr

Convert downloaded GRIB2 files to efficient Zarr format:

```bash
python scripts/preprocess_to_zarr.py --src data/raw --out data/zarr/conus_202501
```

### 4. Compute Statistics

Calculate mean and standard deviation for normalization:

```bash
python scripts/compute_stats.py --zarr data/zarr/conus_202501/hrrr.zarr --out data/stats.json
```

### 5. Train the Model

```bash
python train.py
```

## Data Pipeline Details

### HRRR Variables Used

| Variable | Description | Level | Units |
|----------|-------------|-------|-------|
| REFC | Composite reflectivity | Entire column | dBZ |
| REFD | Reflectivity at 1 km AGL | 1000 m AGL | dBZ |
| CAPE | Convective Available Potential Energy | sfc‑400 hPa | J kg⁻¹ |
| CIN | Convective Inhibition | sfc‑400 hPa | J kg⁻¹ |
| ACPCP | Convective precipitation (1 h) | sfc | mm |
| TMP | 2‑m temperature | 2 m | K |
| DPT | 2‑m dew‑point | 2 m | K |
| UGRD/VGRD | 10‑m winds | 10 m | m s⁻¹ |

### Data Sources

The download scripts automatically try multiple sources in order:
1. **AWS S3**: `s3://noaa-hrrr-bdp-pds/` (fastest, most reliable)
2. **Google Cloud Storage**: Public HRRR archive
3. **NCEP NOMADS**: Operational data (most recent ~2 days)

### Storage Requirements

- Raw GRIB2: ~230 MB per analysis file
- Zarr format: ~3 TB per year at 3km resolution
- Ensure adequate disk space before downloading large datasets

## Model Architecture

The model uses an Attention U-Net with:
- Encoder: 3 downsampling blocks with max pooling
- Bridge: Bottleneck layer with 8x initial filters
- Decoder: 3 upsampling blocks with attention gates
- Skip connections with attention mechanisms
- BatchNorm and ReLU activations

## Training Configuration

Edit `configs/default.yaml` to customize:

```yaml
data:
  zarr: "data/zarr/conus_202501/hrrr.zarr"
  stats: "data/stats.json"
  variables: ["REFC","REFD","CAPE","CIN","ACPCP","TMP","DPT","UGRD","VGRD"]
training:
  lead_hours: 1        # Predict 1 hour ahead
  batch_size: 2        # Adjust based on GPU memory
  num_workers: 4
  epochs: 20
  lr: 1.0e-4
```

## Performance

- Training time: ~1.2 days/epoch on single A100-80GB for full CONUS
- Memory usage: ~400 MB per sample at 3km resolution
- Supports multi-GPU training with PyTorch DDP

## Automation

To continuously update with new HRRR data:

```bash
# Cron job example (every 6 hours)
0 */6 * * * /path/to/scripts/download_hrrr.py --start-date $(date +\%Y\%m\%d) --hours $(date +\%H)
```

## Directory Structure

```
hrrr_convective_model/
├── configs/           # Configuration files
├── data/             # Data directory
│   ├── raw/         # Downloaded GRIB2 files
│   ├── zarr/        # Converted Zarr datasets
│   └── stats.json   # Normalization statistics
├── models/           # Model architectures
├── hrrr_dataset/     # PyTorch dataset implementation
├── scripts/          # Data processing scripts
├── utils/            # Utility functions
└── train.py          # Main training script
```

## Troubleshooting

### Download Issues
- Ensure you have `boto3` installed for AWS downloads
- Check internet connectivity and firewall settings
- Try different hours/dates if specific files are missing

### Memory Issues
- Reduce `batch_size` in config
- Use gradient accumulation for larger effective batch sizes
- Enable mixed precision training (add to train.py if needed)

### GRIB2 Errors
- Ensure `eccodes` is properly installed via conda
- Check GRIB2 file integrity with `grib_ls` command
- Some forecast hours may have different variable availability

## Citation

If you use this code, please cite:
- HRRR Model: Benjamin et al. (2016)
- Architecture inspired by Attention U-Net: Oktay et al. (2018)

## License

This project is released under the MIT License. HRRR data is provided by NOAA and is in the public domain.