#!/bin/bash
# Download HRRR training data with forecast hours F00-F18

# Create data directory
mkdir -p data/raw/training_forecast

# Download 7 days of data with F00-F18 for each cycle
# Using 4 cycles per day (00Z, 06Z, 12Z, 18Z)
python scripts/download_hrrr.py \
    --start-date 20250605 \
    --end-date 20250611 \
    --hours 0 6 12 18 \
    --forecast-hours "0-18" \
    --output-dir data/raw/training_forecast \
    --parallel 4

echo "Download complete! Now preprocess with:"
echo "python scripts/preprocess_hrrr_forecast.py --src data/raw/training_forecast --out data/zarr/training_forecast"