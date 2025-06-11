#!/bin/bash
# Download REAL HRRR GRIB2 files from multiple sources
# NO SAMPLE DATA - REAL ATMOSPHERIC DATA ONLY

set -e

# Configuration
DATE_START=${1:-$(date -d "yesterday" +%Y%m%d)}
DATE_END=${2:-$DATE_START}
HOURS=${3:-"00 06 12 18"}  # UTC hours to download
OUTPUT_DIR=${4:-"data/raw"}

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Downloading HRRR data from $DATE_START to $DATE_END"

# Function to download from AWS
download_aws() {
    local date=$1
    local hour=$2
    local year=${date:0:4}
    local month=${date:4:2}
    local day=${date:6:2}
    
    # AWS S3 path
    local s3_path="s3://noaa-hrrr-bdp-pds/hrrr.${year}${month}${day}/conus/hrrr.t${hour}z.wrfprsf00.grib2"
    local output_file="$OUTPUT_DIR/hrrr.${year}${month}${day}.t${hour}z.wrfprsf00.grib2"
    
    echo "Downloading from AWS: $s3_path"
    aws s3 cp --no-sign-request $s3_path $output_file || return 1
}

# Function to download from Google Cloud
download_gcs() {
    local date=$1
    local hour=$2
    local year=${date:0:4}
    local month=${date:4:2}
    local day=${date:6:2}
    
    # Google Cloud Storage path
    local gcs_url="https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.${year}${month}${day}/conus/hrrr.t${hour}z.wrfprsf00.grib2"
    local output_file="$OUTPUT_DIR/hrrr.${year}${month}${day}.t${hour}z.wrfprsf00.grib2"
    
    echo "Downloading from GCS: $gcs_url"
    wget -q -O $output_file $gcs_url || return 1
}

# Function to download from NCEP
download_ncep() {
    local date=$1
    local hour=$2
    local year=${date:0:4}
    local month=${date:4:2}
    local day=${date:6:2}
    
    # NCEP URL
    local ncep_url="https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.${year}${month}${day}/conus/hrrr.t${hour}z.wrfprsf00.grib2"
    local output_file="$OUTPUT_DIR/hrrr.${year}${month}${day}.t${hour}z.wrfprsf00.grib2"
    
    echo "Downloading from NCEP: $ncep_url"
    wget -q -O $output_file $ncep_url || return 1
}

# Main download loop
current_date=$DATE_START
while [ "$current_date" -le "$DATE_END" ]; do
    year=${current_date:0:4}
    month=${current_date:4:2}
    day=${current_date:6:2}
    
    for hour in $HOURS; do
        output_file="$OUTPUT_DIR/hrrr.${year}${month}${day}.t${hour}z.wrfprsf00.grib2"
        
        # Skip if file already exists
        if [ -f "$output_file" ]; then
            echo "File already exists: $output_file"
            continue
        fi
        
        # Try AWS first
        if download_aws $current_date $hour; then
            echo "Successfully downloaded from AWS"
        # Try Google Cloud
        elif download_gcs $current_date $hour; then
            echo "Successfully downloaded from GCS"
        # Try NCEP
        elif download_ncep $current_date $hour; then
            echo "Successfully downloaded from NCEP"
        else
            echo "Failed to download hrrr.${year}${month}${day}.t${hour}z.wrfprsf00.grib2"
        fi
    done
    
    # Move to next day
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Download complete. Files saved to $OUTPUT_DIR"

# Download forecast files if requested
if [ "${DOWNLOAD_FORECASTS:-0}" = "1" ]; then
    echo "Downloading forecast files..."
    # Add forecast download logic here for f01-f18
fi