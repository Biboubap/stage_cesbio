
#Example for Belcher_10m

# Create directories for median calculation
mkdir -p DataCubeS2/Belcher_10m/mediane_bands
mkdir -p DataCubeS2/Belcher_10m/mediane_indices

# Compute median for clipped bands
echo "Computing median for clipped Sentinel-2 bands..."
for f in DataCubeS2/Bands_S2_10m_2023/*Belcher_deflate.tif; do
    base_filename=$(basename "$f")
    python code/final_codes/regression/sentinel_median/compute_median.py "$f" "DataCubeS2/Belcher_10m/mediane_bands/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

# Compute median for clipped indices
echo "Computing median for clipped Sentinel-2 indices..."
for f in DataCubeS2/Indices_S2_10m_2023/*Belcher_deflate.tif; do
    base_filename=$(basename "$f")
    python code/final_codes/regression/sentinel_median/compute_median.py "$f" "DataCubeS2/Belcher_10m/mediane_indices/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

echo "All processing completed for WAP32 data clipping and median calculation."
