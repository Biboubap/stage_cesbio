#!/bin/bash
#
# Sentinel-2 Temporal Median Calculation
#
# This script computes temporal median composites for Sentinel-2 bands and indices.
# It processes all available bands and indices for a specific site, creating
# cloud-free, stable images that represent the median value across all acquisitions.
#
# These median composites are later used as input features for regression models.
#

# Define the site name (can be changed for different sites)
sitename="Lamprey"

# Create directories for storing median calculation results
mkdir -p DataCubeS2/${sitename}_10m/mediane_bands
mkdir -p DataCubeS2/${sitename}_10m/mediane_indices

# Compute median for clipped spectral bands
echo "Computing median for clipped Sentinel-2 bands..."
for f in DataCubeS2/Bands_S2_10m_2023/*${sitename}_deflate.tif; do
    # Extract base filename for the output
    base_filename=$(basename "$f")
    
    # Run the Python script to compute temporal median for this band
    python code/final_codes/regression/sentinel_median/compute_median.py "$f" "DataCubeS2/${sitename}_10m/mediane_bands/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

# Compute median for clipped spectral indices
echo "Computing median for clipped Sentinel-2 indices..."
for f in DataCubeS2/Indices_S2_10m_2023/*${sitename}_deflate.tif; do
    # Extract base filename for the output
    base_filename=$(basename "$f")
    
    # Run the Python script to compute temporal median for this index
    python code/final_codes/regression/sentinel_median/compute_median.py "$f" "DataCubeS2/${sitename}_10m/mediane_indices/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

echo "All processing completed for ${sitename} data clipping and median calculation."