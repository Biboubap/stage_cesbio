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

# Array of site names
sites=("WAP12" "WAP23" "WAP32" "Belcher" "Chesnay" "Lamprey")

# Array of resolutions
resolutions=("5m" "10m")

# Loop through each site and resolution
for sitename in "${sites[@]}"; do
    for resolution in "${resolutions[@]}"; do

        # Create directories for storing median calculation results
        mkdir -p media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${sitename}_${resolution}/mediane_bands
        mkdir -p media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${sitename}_${resolution}/mediane_indices

        # Compute median for clipped spectral bands
        echo "Computing median for clipped Sentinel-2 bands..."
        for f in media/lcousin/FASTBOYSLIM/Churchill/DataCubeS2/Bands_S2_${resolution}_2023/*${sitename}_deflate.tif; do
            # Extract base filename for the output
            base_filename=$(basename "$f")
            
            # Run the Python script to compute temporal median for this band
            python home/lcousin/stage_cesbio/code/final_codes/regression/sentinel_median/compute_median.py "$f" "media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${sitename}_${resolution}/mediane_bands/mediane_${base_filename}"
            echo "Computed median for $base_filename"
        done

        # Compute median for clipped spectral indices
        echo "Computing median for clipped Sentinel-2 indices..."
        for f in media/lcousin/FASTBOYSLIM/Churchill/DataCubeS2/Indices_S2_${resolution}_2023/*${sitename}_deflate.tif; do
            # Extract base filename for the output
            base_filename=$(basename "$f")
            
            # Run the Python script to compute temporal median for this index
            python home/lcousin/stage_cesbio/code/final_codes/regression/sentinel_median/compute_median.py "$f" "media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${sitename}_${resolution}/mediane_indices/mediane_${base_filename}"
            echo "Computed median for $base_filename"
        done

        echo "All processing completed for ${sitename} data median calculation at ${resolution}."
        
    done
done
