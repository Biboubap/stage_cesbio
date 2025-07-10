#!/bin/bash

# Script to fuse 5m resolution regression predictions to 10m resolution
# This aggregates 4 pixels at 5m into 1 pixel at 10m using averaging

# Create output directory for 5m fusion results
FUSION_OUTPUT_DIR="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_map/regression_map_5m_fusion"
mkdir -p "$FUSION_OUTPUT_DIR"

# Array of site names
sites=("WAP12" "WAP23" "WAP32" "Belcher" "Chesnay" "Lamprey")

# Categories to process
categories=("lichen" "trough")

echo "Starting 5m to 10m fusion process..."

# Process each site and category
for site in "${sites[@]}"; do
    echo "Processing site: $site"
    
    # For each site, get a 10m band to use as reference for target grid
    REFERENCE_BAND="/media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${site}_10m/mediane_bands/mediane_STACK_2023_BandB4_${site}_deflate.tif"
    
    
    # Process each category
    for category in "${categories[@]}"; do
        echo "  Processing category: $category"
        
        # Path to the 5m prediction
        INPUT_5M="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_map/regression_map_5m/${site}_5m_${category}_prediction.tif"
        
        # Path to save the 10m fused output
        OUTPUT_10M="${FUSION_OUTPUT_DIR}/${site}_5m_fusion_${category}_prediction.tif"
        
        if [ ! -f "$INPUT_5M" ]; then
            echo "Warning: Input 5m prediction not found: $INPUT_5M"
            continue
        fi
        
        echo "  Fusing 5m prediction to 10m using average resampling..."
        
        # Use gdalwarp to resample 5m to 10m using average method
        # -tr 10 10: Target resolution 10m x 10m
        # -r average: Use averaging for resampling (fuse 4 pixels into 1)
        # -te: Use the extent from the 10m reference
        # -tap: Align output pixels with the target grid
        
        # First get the extent of the reference band
        EXTENT=$(gdalinfo $REFERENCE_BAND | grep "Lower Left\|Upper Right" | 
                 awk '{gsub(/[(),]/, " "); print $3, $4}' | 
                 awk 'BEGIN {ORS=" "} {print}' | 
                 awk '{print $1, $4, $3, $2}')
        
        # Apply the warp operation
        gdalwarp -overwrite -r average -tr 10 10 -te $EXTENT -tap \
                 "$INPUT_5M" "$OUTPUT_10M" \
                 -co COMPRESS=DEFLATE -co TILED=YES
        
        # Check if the warp was successful
        if [ $? -eq 0 ]; then
            echo "  Successfully created fused 10m prediction: $OUTPUT_10M"
        else
            echo "  Error during fusion process for $site $category"
        fi
    done
done

echo "5m fusion process completed!"
