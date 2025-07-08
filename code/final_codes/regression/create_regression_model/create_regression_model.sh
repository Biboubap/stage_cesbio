#!/bin/bash

# Array of site names
sites=("WAP12" "WAP23" "WAP32" "Belcher" "Chesnay" "Lamprey")

# Array of resolutions
resolutions=("5m" "10m")

# Loop through each site and resolution
for site in "${sites[@]}"; do
    for resolution in "${resolutions[@]}"; do
        echo "Processing site: $site with resolution: $resolution"
        
        # Define paths based on site and resolution
        classification_path="/media/lcousin/FASTBOYSLIM/Loris/final_data/drone_classif_peatcut/${site}_classif_cut.tif"
        sentinel_band_path="/media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${site}_${resolution}/mediane_bands/mediane_STACK_2023_BandB4_${site}_deflate.tif"
        bands_dir="/media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${site}_${resolution}/mediane_bands"
        indices_dir="/media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${site}_${resolution}/mediane_indices"
        output_dir="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/${site}_${resolution}"
        
        # Check if input files exist
        if [ ! -f "$classification_path" ]; then
            echo "Warning: Classification file not found: $classification_path"
            continue
        fi
        
        if [ ! -f "$sentinel_band_path" ]; then
            echo "Warning: Sentinel band file not found: $sentinel_band_path"
            continue
        fi
        
        # Create output directory if it doesn't exist
        mkdir -p "$output_dir"
        
        # Run the compute_proportion.py script
        echo "Running compute_proportion.py for $site with $resolution resolution..."
        python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/compute_proportion.py \
            --classification "$classification_path" \
            --sentinel-band "$sentinel_band_path" \
            --bands-dir "$bands_dir" \
            --indices-dir "$indices_dir" \
            --output-dir "$output_dir" \
            --site-name "${site}_${resolution}"
        
        echo "Finished processing $site with $resolution resolution"
        echo "----------------------------------------"
    done
done

echo "All sites and resolutions processed!"

# # Now merge the files for all sites with 10m resolution
# echo "----------------------------------------"
# echo "Merging proportion files from all sites with 10m resolution..."

# # Create the output directory for merged results
# MERGE_OUTPUT_DIR="/home/lcousin/stage_cesbio/data/regressions/regression_multisite/merged_all"
# mkdir -p "$MERGE_OUTPUT_DIR"

# # Build the command to merge all proportion files
# MERGE_CMD="python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/merge_proportion.py"

# # Add all proportion files to the merge command (10m resolution)
# for site in "${sites[@]}"; do
#     PROP_FILE="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/${site}_10m/proportions_${site}_10m.json"
#     if [ -f "$PROP_FILE" ]; then
#         echo "Adding $site proportion file to merge"
#         MERGE_CMD="$MERGE_CMD $PROP_FILE"
#     else
#         echo "Warning: Proportion file not found for $site: $PROP_FILE"
#     fi
# done

# # Add output options to the merge command
# MERGE_CMD="$MERGE_CMD --output $MERGE_OUTPUT_DIR/merged_pixels_all.json --plots-dir $MERGE_OUTPUT_DIR"

# # Execute the merge command
# echo "Executing merge command:"
# echo "$MERGE_CMD"
# eval "$MERGE_CMD"

# echo "----------------------------------------"
# echo "Merging process completed!"
# echo "Merged file saved to: $MERGE_OUTPUT_DIR/merged_pixels_all.json"
# echo "Plots saved to: $MERGE_OUTPUT_DIR"
