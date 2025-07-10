#!/bin/bash

# Array of site names
sites=("WAP12" "WAP23" "WAP32" "Belcher" "Chesnay" "Lamprey")

# Array of resolutions
resolutions=("5m" "10m")

# Categories to process
categories=("lichen" "trough")

# Create the output directories
for resolution in "${resolutions[@]}"; do
    mkdir -p "/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_map/regression_map_${resolution}"
done

# Loop through each site, resolution, and category
for site in "${sites[@]}"; do
    for resolution in "${resolutions[@]}"; do
        echo "Processing site: $site with resolution: $resolution"
        
        # Define paths based on site and resolution
        bands_dir="/media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${site}_${resolution}/mediane_bands"
        indices_dir="/media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/${site}_${resolution}/mediane_indices"
        output_dir="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_map/regression_map_${resolution}"
        
        # Check if input directories exist
        if [ ! -d "$bands_dir" ]; then
            echo "Warning: Bands directory not found: $bands_dir"
            continue
        fi
        
        if [ ! -d "$indices_dir" ]; then
            echo "Warning: Indices directory not found: $indices_dir"
            continue
        fi
        
        # Process each category
        for category in "${categories[@]}"; do
            echo "Applying regression model for category: $category"
            
            # Define model path and output path
            model_path="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_models_${resolution}/${category}/${category}_proportion_model.joblib"
            output_path="${output_dir}/${site}_${resolution}_${category}_prediction.tif"
            
            # Check if model exists
            if [ ! -f "$model_path" ]; then
                echo "Warning: Model file not found: $model_path"
                continue
            fi
            
            # Run the apply_regression_model.py script
            echo "Running apply_regression_model.py for $site with $resolution resolution for $category category..."
            python /home/lcousin/stage_cesbio/code/final_codes/regression/apply_regression_model/apply_regression_model.py \
                --bands-dir "$bands_dir" \
                --indices-dir "$indices_dir" \
                --model "$model_path" \
                --output "$output_path"
            
            echo "Finished applying regression model for $category on $site with $resolution resolution"
        done
        
        echo "----------------------------------------"
    done
done

echo "All regression maps have been generated!"
