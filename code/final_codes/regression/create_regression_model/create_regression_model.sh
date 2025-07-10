#!/bin/bash

# Array of site names
sites=("Chesnay")

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


echo "----------------------------------------"

sites=("WAP12" "WAP32" "Belcher" "Chesnay")
resolutions=("10m" "5m")
echo "Merging proportion files from all sites with 10m resolution..."
for resolution in "${resolutions[@]}"; do
   # Create the output directory for merged results
    MERGE_OUTPUT_DIR="/home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/merged_${resolution}"
    mkdir -p "$MERGE_OUTPUT_DIR"

    # Build the command to merge all proportion files
    MERGE_CMD="python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/merge_proportion.py"

    # Add all proportion files to the merge command (10m resolution)
    for site in "${sites[@]}"; do
        PROP_FILE="/media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/${site}_${resolution}/proportions_${site}_${resolution}.json"
        if [ -f "$PROP_FILE" ]; then
            echo "Adding $site proportion file to merge"
            MERGE_CMD="$MERGE_CMD $PROP_FILE"
        else
            echo "Warning: Proportion file not found for $site: $PROP_FILE"
        fi
    done

    # Add output options to the merge command
    MERGE_CMD="$MERGE_CMD --output $MERGE_OUTPUT_DIR/merged_pixels_${resolution}.json --plots-dir $MERGE_OUTPUT_DIR"

    # Execute the merge command
    echo "Executing merge command:"
    echo "$MERGE_CMD"
    eval "$MERGE_CMD"

    
    echo "Merging process completed!"
    echo "Merged file saved to: $MERGE_OUTPUT_DIR/merged_pixels_${resolution}.json"
    echo "Plots saved to: $MERGE_OUTPUT_DIR"
    
done

echo "----------------------------------------"

echo "Balancing pixel proportions..."
for resolution in "${resolutions[@]}"; do
    python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/balance_proportion.py \
        "/home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/merged_${resolution}/merged_pixels_${resolution}.json" \
        --output "/home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/balanced_${resolution}" \
        --quantile 0.6 \
        --bins 25
done 
echo "Balancing process completed for all resolutions!"


echo "----------------------------------------"
echo "Training regression model with balanced data..." 
resolutions=("10m" "5m" )
for resolution in "${resolutions[@]}"; do
    for category in "lichen" "trough"; do
        python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/train_regression_model.py \
            "/home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/balanced_${resolution}/balanced_${category}_proportion.json" \
            "/home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/regression_models_${resolution}/${category}" \

    done
done

echo "Regression model training completed for all categories and resolutions!"