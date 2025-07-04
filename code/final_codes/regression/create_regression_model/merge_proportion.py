"""
Merge Proportion

This script combines multiple proportion JSON files from different sites into a
unified dataset for training regression models. It standardizes feature names
across different sites and creates visualizations to help understand the combined dataset.

This is a key step in building a multi-site regression model, as it brings together
data from different geographic locations with potentially different distributions.

Usage:
  python merge_proportion.py json_file1.json json_file2.json ... --output merged_output.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def normalize_feature_name(feature_name):
    """
    Normalize feature names to ensure consistency across different sites.
    
    Different sites may have slightly different naming conventions for the same
    spectral bands or indices. This function extracts the core band/index name
    to enable proper merging of data from different sources.
    
    Args:
        feature_name: Original feature name with site-specific information
        
    Returns:
        normalized_name: Standardized feature name (e.g., "B12", "NDVI")
    """
   
    indices = ['NDVI', 'GNDVI', 'NDWI', 'MNDWI', 'NBR', 'GCC', 'MSI', 'CI_B5', 'CI_B7']
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    
    for index in indices:
        if f'_{index}_' in feature_name:
            return index
    for band in bands : 
        if f'Band{band}_' in feature_name:
            return band
   
    return feature_name

def load_json_files(json_paths):
    """
    Load and combine multiple JSON files containing proportion data.
    
    This function:
    1. Reads each JSON file
    2. Normalizes feature names for consistency
    3. Adds source information to each pixel
    4. Combines all pixels into a single dataset
    
    Args:
        json_paths: List of paths to JSON files to load
        
    Returns:
        merged_data: Combined list of all pixel data
        source_info: Dictionary mapping pixel index to source file
    """
    merged_data = []
    source_info = {}
    
    print(f"Loading {len(json_paths)} JSON files...")
    
    for i, json_path in enumerate(json_paths):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Store source information for each pixel
            base_index = len(merged_data)
            for j in range(len(data)):
                source_info[base_index + j] = os.path.basename(json_path)
            
            # Normalize feature names and add source information to each pixel
            for pixel in data:
                pixel['source'] = os.path.basename(json_path)
                
                # Normalize feature names
                if 'features' in pixel:
                    normalized_features = {}
                    for feat_name, feat_value in pixel['features'].items():
                        normalized_name = normalize_feature_name(feat_name)
                        normalized_features[normalized_name] = feat_value
                    pixel['features'] = normalized_features
            
            # Add to merged data
            merged_data.extend(data)
            print(f"  {json_path}: {len(data)} pixels")
            
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    print(f"Total pixels loaded: {len(merged_data)}")
    return merged_data, source_info

def convert_to_dataframe(merged_data):
    """
    Convert merged JSON data to a pandas DataFrame for easier analysis.
    
    This flattens the nested JSON structure into a tabular format where
    each row is a pixel and columns include features and proportions.
    
    Args:
        merged_data: List of pixel data dictionaries
        
    Returns:
        df: DataFrame with features and proportions as columns
    """
    # Initialize lists to store data
    rows = []
    
    # Process each pixel
    for pixel in merged_data:
        row = {}
        
        # Add source information
        row['source'] = pixel.get('source', 'unknown')
        
        # Add features
        if 'features' in pixel:
            for feat_name, feat_value in pixel['features'].items():
                row[feat_name] = feat_value
        
        # Add proportions
        if 'proportions' in pixel:
            for prop_name, prop_value in pixel['proportions'].items():
                row[prop_name] = prop_value
        
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

def plot_proportion_distributions(df, output_dir, purcent_exclusion=0.05):
    """
    Create histograms showing distribution of proportions in the merged dataset.
    
    These visualizations help understand the overall distribution of classes
    and identify potential biases or imbalances that might need correction.
    
    Args:
        df: DataFrame containing proportion data
        output_dir: Directory to save the plots
        purcent_exclusion: Threshold below which samples are excluded from visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot merged proportions
    merged_props = ['lichen_proportion', 'green_proportion', 'trough_proportion']
    sqrt_merged_props = ['sqrt_lichen_proportion', 'sqrt_green_proportion', 'sqrt_trough_proportion']
    
    # Check if these columns exist
    merged_props = [col for col in merged_props if col in df.columns]
    sqrt_merged_props = [col for col in sqrt_merged_props if col in df.columns]
    
    if not merged_props:
        print("No proportion columns found in the data")
        return
    
    # Plot merged class proportions
    plt.figure(figsize=(15, 5))
    n_cols = len(merged_props)
    
    for i, col in enumerate(merged_props):
        plt.subplot(1, n_cols, i+1)
        low_count = (df[col] <= purcent_exclusion).sum()
        filtered_data = df[df[col] > purcent_exclusion][col]
        filtered_data.hist(bins=20)
        plt.title(f"{col}\n({low_count} samples ≤ {purcent_exclusion*100:.0f}% excluded)")
        plt.xlabel('Proportion')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merged_classes_histograms.png'))
    plt.close()
    
    # Plot sqrt merged class proportions if they exist
    if sqrt_merged_props:
        plt.figure(figsize=(15, 5))
        
        for i, col in enumerate(sqrt_merged_props):
            plt.subplot(1, n_cols, i+1)
            low_count = (df[col] <= purcent_exclusion).sum()
            filtered_data = df[df[col] > purcent_exclusion][col]
            filtered_data.hist(bins=20)
            plt.title(f"{col}\n({low_count} samples ≤ {purcent_exclusion*100:.0f}% excluded)")
            plt.xlabel('Sqrt Proportion')
            plt.ylabel('Count')
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sqrt_merged_classes_histograms.png'))
        plt.close()
    
    # Plot distributions by source if multiple sources exist
    sources = df['source'].unique()
    if len(sources) > 1:
        print(f"Creating per-source distribution plots for {len(sources)} sources...")
        
        # Plot for each proportion column
        for col in merged_props:
            plt.figure(figsize=(10, 6))
            
            for source in sources:
                source_data = df[df['source'] == source][col]
                if not source_data.empty:
                    source_data.hist(alpha=0.6, bins=20, label=source)
            
            plt.title(f"{col} by source")
            plt.xlabel('Proportion')
            plt.ylabel('Count')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_by_source.png'))
            plt.close()
    
    print(f"Proportion distribution plots saved in {output_dir}")

def merge_proportions(json_paths, output_json, plots_dir):
    """
    Main function to merge proportion files and create analysis plots.
    
    This function orchestrates the entire merging process:
    1. Load and combine multiple JSON files
    2. Save the merged data to a new JSON file
    3. Create visualizations of the merged data
    4. Generate summary statistics
    
    Args:
        json_paths: List of paths to JSON files to merge
        output_json: Path to save the merged JSON file
        plots_dir: Directory to save distribution plots
        
    Returns:
        tuple: (merged JSON data, pandas DataFrame of merged data)
    """
    # Step 1: Load and merge JSON files
    merged_data, source_info = load_json_files(json_paths)
    
    if not merged_data:
        print("No data to merge. Exiting.")
        return
    
    # Step 2: Save merged JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"Merged JSON saved to {output_json}")
    
    # Step 3: Convert to DataFrame for analysis and plotting
    df = convert_to_dataframe(merged_data)
    
    # Step 4: Create plots
    plot_proportion_distributions(df, plots_dir, purcent_exclusion=0.00)
    
    # Step 5: Generate summary statistics
    print("\nSummary statistics:")
    print(f"Total pixels: {len(df)}")
    
    # Count pixels by source
    sources = df['source'].unique()
    if len(sources) > 1:
        print("\nPixels by source:")
        for source in sources:
            count = (df['source'] == source).sum()
            print(f"  {source}: {count} pixels ({count/len(df)*100:.1f}%)")
    
    # Show proportion statistics
    proportion_cols = [col for col in df.columns if 'proportion' in col and not col.startswith('sqrt_')]
    if proportion_cols:
        print("\nProportion statistics (mean ± std):")
        for col in proportion_cols:
            mean = df[col].mean()
            std = df[col].std()
            print(f"  {col}: {mean:.3f} ± {std:.3f}")
    
    return merged_data, df

def main():
    """
    Parse command line arguments and execute the merging process.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge multiple JSON files containing Sentinel pixel proportions.")
    parser.add_argument('json_files', nargs='+', help="JSON files to merge")
    parser.add_argument('--output', default="merged_proportions.json", help="Output path for merged JSON")
    parser.add_argument('--plots-dir', default="plots", help="Directory to save distribution plots")
    
    args = parser.parse_args()
    
    # Merge proportions and create plots
    merge_proportions(
        json_paths=args.json_files, 
        output_json=args.output,
        plots_dir=args.plots_dir
    )

if __name__ == "__main__":
    main()

# Command line examples:
"""
Linux example:
python code/final_codes/regression/merge_proportion.py \
 data/regressions/regression_multisite/regression_Belcher_10m/proportions_Belcher_proportions.json \
 data/regressions/regression_multisite/regression_Chesnay_10m/proportions_Chesnay_proportions.json \
 data/regressions/regression_multisite/regression_wap23_10m/proportions_WAP23_proportions.json \
 --output data/regressions/regression_multisite/merged/merged_pixels.json \
 --plots-dir data/regressions/regression_multisite/merged

PowerShell example:
python code/final_codes/regression/merge_proportion.py `
 data/regressions/regression_multisite/regression_Belcher_10m/proportions_Belcher_proportions.json `
 data/regressions/regression_multisite/regression_Chesnay_10m/proportions_Chesnay_proportions.json `
 --output data/regressions/regression_multisite/merged/merged_pixels.json `
 --plots-dir data/regressions/regression_multisite/merged
"""