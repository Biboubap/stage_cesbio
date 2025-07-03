"""
Merge Proportion

This script merges multiple JSON files containing Sentinel-2 pixel proportions and features
from different sites/locations and creates plots showing the distribution of proportions.

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
    Normalize feature names by extracting the core band or index name.
    
    Args:
        feature_name: Original feature name with site-specific information
        
    Returns:
        normalized_name: Standardized feature name (e.g., "B12", "NDVI")
    """
    # Extract band names (format: BandB2, BandB8A, BandB12, etc.)
    band_match = re.search(r'Band(B[0-9]+A?)', feature_name)
    if band_match:
        return band_match.group(1)  # Return just the band name (B2, B8A, B12)
    
    # Extract index names (format: _NDVI_, _MNDWI_, _CI_B7_, etc.)
    # List all indices we're looking for
    indices = ['NDVI', 'GNDVI', 'NDWI', 'MNDWI', 'NBR', 'GCC', 'MSI', 'CI_B5', 'CI_B7']
    
    for index in indices:
        if f'_{index}_' in feature_name:
            return index
    
    # If no match found, check for direct occurrence of band or index names
    for name in indices + ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']:
        if re.search(r'\b' + re.escape(name) + r'\b', feature_name):
            return name
    
    # If no match found, return the original name
    return feature_name

def load_json_files(json_paths):
    """
    Load and combine multiple JSON files.
    
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
    Convert merged JSON data to a pandas DataFrame with flattened structure.
    
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
    Create histograms showing the distribution of class proportions.
    
    Args:
        df: DataFrame containing proportion data
        output_dir: Directory to save the plots
        purcent_exclusion: Threshold below which samples are excluded (default: 0.05 = 5%)
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
    Merge multiple JSON files containing proportions and create plots.
    
    Args:
        json_paths: List of paths to JSON files to merge
        output_json: Path to save the merged JSON file
        plots_dir: Directory to save distribution plots
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

#python merge_proportion.py path/to/proportions1.json path/to/proportions2.json --output merged_output.json --plots-dir plots_directory

# python code/final_codes/regression/merge_proportion.py\
#  data/regressions/regression_multisite/regression_Belcher_10m/proportions_Belcher_proportions.json\
#  data/regressions/regression_multisite/regression_Chesnay_10m/proportions_Chesnay_proportions.json\
#  data/regressions/regression_multisite/regression_wap23_10m/proportions_WAP23_proportions.json\
#  data/regressions/regression_multisite/regression_wap32_10m/proportions_WAP32_proportions.json\
#  data/regressions/regression_multisite/regression_Lamprey_10m/proportions_Lamprey_proportions.json\
#  data/regressions/regression_multisite/regression_WAP12_10m/proportions_WAP12_proportions.json\
# --output data/regressions/regression_multisite/merged/merged_pixels.json\
#  --plots data/regressions/regression_multisite/merged