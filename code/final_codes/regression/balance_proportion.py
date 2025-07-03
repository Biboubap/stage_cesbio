"""
Balance Proportion

This script applies median-based balancing to proportion data from JSON files
(typically output from merge_proportion.py) and creates balanced datasets for
each category (Lichen, Green, Through) in both normal and sqrt-transformed versions.

Usage:
  python balance_proportion.py input.json --output output_dir --quantile 0.5 --bins 25
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def normalize_feature_name(feature_name):
    """
    Normalize feature names by extracting the core band or index name.
    
    Args:
        feature_name: Original feature name with site-specific information
        
    Returns:
        normalized_name: Standardized feature name (e.g., "B12", "NDVI")
    """
    # List of all possible band/index names to extract
    band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    index_names = ['CI_B7', 'NDWI', 'MSI', 'NDVI', 'GCC', 'CI_B5', 'NBR', 'GNDVI', 'MNDWI']
    
    # Check for band names
    for band in band_names:
        if band in feature_name:
            return band
            
    # Check for index names
    for index in index_names:
        if index in feature_name:
            return index
            
    # If no match found, return the original name
    return feature_name

def create_balanced_dataset(input_data, target_col, n_bins=25, quantile=0.5):
    """
    Create a balanced dataset by dividing the values of target_col into n_bins bins of equal size
    and sampling to ensure each bin has approximately the same number of samples.
    
    Args:
        input_data: DataFrame containing the input data
        target_col: Target column to balance (e.g., 'lichen_proportion', 'green_proportion')
        n_bins: Number of bins to divide the data into
        quantile: Quantile to use for determining the number of samples per bin
    
    Returns:
        DataFrame with the balanced dataset
    """
    print(f"Creating balanced dataset for '{target_col}'...")
    
    # Filter out rows with NaN in target column
    df_clean = input_data.dropna(subset=[target_col])
    
    # Filter out zero values
    df_clean = df_clean[df_clean[target_col] > 0]
    
    # Get number of samples
    n_samples = len(df_clean)
    print(f"Original dataset: {n_samples} samples (after removing zeros)")
    
    # Determine the min and max values to create fixed-size bins
    min_val = 0
    max_val = 1
    bin_width = (max_val - min_val) / n_bins
    
    # Create bins of equal size
    bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]
    
    # Create balanced dataset
    balanced_dfs = []
    
    print(f"Creating {n_bins} bins of equal size from {min_val:.4f} to {max_val:.4f}...")
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Get samples in this bin
        if i == n_bins - 1:  # Include upper bound in the last bin
            bin_samples = df_clean[(df_clean[target_col] >= lower) & (df_clean[target_col] <= upper)]
        else:
            bin_samples = df_clean[(df_clean[target_col] >= lower) & (df_clean[target_col] < upper)]
        
        bin_count = len(bin_samples)
        
        print(f"Bin {i+1}/{n_bins}: {lower:.4f} to {upper:.4f}, {bin_count} samples")
        balanced_dfs.append(bin_samples)
    
    # Calculate median number of samples per bin (excluding empty bins)
    non_empty_counts = [len(df) for df in balanced_dfs if len(df) > 0]
    quantile_count = int(np.quantile(non_empty_counts, quantile))
    print(f"{quantile*100}% quantile number of samples per bin: {quantile_count}")
    
    # Sample each bin to have at most quantile_count samples
    final_dfs = []
    for i, bin_df in enumerate(balanced_dfs):
        if len(bin_df) == 0:
            continue
            
        if len(bin_df) > quantile_count:
            # Random sample without replacement
            sampled_df = bin_df.sample(n=quantile_count, random_state=42)
            print(f"Bin {i+1}: Sampled from {len(bin_df)} to {len(sampled_df)} samples")
            final_dfs.append(sampled_df)
        else:
            print(f"Bin {i+1}: Kept all {len(bin_df)} samples")
            final_dfs.append(bin_df)
    
    # Concatenate all bins
    balanced_df = pd.concat(final_dfs)
    print(f"Balanced dataset: {len(balanced_df)} samples")
    
    return balanced_df

def plot_balanced_histogram(data, column, output_path, n_bins=25):
    """
    Create a histogram showing the distribution of values in a column.
    
    Args:
        data: DataFrame containing the data
        column: Column to plot
        output_path: Path to save the plot
        n_bins: Number of bins for the histogram
    """
    plt.figure(figsize=(10, 6))
    
    # Fixed bin edges for consistent histograms
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # Plot histogram
    data[column].hist(bins=bin_edges)
    
    plt.title(f'Distribution of {column}\nSamples: {len(data)}')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()

def balance_json_proportions(json_path, output_dir, n_bins=25, quantile=0.5):
    """
    Balance proportions in a JSON file and save balanced outputs.
    
    Args:
        json_path: Path to input JSON file (from merge_proportion.py)
        output_dir: Directory to save outputs
        n_bins: Number of bins to divide the data into
        quantile: Quantile to use for determining the number of samples per bin
    """
    print(f"Loading JSON data from {json_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} pixels from JSON")
    
    # Normalize feature names if not already normalized
    for pixel in data:
        if 'features' in pixel:
            normalized_features = {}
            for feat_name, feat_value in pixel['features'].items():
                normalized_name = normalize_feature_name(feat_name)
                normalized_features[normalized_name] = feat_value
            pixel['features'] = normalized_features
    
    # Convert to DataFrame for easier processing
    df = pd.json_normalize(data)
    
    # List of columns to balance
    balance_cols = [
        'proportions.lichen_proportion', 
        'proportions.green_proportion', 
        'proportions.trough_proportion',
        'proportions.sqrt_lichen_proportion', 
        'proportions.sqrt_green_proportion', 
        'proportions.sqrt_trough_proportion'
    ]
    
    # Verify which columns exist in the data
    available_cols = [col for col in balance_cols if col in df.columns]
    
    if not available_cols:
        print("Error: No proportion columns found in the JSON data.")
        return
    
    print(f"Found {len(available_cols)} proportion columns: {available_cols}")
    
    # Process each column
    for col in available_cols:
        print(f"\n--- Processing {col} ---")
        
        # Create balanced dataset
        balanced_df = create_balanced_dataset(
            input_data=df,
            target_col=col,
            n_bins=n_bins,
            quantile=quantile
        )
        
        # Create column name for output file
        col_name = col.split('.')[-1]
        
        # Create output JSON
        output_json = []
        
        # Keep only the proportion column we're balancing
        for _, row in balanced_df.iterrows():
            pixel = {}
            
            # Add features
            if 'features' in row:
                pixel['features'] = row['features']
            else:
                pixel['features'] = {feat_col.replace('features.', ''): row[feat_col] 
                                    for feat_col in row.index if feat_col.startswith('features.')}
            
            # Add the specific proportion we're balancing
            pixel['proportions'] = {col_name: row[col]}
            
            # Add source if available
            if 'source' in row:
                pixel['source'] = row['source']
            
            output_json.append(pixel)
        
        # Save balanced JSON
        output_json_path = os.path.join(output_dir, f"balanced_{col_name}.json")
        with open(output_json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        
        print(f"Saved balanced JSON to {output_json_path}")
        
        # Create histogram
        plot_path = os.path.join(output_dir, f"histogram_{col_name}.png")
        plot_balanced_histogram(balanced_df, col, plot_path, n_bins)
        print(f"Saved histogram to {plot_path}")
    
    print("\nAll processing completed!")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Balance proportions in JSON files.")
    parser.add_argument('json_file', help="Input JSON file (from merge_proportion.py)")
    parser.add_argument('--output', default="balanced", help="Output directory for balanced data")
    parser.add_argument('--bins', type=int, default=25, help="Number of bins (default: 25)")
    parser.add_argument('--quantile', type=float, default=0.5, 
                        help="Quantile to use for determining samples per bin (default: 0.5)")
    
    args = parser.parse_args()
    
    # Balance proportions
    balance_json_proportions(
        json_path=args.json_file,
        output_dir=args.output,
        n_bins=args.bins,
        quantile=args.quantile
    )

if __name__ == "__main__":
    main()

# python code/final_codes/regression/balance_proportion.py \
#     data/regressions/regression_multisite/merged/merged_pixels.json \
#     --output data/regressions/regression_multisite/balanced \
#     --quantile 0.6 \
#     --bins 25