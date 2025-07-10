#!/usr/bin/env python3
"""
Band Histograms by Lichen Proportion Bins

This script creates histograms of B2 values for different bins of lichen proportion,
comparing 5m and 10m resolutions side-by-side. For each 5% bin of lichen proportion,
the script creates a histogram pair showing the distribution of B2 values.

Usage:
  python band_histograms.py --balanced-5m path/to/balanced_5m_file.json 
                          --balanced-10m path/to/balanced_10m_file.json
                          --output-dir path/to/output_directory
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def parse_arguments():
    """
    Parse command line arguments for the band histogram analysis.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Create histograms of band values for different proportion bins",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--balanced-5m', required=True,
                        help='Path to the balanced 5m JSON file')
    parser.add_argument('--balanced-10m', required=True,
                        help='Path to the balanced 10m JSON file')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save the output plots')
    
    # Optional arguments
    parser.add_argument('--proportion-type', default='lichen_proportion',
                        help='Type of proportion to analyze (lichen_proportion, trough_proportion, etc.)')
    parser.add_argument('--band', default='B2',
                        help='Band to analyze (B2, B3, etc.)')
    parser.add_argument('--n-bins', type=int, default=30,
                        help='Number of bins for histograms')
    
    return parser.parse_args()

def load_json_data(json_path):
    """
    Load proportion data from a JSON file.
    
    Args:
        json_path: Path to the JSON file with balanced samples
        
    Returns:
        DataFrame containing features and proportions
    """
    print(f"Loading data from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to pandas DataFrame for easier processing
    rows = []
    for item in data:
        row = {}
        
        # Extract features
        if 'features' in item:
            for feat_name, feat_value in item['features'].items():
                row[feat_name] = feat_value
        
        # Extract proportions
        if 'proportions' in item:
            for prop_name, prop_value in item['proportions'].items():
                row[prop_name] = prop_value
            
        # Extract site information
        if 'source' in item:
            row['source'] = item['source']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df

def create_proportion_bin_histograms(df_5m, df_10m, output_dir, proportion_type, band, hist_bins):
    """
    Create histograms of band values for different proportion bins.
    
    For each 5% bin of proportion, creates a pair of histograms showing
    the distribution of band values for 10m (left) and 5m (right) resolutions.
    The 5m data is randomly subsampled to match the number of 10m samples.
    
    Args:
        df_5m: DataFrame containing 5m resolution data
        df_10m: DataFrame containing 10m resolution data
        output_dir: Directory to save the output plots
        proportion_type: Type of proportion to analyze (e.g., 'lichen_proportion')
        band: Band to analyze (e.g., 'B2')
        hist_bins: Number of bins for histograms
    """
    print(f"\nCreating {band} histograms by {proportion_type} bins...")
    
    # Find band columns
    band_col_5m = next((col for col in df_5m.columns if band in col), None)
    band_col_10m = next((col for col in df_10m.columns if band in col), None)
    
    if band_col_5m is None or band_col_10m is None:
        print(f"Warning: Could not find {band} column in one or both datasets")
        print(f"Available columns in 5m data: {df_5m.columns.tolist()}")
        print(f"Available columns in 10m data: {df_10m.columns.tolist()}")
        return
    
    print(f"Using column '{band_col_5m}' for 5m data and '{band_col_10m}' for 10m data")
    
    # Check if proportion type exists in the data
    if proportion_type not in df_5m.columns or proportion_type not in df_10m.columns:
        print(f"Warning: {proportion_type} not found in the data")
        print(f"Available columns in 5m data: {df_5m.columns.tolist()}")
        print(f"Available columns in 10m data: {df_10m.columns.tolist()}")
        return
    
    # Create directory for histogram plots
    hist_dir = os.path.join(output_dir, f'{band}_{proportion_type}_histograms')
    os.makedirs(hist_dir, exist_ok=True)
    
    # Lists to store results for summary plot
    bin_centers = []
    means_5m = []
    stds_5m = []
    means_10m = []
    stds_10m = []
    
    # Define proportion bins (0-5%, 5-10%, etc.) - changed from 10% to 5% bins
    bin_edges = np.linspace(0, 1, 21)  # 21 edges for 20 bins of 5% each
    bin_labels = [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(len(bin_edges)-1)]
    
    # Create a figure for each proportion bin
    for i in range(len(bin_edges)-1):
        # Get bin range
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i+1]
        
        # Filter data for this bin
        df_5m_bin = df_5m[(df_5m[proportion_type] >= lower_bound) & (df_5m[proportion_type] < upper_bound)]
        df_10m_bin = df_10m[(df_10m[proportion_type] >= lower_bound) & (df_10m[proportion_type] < upper_bound)]
        
        # Skip if not enough data
        if len(df_5m_bin) < 10 or len(df_10m_bin) < 10:
            print(f"Skipping {bin_labels[i]} bin due to insufficient data")
            continue
        
        # Randomly subsample 5m data to match the number of 10m samples
        n_samples_10m = len(df_10m_bin)
        n_samples_5m = len(df_5m_bin)
        
        if n_samples_5m > n_samples_10m:
            # Randomly select n_samples_10m from 5m data
            df_5m_bin = df_5m_bin.sample(n=n_samples_10m, random_state=42)
            print(f"Bin {bin_labels[i]}: Subsampled 5m data from {n_samples_5m} to {n_samples_10m} samples")
        elif n_samples_5m < n_samples_10m:
            print(f"Warning: Bin {bin_labels[i]}: 5m samples ({n_samples_5m}) are fewer than 10m samples ({n_samples_10m})")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract band values
        band_values_10m = df_10m_bin[band_col_10m].values
        band_values_5m = df_5m_bin[band_col_5m].values
        
        # Calculate statistics
        mean_10m = np.mean(band_values_10m)
        std_10m = np.std(band_values_10m)
        n_samples_10m = len(band_values_10m)
        
        mean_5m = np.mean(band_values_5m)
        std_5m = np.std(band_values_5m)
        n_samples_5m = len(band_values_5m)
        
        # Store results for summary plot
        bin_center = (lower_bound + upper_bound) / 2
        bin_centers.append(bin_center)
        means_5m.append(mean_5m)
        stds_5m.append(std_5m)
        means_10m.append(mean_10m)
        stds_10m.append(std_10m)
        
        # Determine common x-axis range based on both datasets
        min_val = min(np.min(band_values_5m), np.min(band_values_10m))
        max_val = max(np.max(band_values_5m), np.max(band_values_10m))
        
        # Create histogram bins with common range
        common_bins = np.linspace(min_val, max_val, hist_bins)
        
        # 10m data (left) - normalized by density=True
        ax1.hist(band_values_10m, bins=common_bins, alpha=0.7, color='blue', density=True)
        ax1.set_title(f"10m Resolution, {bin_labels[i]} {proportion_type.split('_')[0]} (n={n_samples_10m})\nMean: {mean_10m:.2f}, Std: {std_10m:.2f}")
        ax1.set_xlabel(f"{band} Value")
        ax1.set_ylabel("Normalized Frequency")
        ax1.grid(alpha=0.3)
        
        # 5m data (right) - normalized by density=True
        ax2.hist(band_values_5m, bins=common_bins, alpha=0.7, color='green', density=True)
        ax2.set_title(f"5m Resolution, {bin_labels[i]} {proportion_type.split('_')[0]} (n={n_samples_5m})\nMean: {mean_5m:.2f}, Std: {std_5m:.2f}")
        ax2.set_xlabel(f"{band} Value")
        ax2.set_ylabel("Normalized Frequency")
        ax2.grid(alpha=0.3)
        
        # Set the same x-axis limits for both plots
        ax1.set_xlim(min_val, max_val)
        ax2.set_xlim(min_val, max_val)
        
        # Add overall title
        prop_name = proportion_type.split('_')[0].capitalize()
        fig.suptitle(f"Distribution of {band} Values for {bin_labels[i]} {prop_name} Proportion", fontsize=16)
        
        # Save figure
        output_path = os.path.join(hist_dir, f'histogram_{band}_{bin_labels[i].replace("%", "pct")}.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.savefig(output_path)
        plt.close()
        
        print(f"Histogram for {bin_labels[i]} saved to {output_path}")
    
    # Create summary plot with mean and std error bars
    create_mean_summary_plot(
        bin_centers, means_5m, stds_5m, means_10m, stds_10m,
        proportion_type, band, output_dir
    )

def create_mean_summary_plot(bin_centers, means_5m, stds_5m, means_10m, stds_10m, proportion_type, band, output_dir):
    """
    Create a summary plot showing mean values with error bars from standard deviations
    for both 5m and 10m resolutions across proportion bins.
    
    Args:
        bin_centers: List of bin center values (proportion bin midpoints)
        means_5m: List of mean values for 5m data
        stds_5m: List of standard deviation values for 5m data
        means_10m: List of mean values for 10m data
        stds_10m: List of standard deviation values for 10m data
        proportion_type: Type of proportion being analyzed
        band: Band being analyzed
        output_dir: Directory to save the output plot
    """
    # Convert lists to numpy arrays
    bin_centers = np.array(bin_centers)
    means_5m = np.array(means_5m)
    stds_5m = np.array(stds_5m)
    means_10m = np.array(means_10m)
    stds_10m = np.array(stds_10m)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot 5m data
    plt.errorbar(
        bin_centers, means_5m, yerr=stds_5m, 
        fmt='o-', color='green', ecolor='green', elinewidth=1, capsize=5, 
        label='5m Resolution'
    )
    
    # Plot 10m data
    plt.errorbar(
        bin_centers, means_10m, yerr=stds_10m, 
        fmt='s-', color='blue', ecolor='blue', elinewidth=1, capsize=5, 
        label='10m Resolution'
    )
    
    # Add labels and legend
    prop_name = proportion_type.split('_')[0].capitalize()
    plt.xlabel(f'{prop_name} Proportion')
    plt.ylabel(f'{band} Value (Mean ± Std)')
    plt.title(f'Mean {band} Values by {prop_name} Proportion with Standard Deviation')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Set x-axis to show proportion as percentage
    plt.xticks(np.linspace(0, 1, 11), [f'{int(x*100)}%' for x in np.linspace(0, 1, 11)])
    
    # Save figure
    summary_dir = os.path.join(output_dir, f'{band}_{proportion_type}_summary')
    os.makedirs(summary_dir, exist_ok=True)
    output_path = os.path.join(summary_dir, f'mean_summary_{band}.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Summary plot saved to {output_path}")

def main():
    """
    Main function to create band histograms by proportion bins.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load 5m data
    df_5m = load_json_data(args.balanced_5m)
    
    # Load 10m data
    df_10m = load_json_data(args.balanced_10m)
    
    # Create histograms by proportion bins
    create_proportion_bin_histograms(
        df_5m=df_5m,
        df_10m=df_10m,
        output_dir=args.output_dir,
        proportion_type=args.proportion_type,
        band=args.band,
        hist_bins=args.n_bins
    )
    
    print("\nHistogram generation completed!")

if __name__ == "__main__":
    main()

"""
python /home/lcousin/stage_cesbio/code/final_codes/regression/study_5m_dispersion/band_histograms.py \
    --balanced-5m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_5m/balanced_lichen_proportion.json \
    --balanced-10m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_10m/balanced_lichen_proportion.json \
    --output-dir /home/lcousin/stage_cesbio/data/study_5m_dispersion/band_histograms \
    --proportion-type lichen_proportion \
    --band B2

python /home/lcousin/stage_cesbio/code/final_codes/regression/study_5m_dispersion/band_histograms.py \
    --balanced-5m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_5m/balanced_trough_proportion.json \
    --balanced-10m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_10m/balanced_trough_proportion.json \
    --output-dir /home/lcousin/stage_cesbio/data/study_5m_dispersion/band_histograms \
    --proportion-type trough_proportion \
    --band B2



"""