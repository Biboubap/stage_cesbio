"""
Process filtered data from sentinel_proportion.py to create balanced datasets
by using median-based sampling across the distribution of each key class.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_balanced_dataset(input_csv, output_dir, target_col, n_bins=10, keep_zero=False, quantile=0.5):
    """
    Create a balanced dataset by dividing the values of target_col into n_bins bins of equal size
    and sampling to ensure each bin has approximately the same number of samples.
    
    Args:
        input_csv: Path to input CSV with class proportions
        output_dir: Directory to save output files
        target_col: Target column to balance ('lichen', 'through_proportion', etc.)
        n_bins: Number of bins to divide the data into
        keep_zero: Whether to keep samples with zero proportion in target_col
    
    Returns:
        DataFrame with the balanced dataset
    """
    print(f"Creating balanced dataset for '{target_col}'...")
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if target_col is in the dataframe
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the input CSV")
    
    # Create output filename based on target_col
    output_csv = os.path.join(output_dir, f"balanced_{target_col.replace('/', '_')}.csv")
    
    # Filter out rows with NaN in target column
    df_clean = df.dropna(subset=[target_col])
    
    # Filter out zero values if specified
    if not keep_zero:
        df_clean = df_clean[df_clean[target_col] > 0]
    
    # Get number of samples
    n_samples = len(df_clean)
    print(f"Original dataset: {n_samples} samples")
    
    # Determine the min and max values to create fixed-size bins
    min_val = 0
    max_val = 1
    bin_width = (max_val - min_val) / n_bins
    
    # Create bins of equal size
    bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]
    
    # Plot histogram of original data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df_clean[target_col].hist(bins=30)
    plt.title(f'Original Distribution of {target_col}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    
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
    print(f"Median number of samples per bin: {quantile_count}")
    
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
    
    # Plot histogram of balanced data
    plt.subplot(1, 2, 2)
    balanced_df[target_col].hist(bins=30)
    plt.title(f'Balanced Distribution of {target_col}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"distribution_{target_col.replace('/', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Save balanced dataset
    balanced_df.to_csv(output_csv, index=False)
    print(f"Balanced dataset saved to {output_csv} ({len(balanced_df)} samples)")
    
    return balanced_df

def process_wap_data(filtered_csv, output_dir):
    """
    Process filtered data from sentinel_proportion.py to create three balanced datasets:
    1. Lichen
    2. Chicoutai and Green Depression combined
    3. Through proportion sqrt (dry_depression + sphaignes + black_depression + watered_depression)
    4. Through proportion without sqrt transform
    
    Args:
        filtered_csv: Path to filtered CSV from sentinel_proportion.py
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process combined chicoutai and green_depression
    # First, add a new column to the original data
    df = pd.read_csv(filtered_csv)
    df['chicoutai_green'] = df['chicoutai'] + df['green_depression']
    df.to_csv(filtered_csv, index=False)  # Save back to add the new column

    n_bins = 25
    
    # Process lichen class
    lichen_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='lichen',
        n_bins=n_bins,
        keep_zero=True,
        quantile=0.6
    )
    
    # Process chicoutai+green class
    chicoutai_green_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='chicoutai_green',
        n_bins=n_bins,
        keep_zero=True, 
        quantile=0.5
    )
    
    # Process through_proportion (sqrt transformed)
    through_sqrt_df = create_balanced_dataset(
        input_csv=filtered_csv, 
        output_dir=output_dir,
        target_col='sqrt_through_proportion',
        n_bins=n_bins,
        keep_zero=True,
        quantile=0.6
    )
    
    # Process through_proportion (without sqrt transform)
    through_df = create_balanced_dataset(
        input_csv=filtered_csv, 
        output_dir=output_dir,
        target_col='through_proportion',
        n_bins=n_bins,
        keep_zero=True,
        quantile=0.6
    )
    
    # Create summary plot with all distributions
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    lichen_df['lichen'].hist(bins=n_bins)
    plt.title('Balanced Lichen')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 4, 2)
    chicoutai_green_df['chicoutai_green'].hist(bins=n_bins)
    plt.title('Balanced Chicoutai+Green')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 4, 3)
    through_sqrt_df['sqrt_through_proportion'].hist(bins=n_bins)
    plt.title('Balanced sqrt(Through Proportion)')
    plt.xlabel('Sqrt Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 4, 4)
    through_df['through_proportion'].hist(bins=n_bins)
    plt.title('Balanced Through Proportion (raw)')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_balanced_distributions.png'))
    plt.close()
    
    # Create comparison between through_proportion and sqrt_through_proportion
    plt.figure(figsize=(12, 6))
    
    # Read original data from filtered_csv to ensure we're comparing same pixels
    original_df = pd.read_csv(filtered_csv)
    
    # Create a common index by using row and column coordinates to join the datasets
    through_df_with_coords = through_df.set_index(['col_s', 'row_s'])
    through_sqrt_df_with_coords = through_sqrt_df.set_index(['col_s', 'row_s'])
    
    # Find common indices (pixels present in both datasets)
    common_indices = through_df_with_coords.index.intersection(through_sqrt_df_with_coords.index)
    
    # Filter to keep only common pixels
    through_values = through_df_with_coords.loc[common_indices, 'through_proportion'].values
    sqrt_values = through_sqrt_df_with_coords.loc[common_indices, 'sqrt_through_proportion'].values
    
    # Square the sqrt values for comparison
    squared_sqrt_values = np.power(sqrt_values, 2)
    
    # Create scatter plot with equal-sized arrays
    plt.scatter(through_values, squared_sqrt_values, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.title('Comparison: through_proportion vs sqrt_through_proportion^2')
    plt.xlabel('Regular through_proportion')
    plt.ylabel('sqrt_through_proportion^2')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'through_proportion_comparison.png'))
    plt.close()
    
    # Create an analysis of the balanced datasets
    summary = {
        'Category': ['Lichen', 'Chicoutai+Green', 'sqrt(Through)', 'Through (raw)'],
        'Original Samples': [
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['lichen'] > 0]),
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['chicoutai_green'] > 0]),
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['sqrt_through_proportion'] > 0]),
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['through_proportion'] > 0])
        ],
        'Balanced Samples': [
            len(lichen_df),
            len(chicoutai_green_df),
            len(through_sqrt_df),
            len(through_df)
        ],
        'Min Value': [
            lichen_df['lichen'].min(),
            chicoutai_green_df['chicoutai_green'].min(),
            through_sqrt_df['sqrt_through_proportion'].min(),
            through_df['through_proportion'].min()
        ],
        'Max Value': [
            lichen_df['lichen'].max(),
            chicoutai_green_df['chicoutai_green'].max(),
            through_sqrt_df['sqrt_through_proportion'].max(),
            through_df['through_proportion'].max()
        ],
        'Mean': [
            lichen_df['lichen'].mean(),
            chicoutai_green_df['chicoutai_green'].mean(),
            through_sqrt_df['sqrt_through_proportion'].mean(),
            through_df['through_proportion'].mean()
        ],
        'Median': [
            lichen_df['lichen'].median(),
            chicoutai_green_df['chicoutai_green'].median(),
            through_sqrt_df['sqrt_through_proportion'].median(),
            through_df['through_proportion'].median()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'balanced_datasets_summary.csv'), index=False)
    print(f"Summary of balanced datasets saved to {os.path.join(output_dir, 'balanced_datasets_summary.csv')}")
    
    # Create a combined dataset containing all balanced samples
    print("Creating combined datasets...")
    
    # Create a combined dataset with sqrt_through_proportion
    combined_sqrt_df = pd.concat([lichen_df, chicoutai_green_df, through_sqrt_df], axis=0)
    combined_sqrt_df = combined_sqrt_df.drop_duplicates(subset=['col_s', 'row_s'])
    combined_sqrt_path = os.path.join(output_dir, 'balanced_combined_sqrt.csv')
    combined_sqrt_df.to_csv(combined_sqrt_path, index=False)
    
    # Create a combined dataset with regular through_proportion
    combined_raw_df = pd.concat([lichen_df, chicoutai_green_df, through_df], axis=0)
    combined_raw_df = combined_raw_df.drop_duplicates(subset=['col_s', 'row_s'])
    combined_raw_path = os.path.join(output_dir, 'balanced_combined_raw.csv')
    combined_raw_df.to_csv(combined_raw_path, index=False)
    
    # Also create the original balanced_combined.csv for backward compatibility
    combined_df = combined_sqrt_df.copy()
    combined_path = os.path.join(output_dir, 'balanced_combined.csv')
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined datasets created:")
    print(f"- With sqrt transform: {combined_sqrt_path} ({len(combined_sqrt_df)} samples)")
    print(f"- Without sqrt transform: {combined_raw_path} ({len(combined_raw_df)} samples)")
    print(f"- Legacy combined file: {combined_path}")
    print(f"- Lichen samples: {len(lichen_df)}")
    print(f"- Chicoutai+Green samples: {len(chicoutai_green_df)}")
    print(f"- sqrt(Through) samples: {len(through_sqrt_df)}")
    print(f"- Through (raw) samples: {len(through_df)}")
    
    # Add combined dataset to summary
    summary['Category'].extend(['Combined Sqrt (Unique)', 'Combined Raw (Unique)'])
    summary['Original Samples'].extend([len(pd.read_csv(filtered_csv)), len(pd.read_csv(filtered_csv))])
    summary['Balanced Samples'].extend([len(combined_sqrt_df), len(combined_raw_df)])
    summary['Min Value'].extend([None, None])  # Not applicable
    summary['Max Value'].extend([None, None])  # Not applicable
    summary['Mean'].extend([None, None])       # Not applicable
    summary['Median'].extend([None, None])     # Not applicable
    
    # Update summary CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'balanced_datasets_summary.csv'), index=False)

def visualize_bins(input_csv, target_col, output_path, n_bins=10):
    """
    Visualize how the data is distributed across bins before and after balancing.
    
    Args:
        input_csv: Path to CSV file
        target_col: Target column to analyze
        output_path: Path to save the visualization
        n_bins: Number of bins
    """
    # Load data
    df = pd.read_csv(input_csv)
    
    # Filter non-zero values
    df_nonzero = df[df[target_col] > 0]
    
    # Determine min and max values for equal-sized bins
    min_val = df_nonzero[target_col].min()
    max_val = df_nonzero[target_col].max()
    bin_width = (max_val - min_val) / n_bins
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Compute bin counts
    counts = []
    bin_labels = []
    for i in range(n_bins):
        lower = min_val + i * bin_width
        upper = min_val + (i + 1) * bin_width
        
        if i == n_bins - 1:  # Include upper bound in the last bin
            bin_samples = df_nonzero[(df_nonzero[target_col] >= lower) & (df_nonzero[target_col] <= upper)]
        else:
            bin_samples = df_nonzero[(df_nonzero[target_col] >= lower) & (df_nonzero[target_col] < upper)]
        
        counts.append(len(bin_samples))
        bin_labels.append(f"{lower:.3f}-{upper:.3f}")
    
    # Plot
    plt.bar(range(n_bins), counts)
    plt.xticks(range(n_bins), bin_labels, rotation=45, ha='right')
    plt.title(f'Sample Distribution Across {n_bins} Equal-Width Bins for {target_col}')
    plt.xlabel('Bins (Equal Width)')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    wap = 23

    use_peat = True
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""

    # Input/output paths
    filtered_csv = f"data/samples/selection15/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/class_proportions_WAP{wap}_filtered.csv"
    output_dir = f"data/samples/selection15/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/balanced"

    # Process data to create balanced datasets
    process_wap_data(filtered_csv, output_dir)
    
    # Create additional visualization for bin distribution before balancing
    visualize_bins(
        input_csv=filtered_csv,
        target_col='lichen',
        output_path=os.path.join(output_dir, 'lichen_bins_distribution.png')
    )
    
    visualize_bins(
        input_csv=filtered_csv,
        target_col='chicoutai_green',
        output_path=os.path.join(output_dir, 'chicoutai_green_bins_distribution.png')
    )
    
    visualize_bins(
        input_csv=filtered_csv,
        target_col='sqrt_through_proportion',
        output_path=os.path.join(output_dir, 'sqrt_through_bins_distribution.png')
    )
    
    print("All processing completed!")
