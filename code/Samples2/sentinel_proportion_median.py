"""
Process filtered data from sentinel_proportion.py to create balanced datasets
by using median-based sampling across the distribution of each key class.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_balanced_dataset(input_csv, output_dir, target_col, n_bins=10, keep_zero=False):
    """
    Create a balanced dataset by dividing the values of target_col into n_bins bins
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
    
    # Create bins based on percentiles of target_col
    bin_edges = np.linspace(0, 100, n_bins + 1)
    percentiles = np.percentile(df_clean[target_col], bin_edges)
    
    # Plot histogram of original data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df_clean[target_col].hist(bins=30)
    plt.title(f'Original Distribution of {target_col}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    
    # Create balanced dataset
    balanced_dfs = []
    
    print(f"Creating {n_bins} bins based on percentiles of {target_col}...")
    for i in range(n_bins):
        lower = percentiles[i]
        upper = percentiles[i + 1]
        
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
    median_count = int(np.median(non_empty_counts))
    print(f"Median number of samples per bin: {median_count}")
    
    # Sample each bin to have at most median_count samples
    final_dfs = []
    for i, bin_df in enumerate(balanced_dfs):
        if len(bin_df) == 0:
            continue
            
        if len(bin_df) > median_count:
            # Random sample without replacement
            sampled_df = bin_df.sample(n=median_count, random_state=42)
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
    3. Through proportion sqrt (dry_depression + sphaignes + black_depression)
    
    Args:
        filtered_csv: Path to filtered CSV from sentinel_proportion.py
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process lichen class
    lichen_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='lichen',
        n_bins=10,
        keep_zero=False
    )
    
    # Process combined chicoutai and green_depression
    # First, add a new column to the original data
    df = pd.read_csv(filtered_csv)
    df['chicoutai_green'] = df['chicoutai'] + df['green_depression']
    df.to_csv(filtered_csv, index=False)  # Save back to add the new column
    
    chicoutai_green_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='chicoutai_green',
        n_bins=10,
        keep_zero=False
    )
    
    # Process through_proportion (sqrt transformed)
    through_sqrt_df = create_balanced_dataset(
        input_csv=filtered_csv, 
        output_dir=output_dir,
        target_col='sqrt_through_proportion',
        n_bins=10, 
        keep_zero=False
    )
    
    # Create summary plot with all three distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    lichen_df['lichen'].hist(bins=20)
    plt.title('Balanced Lichen')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    chicoutai_green_df['chicoutai_green'].hist(bins=20)
    plt.title('Balanced Chicoutai+Green')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    through_sqrt_df['sqrt_through_proportion'].hist(bins=20)
    plt.title('Balanced sqrt(Through Proportion)')
    plt.xlabel('Sqrt Proportion')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_balanced_distributions.png'))
    plt.close()
    
    # Create an analysis of the balanced datasets
    summary = {
        'Category': ['Lichen', 'Chicoutai+Green', 'sqrt(Through)'],
        'Original Samples': [
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['lichen'] > 0]),
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['chicoutai_green'] > 0]),
            len(pd.read_csv(filtered_csv)[pd.read_csv(filtered_csv)['sqrt_through_proportion'] > 0])
        ],
        'Balanced Samples': [
            len(lichen_df),
            len(chicoutai_green_df),
            len(through_sqrt_df)
        ],
        'Min Value': [
            lichen_df['lichen'].min(),
            chicoutai_green_df['chicoutai_green'].min(),
            through_sqrt_df['sqrt_through_proportion'].min()
        ],
        'Max Value': [
            lichen_df['lichen'].max(),
            chicoutai_green_df['chicoutai_green'].max(),
            through_sqrt_df['sqrt_through_proportion'].max()
        ],
        'Mean': [
            lichen_df['lichen'].mean(),
            chicoutai_green_df['chicoutai_green'].mean(),
            through_sqrt_df['sqrt_through_proportion'].mean()
        ],
        'Median': [
            lichen_df['lichen'].median(),
            chicoutai_green_df['chicoutai_green'].median(),
            through_sqrt_df['sqrt_through_proportion'].median()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'balanced_datasets_summary.csv'), index=False)
    print(f"Summary of balanced datasets saved to {os.path.join(output_dir, 'balanced_datasets_summary.csv')}")

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
    
    # Calculate bin edges based on percentiles
    bin_edges = np.linspace(0, 100, n_bins + 1)
    percentiles = np.percentile(df_nonzero[target_col], bin_edges)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Compute bin counts
    counts = []
    bin_labels = []
    for i in range(n_bins):
        lower = percentiles[i]
        upper = percentiles[i + 1]
        
        if i == n_bins - 1:  # Include upper bound in the last bin
            bin_samples = df_nonzero[(df_nonzero[target_col] >= lower) & (df_nonzero[target_col] <= upper)]
        else:
            bin_samples = df_nonzero[(df_nonzero[target_col] >= lower) & (df_nonzero[target_col] < upper)]
        
        counts.append(len(bin_samples))
        bin_labels.append(f"{lower:.3f}-{upper:.3f}")
    
    # Plot
    plt.bar(range(n_bins), counts)
    plt.xticks(range(n_bins), bin_labels, rotation=45, ha='right')
    plt.title(f'Sample Distribution Across {n_bins} Percentile Bins for {target_col}')
    plt.xlabel('Bins (Percentile-based)')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    wap = 32
    use_peat = False
    peat_suffix = "_peat" if use_peat else ""
    
    # Input/output paths
    filtered_csv = f"data/samples/selection13/regression_wap{wap}_5wd{peat_suffix}/class_proportions_WAP{wap}_filtered.csv"
    output_dir = f"data/samples/selection13/regression_wap{wap}_5wd{peat_suffix}/balanced"
    
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
