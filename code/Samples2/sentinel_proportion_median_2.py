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
    Process filtered data from sentinel_proportion.py to create five balanced datasets:
    1. Pure_Lichen
    2. Degraded_Lichen
    3. Merged_Lichen (Pure_Lichen + Degraded_Lichen)
    4. Green
    5. Through proportion (without sqrt transform)
    
    Args:
        filtered_csv: Path to filtered CSV from sentinel_proportion.py
        output_dir: Directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(filtered_csv)
    
    # Create merged_lichen column (Pure_Lichen + Degraded_Lichen)
    if 'Pure_Lichen' in df.columns and 'Degraded_Lichen' in df.columns:
        df['Merged_Lichen'] = df['Pure_Lichen'] + df['Degraded_Lichen']
        df.to_csv(filtered_csv, index=False)  # Save back to add the new column

    n_bins = 25
    
    # Process Pure_Lichen class
    pure_lichen_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='Pure_Lichen',
        n_bins=n_bins,
        keep_zero=True,
        quantile=0.5
    )
    
    # Process Degraded_Lichen class
    degraded_lichen_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='Degraded_Lichen',
        n_bins=n_bins,
        keep_zero=True,
        quantile=0.6
    )
    
    # Process Merged_Lichen class (Pure_Lichen + Degraded_Lichen)
    merged_lichen_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='Merged_Lichen',
        n_bins=n_bins,
        keep_zero=True,
        quantile=0.6
    )
    
    # Process Green class
    green_df = create_balanced_dataset(
        input_csv=filtered_csv,
        output_dir=output_dir,
        target_col='Green',
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
        quantile=0.5
    )
    
    # Create summary plot with all distributions
    plt.figure(figsize=(20, 8))
    
    plt.subplot(1, 5, 1)
    pure_lichen_df['Pure_Lichen'].hist(bins=n_bins)
    plt.title('Balanced Pure_Lichen')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 5, 2)
    degraded_lichen_df['Degraded_Lichen'].hist(bins=n_bins)
    plt.title('Balanced Degraded_Lichen')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 5, 3)
    merged_lichen_df['Merged_Lichen'].hist(bins=n_bins)
    plt.title('Balanced Merged_Lichen')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 5, 4)
    green_df['Green'].hist(bins=n_bins)
    plt.title('Balanced Green')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.subplot(1, 5, 5)
    through_df['through_proportion'].hist(bins=n_bins)
    plt.title('Balanced Through Proportion')
    plt.xlabel('Proportion')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_balanced_distributions.png'))
    plt.close()
    
    # Create an analysis of the balanced datasets
    summary = {
        'Category': ['Pure_Lichen', 'Degraded_Lichen', 'Merged_Lichen', 'Green', 'Through'],
        'Original Samples': [
            len(df[df['Pure_Lichen'] > 0]),
            len(df[df['Degraded_Lichen'] > 0]),
            len(df[df['Merged_Lichen'] > 0]),
            len(df[df['Green'] > 0]),
            len(df[df['through_proportion'] > 0])
        ],
        'Balanced Samples': [
            len(pure_lichen_df),
            len(degraded_lichen_df),
            len(merged_lichen_df),
            len(green_df),
            len(through_df)
        ],
        'Min Value': [
            pure_lichen_df['Pure_Lichen'].min(),
            degraded_lichen_df['Degraded_Lichen'].min(),
            merged_lichen_df['Merged_Lichen'].min(),
            green_df['Green'].min(),
            through_df['through_proportion'].min()
        ],
        'Max Value': [
            pure_lichen_df['Pure_Lichen'].max(),
            degraded_lichen_df['Degraded_Lichen'].max(),
            merged_lichen_df['Merged_Lichen'].max(),
            green_df['Green'].max(),
            through_df['through_proportion'].max()
        ],
        'Mean': [
            pure_lichen_df['Pure_Lichen'].mean(),
            degraded_lichen_df['Degraded_Lichen'].mean(),
            merged_lichen_df['Merged_Lichen'].mean(),
            green_df['Green'].mean(),
            through_df['through_proportion'].mean()
        ],
        'Median': [
            pure_lichen_df['Pure_Lichen'].median(),
            degraded_lichen_df['Degraded_Lichen'].median(),
            merged_lichen_df['Merged_Lichen'].median(),
            green_df['Green'].median(),
            through_df['through_proportion'].median()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'balanced_datasets_summary.csv'), index=False)
    print(f"Summary of balanced datasets saved to {os.path.join(output_dir, 'balanced_datasets_summary.csv')}")
    
    # Create a combined dataset containing all balanced samples
    print("Creating combined datasets...")
    
    # Create a combined dataset with all classes
    combined_df = pd.concat([pure_lichen_df, degraded_lichen_df, merged_lichen_df, green_df, through_df], axis=0)
    combined_df = combined_df.drop_duplicates(subset=['col_s', 'row_s'])
    combined_path = os.path.join(output_dir, 'balanced_combined.csv')
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined dataset created: {combined_path} ({len(combined_df)} samples)")
    print(f"- Pure_Lichen samples: {len(pure_lichen_df)}")
    print(f"- Degraded_Lichen samples: {len(degraded_lichen_df)}")
    print(f"- Merged_Lichen samples: {len(merged_lichen_df)}")
    print(f"- Green samples: {len(green_df)}")
    print(f"- Through samples: {len(through_df)}")
    
    # Add combined dataset to summary
    summary['Category'].append('Combined (Unique)')
    summary['Original Samples'].append(len(df))
    summary['Balanced Samples'].append(len(combined_df))
    summary['Min Value'].append(None)  # Not applicable
    summary['Max Value'].append(None)  # Not applicable
    summary['Mean'].append(None)       # Not applicable
    summary['Median'].append(None)     # Not applicable
    
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
    wap = 32

    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""

    
    filtered_csv = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/class_proportions_WAP{wap}_filtered.csv"
    output_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/balanced"

    # Process data to create balanced datasets
    process_wap_data(filtered_csv, output_dir)
    
    # Create additional visualization for bin distribution before balancing
    visualize_bins(
        input_csv=filtered_csv,
        target_col='Pure_Lichen',
        output_path=os.path.join(output_dir, 'pure_lichen_bins_distribution.png')
    )
    
    visualize_bins(
        input_csv=filtered_csv,
        target_col='Degraded_Lichen',
        output_path=os.path.join(output_dir, 'degraded_lichen_bins_distribution.png')
    )
    
    visualize_bins(
        input_csv=filtered_csv,
        target_col='Merged_Lichen',
        output_path=os.path.join(output_dir, 'merged_lichen_bins_distribution.png')
    )
    
    visualize_bins(
        input_csv=filtered_csv,
        target_col='Green',
        output_path=os.path.join(output_dir, 'green_bins_distribution.png')
    )
    
    visualize_bins(
        input_csv=filtered_csv,
        target_col='through_proportion',
        output_path=os.path.join(output_dir, 'through_bins_distribution.png')
    )
    
   