#!/usr/bin/env python3
"""
Study Band Dispersion

This script analyzes the relationship between Sentinel-2 spectral bands (B2, B3) and
lichen proportion in different sites. It creates scatter plots showing median values
and standard deviations across bins of spectral values.

Usage:
  python study_band_dispersion.py --balanced-5m path/to/balanced_5m_file.json 
                                --balanced-10m path/to/balanced_10m_file.json
                                --output-dir path/to/output_directory
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Add parent directory to path to import functions from other regression modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_arguments():
    """
    Parse command line arguments for the band dispersion analysis.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Analyze relationship between spectral bands and lichen proportion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--balanced-5m', required=True,
                        help='Path to the balanced 5m lichen JSON file')
    parser.add_argument('--balanced-10m', required=True,
                        help='Path to the balanced 10m lichen JSON file')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save the output plots')
    
    # Optional arguments
    parser.add_argument('--n-bins', type=int, default=50,
                        help='Number of bins for analysis')
    parser.add_argument('--band-min', type=float, default=0,
                        help='Minimum band value for binning')
    parser.add_argument('--band-max', type=float, default=2000,
                        help='Maximum band value for binning')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse axes: plot band values on X-axis and proportion on Y-axis')
    parser.add_argument('--no-bins', action='store_true',
                        help='Use raw data points without binning')
    
    return parser.parse_args()

def load_json_data(json_path):
    """
    Load lichen proportion data from a JSON file.
    
    Args:
        json_path: Path to the JSON file with balanced samples
        
    Returns:
        DataFrame containing features and lichen proportions
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
        
        # Extract lichen proportion
        if 'proportions' in item and 'lichen_proportion' in item['proportions']:
            row['lichen_proportion'] = item['proportions']['lichen_proportion']
            
        # Extract site information
        if 'source' in item:
            row['source'] = item['source']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df

def create_binned_statistics(df, band_col, proportion_col, n_bins=50, proportion_min=0, proportion_max=1):
    """
    Create binned statistics (median and standard deviation) for proportion vs band.
    
    Args:
        df: DataFrame containing the data
        band_col: Column name for the band values
        proportion_col: Column name for the proportion values
        n_bins: Number of bins to create
        proportion_min: Minimum proportion value for binning (0-1 scale)
        proportion_max: Maximum proportion value for binning (0-1 scale)
        
    Returns:
        Dictionary with bin centers, medians, and standard deviations
    """
    # Create bins for proportion values
    bins = np.linspace(proportion_min, proportion_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize arrays for results
    medians = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    
    # Calculate bin statistics
    for i in range(n_bins):
        # Get samples in this bin
        if i == n_bins - 1:  # Include upper bound in the last bin
            mask = (df[proportion_col] >= bins[i]) & (df[proportion_col] <= bins[i+1])
        else:
            mask = (df[proportion_col] >= bins[i]) & (df[proportion_col] < bins[i+1])
            
        bin_values = df.loc[mask, band_col].values
        counts[i] = len(bin_values)
        
        if len(bin_values) > 0:
            medians[i] = np.median(bin_values)
            stds[i] = np.std(bin_values)
    
    return {
        'bin_centers': bin_centers,
        'medians': medians,
        'stds': stds,
        'counts': counts
    }

def fit_linear_regression(x, y):
    """
    Fit a linear regression model and calculate metrics.
    
    Args:
        x: Array of x values (proportion)
        y: Array of y values (band values)
        
    Returns:
        dict: Regression results including model, coefficients, R², RMSE
    """
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = np.array(x[mask]).reshape(-1, 1)
    y_clean = np.array(y[mask])
    
    if len(x_clean) < 2:
        return {
            'model': None,
            'slope': np.nan,
            'intercept': np.nan,
            'r2': np.nan,
            'rmse': np.nan
        }
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x_clean, y_clean)
    
    # Make predictions
    y_pred = model.predict(x_clean)
    
    # Calculate metrics
    r2 = r2_score(y_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
    
    return {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'rmse': rmse
    }

def plot_global_proportion_vs_band(df, band_col, proportion_col, output_path, 
                                 n_bins=50, proportion_min=0, proportion_max=1,
                                 resolution='5m', reverse=False, no_bins=False):
    """
    Create a global scatter plot of proportion values vs band with binned statistics and linear regression.
    
    Args:
        df: DataFrame containing the data
        band_col: Column name for the band values
        proportion_col: Column name for the proportion values
        output_path: Path to save the output plot
        n_bins: Number of bins to create
        proportion_min: Minimum proportion value for binning (0-1 scale)
        proportion_max: Maximum proportion value for binning (0-1 scale)
        resolution: Data resolution for the plot title
        reverse: If True, plot band on X-axis and proportion on Y-axis
        no_bins: If True, use raw data points instead of binned statistics
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    if no_bins:
        # Use raw data points without binning
        if reverse:
            # Band values on X-axis, proportion on Y-axis
            x = df[band_col].values
            y = df[proportion_col].values * 100  # Convert to percentage
            
            plt.scatter(x, y, s=2, alpha=0.1, color='blue', label=f"All sites (n={len(df)})")
            
            # Perform linear regression on raw data
            mask = ~np.isnan(x) & ~np.isnan(y/100)  # Use raw proportions for regression
            x_clean = x[mask].reshape(-1, 1)
            y_clean = (y/100)[mask].reshape(-1, 1)  # Convert back to 0-1 for regression
            
            # Fit model
            model = LinearRegression()
            model.fit(x_clean, y_clean)
            
            # Calculate metrics
            y_pred = model.predict(x_clean)
            r2 = r2_score(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
            
            # Plot regression line
            from_min = float(df[band_col].min())
            from_max = float(df[band_col].max())
            x_line = np.array([from_min, from_max])
            y_line = model.predict(x_line.reshape(-1, 1)).flatten() * 100  # Convert to percentage
            plt.plot(x_line, y_line, 'r--', 
                    label=f"Linear fit: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")
            
            # Set axis labels
            plt.xlabel(f"{band_col} Value")
            plt.ylabel("Lichen Proportion (%)")
            title = f"Relationship between {band_col} and Lichen Proportion ({resolution})\nR² = {r2:.3f}, RMSE = {rmse:.2f}"
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
        
        else:
            # Proportion on X-axis, band values on Y-axis
            x = df[proportion_col].values * 100  # Convert to percentage for x-axis
            y = df[band_col].values
            
            plt.scatter(x, y, s=2, alpha=0.1, color='blue', label=f"All sites (n={len(df)})")
            
            # Perform linear regression on raw data
            mask = ~np.isnan(x/100) & ~np.isnan(y)  # Use raw proportions for regression
            x_clean = (x/100)[mask].reshape(-1, 1)  # Convert back to 0-1 for regression
            y_clean = y[mask].reshape(-1, 1)
            
            # Fit model
            model = LinearRegression()
            model.fit(x_clean, y_clean)
            
            # Calculate metrics
            y_pred = model.predict(x_clean)
            r2 = r2_score(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
            
            # Plot regression line
            x_line = np.array([proportion_min, proportion_max]) * 100  # Convert to percentage
            y_line = model.predict(np.array([[proportion_min], [proportion_max]])).flatten()
            plt.plot(x_line, y_line, 'r--', 
                    label=f"Linear fit: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")
            
            # Set axis labels
            plt.xlabel("Lichen Proportion (%)")
            plt.ylabel(f"{band_col} Value")
            title = f"Relationship between Lichen Proportion and {band_col} ({resolution})\nR² = {r2:.3f}, RMSE = {rmse:.2f}"
            plt.gca().xaxis.set_major_formatter(PercentFormatter())
    
    else:
        # Use existing binning logic
        # Get binned statistics - determine which variable to bin by
        if reverse:
            # Bin by band values
            from_min = float(df[band_col].min())
            from_max = float(df[band_col].max())
            bin_edges = np.linspace(from_min, from_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate statistics for each bin
            medians = np.full(n_bins, np.nan)
            stds = np.full(n_bins, np.nan)
            
            for i in range(n_bins):
                if i == n_bins - 1:  # Include upper bound in the last bin
                    mask = (df[band_col] >= bin_edges[i]) & (df[band_col] <= bin_edges[i+1])
                else:
                    mask = (df[band_col] >= bin_edges[i]) & (df[band_col] < bin_edges[i+1])
                    
                bin_values = df.loc[mask, proportion_col].values
                
                if len(bin_values) > 0:
                    medians[i] = np.median(bin_values)
                    stds[i] = np.std(bin_values)
            
            # Plot median with error bars - band on X, proportion on Y
            plt.errorbar(
                bin_centers,  # Band values on X-axis
                medians * 100,  # Proportion values on Y-axis (as percentage)
                yerr=stds * 100,  # Standard deviation of proportions
                fmt='o-',
                label=f"All sites (n={len(df)})",
                color='blue',
                alpha=0.7,
                capsize=5
            )
            
            # Perform linear regression
            reg_results = fit_linear_regression(bin_centers, medians)
            
        else:
            # Use existing function for proportion on X-axis
            stats = create_binned_statistics(
                df, band_col, proportion_col, n_bins, proportion_min, proportion_max
            )
            
            # Plot median with error bars
            plt.errorbar(
                stats['bin_centers'] * 100,  # Convert to percentage for x-axis
                stats['medians'],  # Band value for y-axis
                yerr=stats['stds'],  # Standard deviation of band values
                fmt='o-',
                label=f"All sites (n={len(df)})",
                color='blue',
                alpha=0.7,
                capsize=5
            )
            
            # Perform linear regression
            reg_results = fit_linear_regression(stats['bin_centers'], stats['medians'])
        
        # Plot regression line if successful
        if not np.isnan(reg_results['slope']):
            if reverse:
                x_line = np.array([from_min, from_max])
                y_line = reg_results['slope'] * x_line + reg_results['intercept']
                y_line = y_line * 100  # Convert to percentage
            else:
                x_line = np.array([proportion_min, proportion_max]) * 100  # percentage scale
                y_line = reg_results['slope'] * np.array([proportion_min, proportion_max]) + reg_results['intercept']
                
            plt.plot(x_line, y_line, 'r--', 
                    label=f"Linear fit: y = {reg_results['slope']:.2f}x + {reg_results['intercept']:.2f}")
        
        # Set labels and title
        band_name = band_col.split('_')[-1] if '_' in band_col else band_col
        
        if reverse:
            plt.xlabel(f"{band_name} Value")
            plt.ylabel("Lichen Proportion (%)")
            title = f"Relationship between {band_name} and Lichen Proportion ({resolution})"
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(PercentFormatter())
        else:
            plt.xlabel("Lichen Proportion (%)")
            plt.ylabel(f"{band_name} Value")
            title = f"Relationship between Lichen Proportion and {band_name} ({resolution})"
            # Format x-axis as percentage
            plt.gca().xaxis.set_major_formatter(PercentFormatter())
    
    plt.title(title)
    
    # Add grid and legend
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Global plot saved to {output_path}")

def plot_site_subplots_proportion_vs_band(df, band_col, proportion_col, output_path, 
                                        n_bins=50, proportion_min=0, proportion_max=1,
                                        resolution='5m', reverse=False, no_bins=False):
    """
    Create subplots of proportion values vs band for each site with linear regression.
    
    Args:
        df: DataFrame containing the data
        band_col: Column name for the band values
        proportion_col: Column name for the proportion values
        output_path: Path to save the output plot
        n_bins: Number of bins to create
        proportion_min: Minimum proportion value for binning (0-1 scale)
        proportion_max: Maximum proportion value for binning (0-1 scale)
        resolution: Data resolution for the plot title
        reverse: If True, plot band on X-axis and proportion on Y-axis
        no_bins: If True, use raw data points instead of binned statistics
    """
    # Get unique sites
    if 'source' not in df.columns:
        print("No 'source' column in DataFrame, cannot create site subplots")
        return
        
    sites = sorted(df['source'].unique())
    n_sites = len(sites)
    
    if n_sites < 1:
        print("No sites found in the data")
        return
    
    # Determine subplot layout - 2 rows, 3 columns (or fewer if less sites)
    n_cols = min(3, n_sites)
    n_rows = (n_sites + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 10), constrained_layout=True)
    
    # Convert axs to 2D array if it's not already
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Plot for each site
    band_name = band_col.split('_')[-1] if '_' in band_col else band_col
    
    for i, site in enumerate(sites):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        # Get data for this site
        site_df = df[df['source'] == site]
        
        # Skip sites with too few samples
        if len(site_df) < 20:
            ax.text(0.5, 0.5, f"Not enough samples\n{site}: {len(site_df)} samples", 
                   ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        if no_bins:
            # Use raw data points without binning
            if reverse:
                # Band on X-axis, proportion on Y-axis
                x = site_df[band_col].values
                y = site_df[proportion_col].values * 100  # Convert to percentage
                
                ax.scatter(x, y, s=2, alpha=0.3, color='blue')
                
                # Linear regression on raw data
                mask = ~np.isnan(x) & ~np.isnan(y/100)
                if np.sum(mask) > 1:  # Need at least 2 points for regression
                    x_clean = x[mask].reshape(-1, 1)
                    y_clean = (y/100)[mask].reshape(-1, 1)
                    
                    model = LinearRegression()
                    model.fit(x_clean, y_clean)
                    
                    # Calculate metrics
                    y_pred = model.predict(x_clean)
                    r2 = r2_score(y_clean, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
                    
                    # Plot regression line
                    from_min = float(site_df[band_col].min())
                    from_max = float(site_df[band_col].max())
                    x_line = np.array([from_min, from_max])
                    y_line = model.predict(x_line.reshape(-1, 1)).flatten() * 100
                    ax.plot(x_line, y_line, 'r--')
                    
                    # Format y-axis as percentage
                    ax.yaxis.set_major_formatter(PercentFormatter())
                    
                    # Extract site name without extension
                    site_name = os.path.splitext(site)[0]
                    
                    # Set title with R² and RMSE
                    title = f"{site_name} (n={len(site_df)})\nR² = {r2:.3f}, RMSE = {rmse:.2f}"
                else:
                    title = f"{site} (n={len(site_df)})\nNot enough valid data for regression"
            
            else:
                # Proportion on X-axis, band on Y-axis
                x = site_df[proportion_col].values * 100  # Convert to percentage
                y = site_df[band_col].values
                
                ax.scatter(x, y, s=2, alpha=0.3, color='blue')
                
                # Linear regression on raw data
                mask = ~np.isnan(x/100) & ~np.isnan(y)
                if np.sum(mask) > 1:  # Need at least 2 points for regression
                    x_clean = (x/100)[mask].reshape(-1, 1)
                    y_clean = y[mask].reshape(-1, 1)
                    
                    model = LinearRegression()
                    model.fit(x_clean, y_clean)
                    
                    # Calculate metrics
                    y_pred = model.predict(x_clean)
                    r2 = r2_score(y_clean, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
                    
                    # Plot regression line
                    x_line = np.array([proportion_min, proportion_max]) * 100
                    y_line = model.predict(np.array([[proportion_min], [proportion_max]])).flatten()
                    ax.plot(x_line, y_line, 'r--')
                    
                    # Format x-axis as percentage
                    ax.xaxis.set_major_formatter(PercentFormatter())
                    
                    # Extract site name without extension
                    site_name = os.path.splitext(site)[0]
                    
                    # Set title with R² and RMSE
                    title = f"{site_name} (n={len(site_df)})\nR² = {r2:.3f}, RMSE = {rmse:.2f}"
                else:
                    title = f"{site} (n={len(site_df)})\nNot enough valid data for regression"
        
        else:
            # Use existing binning logic
            if reverse:
                # Bin by band values
                from_min = float(site_df[band_col].min())
                from_max = float(site_df[band_col].max())
                bin_edges = np.linspace(from_min, from_max, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Calculate statistics for each bin
                medians = np.full(n_bins, np.nan)
                stds = np.full(n_bins, np.nan)
                
                for j in range(n_bins):
                    if j == n_bins - 1:  # Include upper bound in the last bin
                        mask = (site_df[band_col] >= bin_edges[j]) & (site_df[band_col] <= bin_edges[j+1])
                    else:
                        mask = (site_df[band_col] >= bin_edges[j]) & (site_df[band_col] < bin_edges[j+1])
                        
                    bin_values = site_df.loc[mask, proportion_col].values
                    
                    if len(bin_values) > 0:
                        medians[j] = np.median(bin_values)
                        stds[j] = np.std(bin_values)
                
                # Plot median with error bars - band on X, proportion on Y
                ax.errorbar(
                    bin_centers,  # Band values on X-axis
                    medians * 100,  # Proportion values on Y-axis (as percentage)
                    yerr=stds * 100,  # Standard deviation of proportions
                    fmt='o-',
                    color='blue',
                    alpha=0.7,
                    capsize=3,
                    markersize=4
                )
                
                # Perform linear regression
                reg_results = fit_linear_regression(bin_centers, medians)
                
                # Plot regression line if successful
                if not np.isnan(reg_results['slope']):
                    x_line = np.array([from_min, from_max])
                    y_line = reg_results['slope'] * x_line + reg_results['intercept']
                    y_line = y_line * 100  # Convert to percentage
                    ax.plot(x_line, y_line, 'r--')
            
            else:
                # Get binned statistics
                stats = create_binned_statistics(
                    site_df, band_col, proportion_col, n_bins, proportion_min, proportion_max
                )
                
                # Plot median with error bars
                ax.errorbar(
                    stats['bin_centers'] * 100,  # Convert to percentage for x-axis
                    stats['medians'],  # Band value for y-axis
                    yerr=stats['stds'],  # Standard deviation of band values
                    fmt='o-',
                    color='blue',
                    alpha=0.7,
                    capsize=3,
                    markersize=4
                )
                
                # Perform linear regression
                reg_results = fit_linear_regression(stats['bin_centers'], stats['medians'])
                
                # Plot regression line if successful
                if not np.isnan(reg_results['slope']):
                    x_line = np.array([proportion_min, proportion_max]) * 100  # percentage scale
                    y_line = reg_results['slope'] * np.array([proportion_min, proportion_max]) + reg_results['intercept']
                    ax.plot(x_line, y_line, 'r--')
            
            # Extract site name without extension
            site_name = os.path.splitext(site)[0]
            
            # Set title with R² and RMSE
            title = f"{site_name} (n={len(site_df)})"
            if not np.isnan(reg_results['r2']):
                title += f"\nR² = {reg_results['r2']:.3f}, RMSE = {reg_results['rmse']:.2f}"
        
        ax.set_title(title)
        
        # Set appropriate formatter and labels based on orientation
        if reverse:
            # Only add x-label for bottom row
            if row == n_rows - 1:
                ax.set_xlabel(f"{band_name} Value")
                
            # Only add y-label for first column
            if col == 0:
                ax.set_ylabel("Lichen Proportion (%)")
        else:
            # Only add x-label for bottom row
            if row == n_rows - 1:
                ax.set_xlabel("Lichen Proportion (%)")
                
            # Only add y-label for first column
            if col == 0:
                ax.set_ylabel(f"{band_name} Value")
        
        # Add grid
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_sites, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].set_visible(False)
    
    # Add overall title
    if reverse:
        fig.suptitle(f"Relationship between {band_name} and Lichen Proportion by Site ({resolution})", 
                    fontsize=16)
    else:
        fig.suptitle(f"Relationship between Lichen Proportion and {band_name} by Site ({resolution})", 
                    fontsize=16)
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"Site subplot saved to {output_path}")

def process_resolution_data(json_path, output_dir, n_bins, band_min, band_max, resolution, reverse=False, no_bins=False):
    """
    Process data for a specific resolution and create plots.
    
    Args:
        json_path: Path to the JSON file with balanced samples
        output_dir: Directory to save output plots
        n_bins: Number of bins for analysis
        band_min: Minimum band value for binning
        band_max: Maximum band value for binning
        resolution: Data resolution string (e.g., '5m', '10m')
        reverse: If True, plot band on X-axis and proportion on Y-axis
        no_bins: If True, use raw data points instead of binned statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_json_data(json_path)
    
    # Find B2 and B3 columns
    b2_col = next((col for col in df.columns if 'B2' in col), None)
    b3_col = next((col for col in df.columns if 'B3' in col), None)
    
    if b2_col is None or b3_col is None:
        print(f"Warning: Could not find B2 or B3 columns in the data")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print(f"Using column '{b2_col}' for B2 and '{b3_col}' for B3")
    
    # Determine file name prefix based on axis orientation and binning
    orientation = "band_vs_lichen" if reverse else "lichen_vs_band"
    binning = "raw" if no_bins else "binned"
    
    # Create global plots for B2 and B3
    print(f"Creating global plots for {resolution} ({binning})...")
    plot_global_proportion_vs_band(
        df=df,
        band_col=b2_col,
        proportion_col='lichen_proportion',
        output_path=os.path.join(output_dir, f'global_{orientation}_B2_{resolution}_{binning}.png'),
        n_bins=n_bins,
        proportion_min=0,
        proportion_max=1,
        resolution=resolution,
        reverse=reverse,
        no_bins=no_bins
    )
    
    plot_global_proportion_vs_band(
        df=df,
        band_col=b3_col,
        proportion_col='lichen_proportion',
        output_path=os.path.join(output_dir, f'global_{orientation}_B3_{resolution}_{binning}.png'),
        n_bins=n_bins,
        proportion_min=0,
        proportion_max=1,
        resolution=resolution,
        reverse=reverse,
        no_bins=no_bins
    )
    
    # Create site subplot plots for B2 and B3
    print(f"Creating site subplots for {resolution} ({binning})...")
    plot_site_subplots_proportion_vs_band(
        df=df,
        band_col=b2_col,
        proportion_col='lichen_proportion',
        output_path=os.path.join(output_dir, f'sites_{orientation}_B2_{resolution}_{binning}.png'),
        n_bins=n_bins,
        proportion_min=0,
        proportion_max=1,
        resolution=resolution,
        reverse=reverse,
        no_bins=no_bins
    )
    
    plot_site_subplots_proportion_vs_band(
        df=df,
        band_col=b3_col,
        proportion_col='lichen_proportion',
        output_path=os.path.join(output_dir, f'sites_{orientation}_B3_{resolution}_{binning}.png'),
        n_bins=n_bins,
        proportion_min=0,
        proportion_max=1,
        resolution=resolution,
        reverse=reverse,
        no_bins=no_bins
    )

def main():
    """
    Main function to analyze relationship between spectral bands and lichen proportion.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process 5m data
    print("\nProcessing 5m resolution data...")
    process_resolution_data(
        args.balanced_5m,
        args.output_dir,
        args.n_bins,
        args.band_min,
        args.band_max,
        '5m',
        args.reverse,
        args.no_bins
    )
    
    # Process 10m data
    print("\nProcessing 10m resolution data...")
    process_resolution_data(
        args.balanced_10m,
        args.output_dir,
        args.n_bins,
        args.band_min,
        args.band_max,
        '10m',
        args.reverse,
        args.no_bins
    )
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()

"""
python /home/lcousin/stage_cesbio/code/final_codes/regression/study_5m_dispersion/study_band_dispersion.py \
    --balanced-5m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_5m/balanced_lichen_proportion.json \
    --balanced-10m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_10m/balanced_lichen_proportion.json \
    --output-dir /home/lcousin/stage_cesbio/data/study_5m_dispersion \
    --n-bins 1000 \
    --reverse


python /home/lcousin/stage_cesbio/code/final_codes/regression/study_5m_dispersion/study_band_dispersion.py \
    --balanced-5m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_5m/balanced_lichen_proportion.json \
    --balanced-10m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_10m/balanced_lichen_proportion.json \
    --output-dir /home/lcousin/stage_cesbio/data/study_5m_dispersion_2 \
    --n-bins 25 

python /home/lcousin/stage_cesbio/code/final_codes/regression/study_5m_dispersion/study_band_dispersion.py \
    --balanced-5m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_5m/balanced_lichen_proportion.json \
    --balanced-10m /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/balanced_10m/balanced_lichen_proportion.json \
    --output-dir /home/lcousin/stage_cesbio/data/study_5m_dispersion_3 \
    --reverse \
    --no-bins

"""