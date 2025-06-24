"""
Evaluate regression results by comparing theoretical proportions from one TIF
with predicted proportions from another TIF.

This script:
1. Loads theoretical (ground truth) and predicted proportion TIFs
2. Compares them pixel-by-pixel
3. Calculates metrics (R², RMSE, Pearson's r)
4. Creates scatter plots of real vs predicted values for each class
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import argparse

def load_tif_bands(tif_path):
    """
    Load all bands from a multi-band TIF file
    
    Args:
        tif_path: Path to the TIF file
        
    Returns:
        bands_data: Dictionary with band names as keys and band data as values
        band_names: List of band names in order
    """
    ds = gdal.Open(tif_path)
    if ds is None:
        raise ValueError(f"Could not open TIF file: {tif_path}")
    
    # Get band descriptions from metadata if available
    band_names = []
    bands_data = {}
    
    # Get metadata
    metadata = ds.GetMetadata()
    
    # Try to get band names from metadata
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        band_desc = band.GetDescription()
        
        # If band has a description, use that; otherwise try metadata or use a default name
        if band_desc:
            band_name = band_desc
        elif f'BAND_{i}_NAME' in metadata:
            band_name = metadata[f'BAND_{i}_NAME']
        else:
            band_name = f"band_{i}"
        
        # Read band data
        band_data = band.ReadAsArray().astype(np.float32)
        
        # Convert from percentage (0-100) to proportion (0-1)
        band_data = band_data / 100.0
        
        # Replace no data values with NaN
        nodata = band.GetNoDataValue()
        if nodata is not None:
            band_data[band_data == nodata/100.0] = np.nan
        
        band_names.append(band_name)
        bands_data[band_name] = band_data
    
    return bands_data, band_names

def compare_bands(truth_bands, pred_bands, selected_bands=None, band_mapping=None):
    """
    Compare corresponding bands from ground truth and predictions
    
    Args:
        truth_bands: Dictionary of ground truth bands
        pred_bands: Dictionary of prediction bands
        selected_bands: List of band names to compare (default: compare all common bands)
        band_mapping: Dictionary mapping prediction band names to truth band names
        
    Returns:
        data: Dictionary of dictionaries with metrics and arrays for each band
    """
    # If no specific bands are selected, compare all common bands
    if selected_bands is None:
        selected_bands = list(set(truth_bands.keys()) & set(pred_bands.keys()))
    
    print(f"Comparing the following bands: {selected_bands}")
    
    # Initialize results dictionary
    results = {}
    
    for band_name in selected_bands:
        # Get the corresponding band names in prediction and truth
        if band_mapping and band_name in band_mapping:
            pred_band_name = band_mapping[band_name]
            truth_band_name = band_name
            print(f"Mapping prediction band '{pred_band_name}' to truth band '{truth_band_name}'")
        else:
            pred_band_name = truth_band_name = band_name
        
        # Check if we need to handle sqrt transformed bands
        if truth_band_name not in truth_bands or pred_band_name not in pred_bands:
            # Try with sqrt prefix for both
            sqrt_truth_band_name = f"sqrt_{truth_band_name}" if not truth_band_name.startswith("sqrt_") else truth_band_name
            sqrt_pred_band_name = f"sqrt_{pred_band_name}" if not pred_band_name.startswith("sqrt_") else pred_band_name
            
            # Check if we have sqrt version in both truth and pred
            if sqrt_truth_band_name in truth_bands and sqrt_pred_band_name in pred_bands:
                print(f"Using sqrt-transformed bands '{sqrt_truth_band_name}' and '{sqrt_pred_band_name}'")
                
                # Get band data and square it to get back original scale
                truth_data = np.square(truth_bands[sqrt_truth_band_name])
                pred_data = np.square(pred_bands[sqrt_pred_band_name])
            else:
                print(f"Warning: Couldn't find matching bands for '{band_name}'")
                print(f"Available truth bands: {list(truth_bands.keys())}")
                print(f"Available pred bands: {list(pred_bands.keys())}")
                continue
        else:
            # Get band data directly
            truth_data = truth_bands[truth_band_name]
            pred_data = pred_bands[pred_band_name]
        
        # Make sure arrays have the same shape
        if truth_data.shape != pred_data.shape:
            print(f"Warning: Shape mismatch for band '{band_name}': "
                  f"truth shape {truth_data.shape}, pred shape {pred_data.shape}")
            print(f"Warning: Shape mismatch for band {band_name}")
            continue
        
        # Flatten arrays and filter out NaN values
        valid_mask = ~np.isnan(truth_data) & ~np.isnan(pred_data)
        truth_flat = truth_data[valid_mask].flatten()
        pred_flat = pred_data[valid_mask].flatten()
        
        # Skip if no valid pixels
        if len(truth_flat) == 0:
            print(f"Warning: No valid pixels for band {band_name}")
            continue
        
        # Calculate metrics
        r2 = r2_score(truth_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(truth_flat, pred_flat))
        pearson_coef, _ = pearsonr(truth_flat, pred_flat)
        
        results[band_name] = {
            'truth': truth_flat,
            'pred': pred_flat,
            'metrics': {
                'r2': r2,
                'rmse': rmse,
                'pearson': pearson_coef
            }
        }
    
    return results

def plot_results(results, output_path):
    """
    Plot scatter plots of truth vs prediction for each band
    
    Args:
        results: Dictionary with comparison results
        output_path: Path to save the output plot
    """
    # Number of bands to plot
    n_bands = len(results)
    
    if n_bands == 0:
        print("No valid bands to plot")
        return
    
    # Create a figure with subplots (1 row, n_bands columns)
    fig, axes = plt.subplots(1, n_bands, figsize=(n_bands * 5, 5))
    
    # If there's only one band, axes won't be an array
    if n_bands == 1:
        axes = [axes]
    
    # Plot each band
    for i, (band_name, band_data) in enumerate(results.items()):
        ax = axes[i]
        truth = band_data['truth']
        pred = band_data['pred']
        metrics = band_data['metrics']
        
        # Create scatter plot
        ax.scatter(truth, pred, alpha=0.1, s=1)
        
        # Get max value for plot limits
        max_val = max(np.max(truth), np.max(pred))
        
        # Plot identity line (y=x)
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Set labels and title
        ax.set_xlabel("Ground Truth Proportion")
        ax.set_ylabel("Predicted Proportion")
        ax.set_title(f"{band_name}\nR² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nr = {metrics['pearson']:.3f}")
        
        # Set axis limits
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        
        # Add grid
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results plot saved to {output_path}")
    
    # Create a version with hexbin for better visualization of dense data
    fig, axes = plt.subplots(1, n_bands, figsize=(n_bands * 5, 5))
    
    # If there's only one band, axes won't be an array
    if n_bands == 1:
        axes = [axes]
    
    # Plot each band using hexbin
    for i, (band_name, band_data) in enumerate(results.items()):
        ax = axes[i]
        truth = band_data['truth']
        pred = band_data['pred']
        metrics = band_data['metrics']
        
        # Create hexbin plot
        hb = ax.hexbin(truth, pred, gridsize=50, cmap='viridis', mincnt=1)
        
        # Get max value for plot limits
        max_val = max(np.max(truth), np.max(pred))
        
        # Plot identity line (y=x)
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Set labels and title
        ax.set_xlabel("Ground Truth Proportion")
        ax.set_ylabel("Predicted Proportion")
        ax.set_title(f"{band_name}\nR² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nr = {metrics['pearson']:.3f}")
        
        # Set axis limits
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Add colorbar
        fig.colorbar(hb, ax=ax, label='Count')
    
    plt.tight_layout()
    hexbin_path = output_path.replace('.png', '_hexbin.png')
    plt.savefig(hexbin_path)
    print(f"Hexbin plot saved to {hexbin_path}")
    
    # Display metrics in a more readable format
    print("\nPerformance metrics summary:")
    print("=" * 60)
    print(f"{'Band':<25} {'R²':>10} {'RMSE':>10} {'Pearson r':>10}")
    print("-" * 60)
    for band_name, band_data in results.items():
        metrics = band_data['metrics']
        print(f"{band_name:<25} {metrics['r2']:>10.4f} {metrics['rmse']:>10.4f} {metrics['pearson']:>10.4f}")
    print("=" * 60)
    
    # Save metrics to CSV
    metrics_path = output_path.replace('.png', '_metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write("band,r2,rmse,pearson\n")
        for band_name, band_data in results.items():
            metrics = band_data['metrics']
            f.write(f"{band_name},{metrics['r2']:.6f},{metrics['rmse']:.6f},{metrics['pearson']:.6f}\n")
    print(f"Metrics saved to {metrics_path}")

def main(band_mapping, truth_path=None, pred_path=None, output_path=None, ):
    """
    Main function to compare ground truth and predicted proportion TIFs
    
    Args:
        truth_path: Path to ground truth TIF
        pred_path: Path to prediction TIF
        output_path: Path to save output plot
        bands: List of band names to compare
        band_mapping: Dictionary mapping prediction band names to truth band names
    """
    bands = band_mapping.keys()
        
    # Load TIFs
    print(f"Loading ground truth TIF: {truth_path}")
    truth_bands, truth_band_names = load_tif_bands(truth_path)
    print(f"Found bands: {truth_band_names}")
    
    print(f"\nLoading prediction TIF: {pred_path}")
    pred_bands, pred_band_names = load_tif_bands(pred_path)
    print(f"Found bands: {pred_band_names}")
    
    # Print information about sqrt-transformed bands, if any
    sqrt_bands = [name for name in truth_band_names if name.startswith('sqrt_')]
    if sqrt_bands:
        print(f"\nFound sqrt-transformed bands in ground truth: {sqrt_bands}")
    
    sqrt_bands = [name for name in pred_band_names if name.startswith('sqrt_')]
    if sqrt_bands:
        print(f"Found sqrt-transformed bands in predictions: {sqrt_bands}")
    
    # Compare bands
    results = compare_bands(truth_bands, pred_bands, selected_bands=bands, band_mapping=band_mapping)
    
    # Plot results
    plot_results(results, output_path)

if __name__ == "__main__":
    
    # Default paths
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""
    mediane_dir = "mediane" if not moy5m else "mediane_10m"
    file_prefix = "" if not moy5m else "10m_"
    model_type = "individual" 

    
    # Truth TIF (from create_proportion_tif.py)
    truth_tif = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/proportions_WAP{wap}.tif"

    # Prediction TIF (from plot_regression_tif.py)
    pred_tif = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_predictions_WAP{wap}{peat_suffix}{resolution_suffix}.tif"

    # Output path
    output_path = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_evaluation.png"

    # Default bands to compare


    # Define a mapping between truth band names and prediction band names
    # This is needed if the band names in the prediction tiff don't match the truth tiff
    band_mapping = {
        "Pure_Lichen": "band_1",
        "Degraded_Lichen": "band_2",
        "Green": "band_3",
        "all_lichen": "band_4",
        "through_proportion": "band_5"
    }

    # Call main function with parameters directly
    main(band_mapping, truth_tif, pred_tif, output_path)

