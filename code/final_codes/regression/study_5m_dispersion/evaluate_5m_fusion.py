#!/usr/bin/env python3
"""
Study 5m Dispersion and Fusion Quality

This script evaluates the quality of 5m predictions fused to 10m resolution
by comparing them with the ground truth proportions and with direct 10m predictions.

The script:
1. Loads ground truth proportion maps (from compute_proportion.py)
2. Loads both 10m predictions and 5m-fused predictions
3. Compares predictions against ground truth
4. Creates scatter plots and calculates metrics for each site
5. Analyzes whether 5m fusion provides better results than direct 10m prediction

Usage:
    python study_5m_dispersion.py --truth path/to/proportions.tif 
                                 --fusion path/to/5m_fusion.tif
                                 [--native path/to/10m_prediction.tif]
                                 --category {lichen|trough}
                                 --site SITE_NAME
                                 --output path/to/output_dir
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import argparse

def parse_arguments():
    """
    Parse command line arguments for the 5m fusion evaluation.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Evaluate 5m fusion predictions against ground truth and 10m predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--truth', required=True,
                        help='Path to ground truth proportion TIF (from compute_proportion)')
    parser.add_argument('--fusion', required=True,
                        help='Path to 5m fusion prediction TIF (resampled to 10m)')
    parser.add_argument('--category', required=True, choices=['lichen', 'trough', 'green'],
                        help='Proportion category to evaluate')
    parser.add_argument('--site', required=True,
                        help='Site name for labeling plots')
    parser.add_argument('--output', required=True,
                        help='Directory to save output plots and metrics')
    
    # Optional arguments
    parser.add_argument('--native', 
                        help='Path to native 10m prediction TIF for comparison (optional)')
    parser.add_argument('--no-hexbin', action='store_true',
                        help='Disable hexbin plot creation')
    
    return parser.parse_args()

def load_tif_band(tif_path, band_index=1):
    """
    Load a specific band from a TIF file as a NumPy array.
    
    Args:
        tif_path: Path to the TIF file
        band_index: Band index to load (default: 1)
        
    Returns:
        Tuple of (band_data, geo_transform, projection)
    """
    ds = gdal.Open(tif_path)
    if ds is None:
        raise ValueError(f"Could not open TIF file: {tif_path}")
    
    band = ds.GetRasterBand(band_index)
    band_data = band.ReadAsArray().astype(np.float32)
    
    # Convert from percentage (0-100) to proportion (0-1)
    band_data = band_data / 100.0
    
    # Replace no data values with NaN
    nodata = band.GetNoDataValue()
    if nodata is not None:
        band_data[band_data == nodata/100.0] = np.nan
    
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    return band_data, geo_transform, projection

def find_proportion_band(tif_path, category):
    """
    Find the band index that corresponds to the requested category in a proportion TIF.
    
    Args:
        tif_path: Path to the proportion TIF file
        category: Category to find ('lichen', 'trough', or 'green')
        
    Returns:
        Band index (1-based) for the requested category
    """
    ds = gdal.Open(tif_path)
    if ds is None:
        raise ValueError(f"Could not open TIF file: {tif_path}")
    
    # Known band mapping for proportion TIF files
    # Band 1: lichen proportion
    # Band 2: green proportion
    # Band 3: trough proportion
    if category.lower() == 'lichen':
        print(f"Using known band mapping: 'lichen' = band 1")
        return 1
    elif category.lower() == 'green':
        print(f"Using known band mapping: 'green' = band 2")
        return 2
    elif category.lower() == 'trough':
        print(f"Using known band mapping: 'trough' = band 3")
        return 3
    
    # Check if band descriptions are available (as fallback)
    band_count = ds.RasterCount
    metadata = ds.GetMetadata()
    
    # Try metadata
    for i in range(1, band_count + 1):
        band_key = f"BAND_{i}_NAME"
        if band_key in metadata:
            band_name = metadata[band_key].lower()
            if category.lower() in band_name:
                print(f"Found category '{category}' in band {i} with name '{metadata[band_key]}'")
                return i
    
    # Try band descriptions
    for i in range(1, band_count + 1):
        band = ds.GetRasterBand(i)
        band_desc = band.GetDescription().lower()
        if band_desc and category.lower() in band_desc:
            print(f"Found category '{category}' in band {i} with description '{band.GetDescription()}'")
            return i
    
    # If no match found, warn and use default mapping
    print(f"Warning: Could not identify band for category '{category}'. Using default mapping.")
    
    # Default fallback mapping
    if category.lower() == 'lichen':
        return 1
    elif category.lower() == 'green':
        return 2
    elif category.lower() == 'trough':
        return 3
    else:
        print(f"Warning: Unknown category '{category}'. Using band 1.")
        return 1

def evaluate_5m_fusion(truth_path, fusion_path, native_path=None, category='lichen', site_name='Unknown', output_dir='.', make_hexbin=True):
    """
    Evaluate 5m fusion predictions against ground truth and optionally against native 10m predictions.
    
    Args:
        truth_path: Path to ground truth proportion TIF
        fusion_path: Path to 5m fusion prediction TIF
        native_path: Path to native 10m prediction TIF (optional)
        category: Category to evaluate ('lichen', 'trough', or 'green')
        site_name: Site name for labeling
        output_dir: Directory to save output plots and metrics
        make_hexbin: Whether to create hexbin plots (default: True)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {category} proportion predictions for site: {site_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find appropriate bands in each TIF
    truth_band_idx = find_proportion_band(truth_path, category)
    
    # Load ground truth data
    print(f"Loading ground truth from {truth_path} (band {truth_band_idx})...")
    
    # Open the datasets to get geospatial information
    truth_ds = gdal.Open(truth_path)
    fusion_ds = gdal.Open(fusion_path)
    
    if truth_ds is None or fusion_ds is None:
        raise ValueError("Could not open one of the TIF files")
    
    truth_gt = truth_ds.GetGeoTransform()
    truth_proj = truth_ds.GetProjection()
    fusion_gt = fusion_ds.GetGeoTransform()
    fusion_proj = fusion_ds.GetProjection()
    
    # Read truth data
    truth_band = truth_ds.GetRasterBand(truth_band_idx)
    truth_data = truth_band.ReadAsArray().astype(np.float32) / 100.0  # Convert to proportion
    truth_nodata = truth_band.GetNoDataValue()
    if truth_nodata is not None:
        truth_nodata = truth_nodata / 100.0
    
    # Create mask of valid truth data
    truth_valid_mask = np.ones_like(truth_data, dtype=bool)
    if truth_nodata is not None:
        truth_valid_mask = (truth_data != truth_nodata) & ~np.isnan(truth_data) & (truth_data >= 0)
    
    print(f"Truth data shape: {truth_data.shape}, valid pixels: {np.sum(truth_valid_mask)}")
    
    # Create a memory raster of the truth valid mask
    driver = gdal.GetDriverByName('MEM')
    mask_ds = driver.Create('', truth_ds.RasterXSize, truth_ds.RasterYSize, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(truth_gt)
    mask_ds.SetProjection(truth_proj)
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.WriteArray(truth_valid_mask.astype(np.uint8))
    mask_band.SetNoDataValue(0)
    
    # Now warp the fusion raster to match the truth raster's extent and resolution
    print(f"Loading 5m fusion prediction from {fusion_path} and aligning with truth data...")
    
    # Use gdal.Warp to properly align the fusion data with the truth data
    aligned_fusion_ds = gdal.Warp('', fusion_path, 
                                format='MEM',
                                outputBounds=[truth_gt[0], truth_gt[3] + truth_gt[5] * truth_ds.RasterYSize,
                                             truth_gt[0] + truth_gt[1] * truth_ds.RasterXSize, truth_gt[3]],
                                width=truth_ds.RasterXSize,
                                height=truth_ds.RasterYSize,
                                resampleAlg=gdal.GRA_Bilinear)
    
    if aligned_fusion_ds is None:
        raise ValueError("Failed to align fusion data with truth data")
    
    # Read the aligned fusion data
    fusion_data = aligned_fusion_ds.GetRasterBand(1).ReadAsArray().astype(np.float32) / 100.0
    
    # Apply the truth mask to the fusion data
    fusion_valid = fusion_data * truth_valid_mask
    
    # Get only the valid pixels from both datasets
    valid_mask = truth_valid_mask & ~np.isnan(fusion_data) & (fusion_data >= 0)
    
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid overlapping pixels found for comparison after geospatial alignment")
    
    print(f"Found {np.sum(valid_mask)} valid overlapping pixels for comparison")
    
    # Extract valid pixels for comparison
    truth_valid = truth_data[valid_mask].flatten()
    fusion_valid = fusion_data[valid_mask].flatten()
    
    # Calculate metrics for 5m fusion vs ground truth
    r2_fusion = r2_score(truth_valid, fusion_valid)
    rmse_fusion = np.sqrt(mean_squared_error(truth_valid, fusion_valid))
    pearson_fusion, _ = pearsonr(truth_valid, fusion_valid)
    
    print("\n5m Fusion vs Ground Truth Metrics:")
    print(f"  R²: {r2_fusion:.4f}")
    print(f"  RMSE: {rmse_fusion:.4f}")
    print(f"  Pearson r: {pearson_fusion:.4f}")
    
    # Handle native 10m prediction if provided
    native_valid = None
    if native_path:
        print(f"Loading native 10m prediction from {native_path}...")
        
        # Align native 10m prediction with truth data using the same approach
        aligned_native_ds = gdal.Warp('', native_path, 
                                    format='MEM',
                                    outputBounds=[truth_gt[0], truth_gt[3] + truth_gt[5] * truth_ds.RasterYSize,
                                                 truth_gt[0] + truth_gt[1] * truth_ds.RasterXSize, truth_gt[3]],
                                    width=truth_ds.RasterXSize,
                                    height=truth_ds.RasterYSize,
                                    resampleAlg=gdal.GRA_Bilinear)
        
        if aligned_native_ds is None:
            print("Warning: Failed to align native 10m prediction with truth data")
        else:
            # Read the aligned native data
            native_data = aligned_native_ds.GetRasterBand(1).ReadAsArray().astype(np.float32) / 100.0
            
            # Apply the same mask to get valid pixels
            valid_mask_native = valid_mask & ~np.isnan(native_data) & (native_data >= 0)
            
            if np.sum(valid_mask_native) == 0:
                print("Warning: No valid overlapping pixels found for native 10m prediction")
            else:
                # Extract valid pixels from all three datasets
                truth_valid_all = truth_data[valid_mask_native].flatten()
                fusion_valid_all = fusion_data[valid_mask_native].flatten()
                native_valid = native_data[valid_mask_native].flatten()
                
                # Calculate metrics for native 10m vs ground truth
                r2_native = r2_score(truth_valid_all, native_valid)
                rmse_native = np.sqrt(mean_squared_error(truth_valid_all, native_valid))
                pearson_native, _ = pearsonr(truth_valid_all, native_valid)
                
                print("\nNative 10m vs Ground Truth Metrics:")
                print(f"  R²: {r2_native:.4f}")
                print(f"  RMSE: {rmse_native:.4f}")
                print(f"  Pearson r: {pearson_native:.4f}")
                
                # Recalculate fusion metrics for the common set of valid pixels
                r2_fusion_all = r2_score(truth_valid_all, fusion_valid_all)
                rmse_fusion_all = np.sqrt(mean_squared_error(truth_valid_all, fusion_valid_all))
                pearson_fusion_all, _ = pearsonr(truth_valid_all, fusion_valid_all)
                
                print("\n5m Fusion vs Ground Truth (common pixels with 10m):")
                print(f"  R²: {r2_fusion_all:.4f}")
                print(f"  RMSE: {rmse_fusion_all:.4f}")
                print(f"  Pearson r: {pearson_fusion_all:.4f}")
                
                # Calculate improvement percentage
                r2_improvement = (r2_fusion_all - r2_native) / abs(r2_native) * 100
                rmse_improvement = (rmse_native - rmse_fusion_all) / rmse_native * 100
                
                print("\nImprovement from 10m to 5m fusion:")
                print(f"  R² improvement: {r2_improvement:.2f}%")
                print(f"  RMSE improvement: {rmse_improvement:.2f}%")
                
                # Save comparison rasters for visual inspection
                comparison_dir = os.path.join(output_dir, "comparison_rasters")
                os.makedirs(comparison_dir, exist_ok=True)
                
                # Save masked versions of each raster to see the comparison area
                for data, name in [(truth_data * valid_mask_native, "truth_masked"),
                                  (fusion_data * valid_mask_native, "fusion_masked"),
                                  (native_data * valid_mask_native, "native_masked"),
                                  (valid_mask_native.astype(np.float32), "comparison_mask")]:
                    out_path = os.path.join(comparison_dir, f"{site_name}_{category}_{name}.tif")
                    out_ds = driver.Create(out_path, truth_ds.RasterXSize, truth_ds.RasterYSize, 1, gdal.GDT_Float32)
                    out_ds.SetGeoTransform(truth_gt)
                    out_ds.SetProjection(truth_proj)
                    out_band = out_ds.GetRasterBand(1)
                    out_band.WriteArray(data)
                    out_band.SetNoDataValue(0)
                    out_ds = None  # Close the dataset
                    print(f"Saved {out_path} for visual inspection")
    
    # Create scatter plot
    create_scatter_plot(
        truth_valid, fusion_valid, native_valid,
        output_dir, site_name, category,
        r2_fusion, rmse_fusion, pearson_fusion,
        make_hexbin
    )
    
    # Save metrics to CSV
    metrics_path = os.path.join(output_dir, f"{site_name}_{category}_metrics.csv")
    with open(metrics_path, 'w') as f:
        f.write("metric,fusion_value\n")
        f.write(f"r2,{r2_fusion:.6f}\n")
        f.write(f"rmse,{rmse_fusion:.6f}\n")
        f.write(f"pearson,{pearson_fusion:.6f}\n")
        f.write(f"valid_pixels,{len(truth_valid)}\n")
    
    print(f"\nResults saved to {output_dir}")
    
    # Close datasets
    truth_ds = None
    fusion_ds = None
    mask_ds = None
    aligned_fusion_ds = None
    if native_path:
        aligned_native_ds = None
    
    # Return metrics dictionary
    metrics = {
        'r2': r2_fusion,
        'rmse': rmse_fusion,
        'pearson': pearson_fusion,
        'n_pixels': len(truth_valid)
    }
    
    if native_valid is not None:
        metrics['native_r2'] = r2_native
        metrics['native_rmse'] = rmse_native
        metrics['native_pearson'] = pearson_native
        metrics['r2_improvement'] = r2_improvement
        metrics['rmse_improvement'] = rmse_improvement
    
    return metrics

def create_scatter_plot(truth, fusion, native=None, output_dir='.', site_name='Unknown', 
                       category='lichen', r2=None, rmse=None, pearson=None, make_hexbin=True):
    """
    Create scatter plots comparing predicted values against ground truth.
    
    Args:
        truth: Ground truth values
        fusion: 5m fusion prediction values
        native: Native 10m prediction values (optional)
        output_dir: Directory to save plots
        site_name: Site name for labeling
        category: Category name ('lichen', 'trough', or 'green')
        r2, rmse, pearson: Metrics to display on plot
        make_hexbin: Whether to create hexbin plots
    """
    # Regular scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot 5m fusion predictions
    plt.scatter(truth, fusion, alpha=0.3, s=5, label='5m Fusion', color='blue')
    
    # Add native 10m predictions if provided
    if native is not None:
        plt.scatter(truth, native, alpha=0.3, s=5, label='Native 10m', color='red')
    
    # Add identity line
    max_val = max(np.max(truth), np.max(fusion))
    if native is not None:
        max_val = max(max_val, np.max(native))
    plt.plot([0, max_val], [0, max_val], 'k--', label='1:1 Line')
    
    # Add labels and title
    plt.xlabel('Ground Truth Proportion')
    plt.ylabel('Predicted Proportion')
    
    title = f"{site_name}: {category.capitalize()} Proportion"
    if r2 is not None:
        title += f"\nR² = {r2:.3f}"
    if rmse is not None:
        title += f", RMSE = {rmse:.3f}"
    if pearson is not None:
        title += f", r = {pearson:.3f}"
    plt.title(title)
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    scatter_path = os.path.join(output_dir, f"{site_name}_{category}_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    
    # Create hexbin plot for better visualization of dense data
    if make_hexbin:
        plt.figure(figsize=(10, 8))
        
        # Plot 5m fusion predictions as hexbin
        hb = plt.hexbin(truth, fusion, gridsize=50, cmap='Blues', mincnt=1, alpha=0.8, label='5m Fusion')
        
        # Add identity line
        plt.plot([0, max_val], [0, max_val], 'k--', label='1:1 Line')
        
        # Add colorbar
        cb = plt.colorbar(hb, label='Count')
        
        # Add labels and title
        plt.xlabel('Ground Truth Proportion')
        plt.ylabel('Predicted Proportion')
        plt.title(title)
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        hexbin_path = os.path.join(output_dir, f"{site_name}_{category}_hexbin.png")
        plt.savefig(hexbin_path)
        plt.close()

def main():
    """
    Main function to evaluate 5m fusion predictions.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Evaluate 5m fusion predictions
    evaluate_5m_fusion(
        truth_path=args.truth,
        fusion_path=args.fusion,
        native_path=args.native,
        category=args.category,
        site_name=args.site,
        output_dir=args.output,
        make_hexbin=not args.no_hexbin
    )
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()

"""
python /home/lcousin/stage_cesbio/code/final_codes/regression/study_5m_dispersion/evaluate_5m_fusion.py\
    --truth /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/WAP23_5m/proportions_WAP23_5m.tif\
    --fusion /home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/regression_models_5m/lichen/WAP23_10m_lichen_proportion_prediction.tif\
    --category lichen \
    --site WAP23 \
    --output /home/lcousin/stage_cesbio/data/study_5m_dispersion/training_no_Lamprey_WAP23/WAP23_lichen_5m


    
"""