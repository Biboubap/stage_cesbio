"""
Compute Median

This script computes the temporal median for a multi-temporal Sentinel-2 raster stack.
It's used as a preprocessing step to create stable, cloud-free composite images from
multiple Sentinel-2 acquisitions.

The median operation helps eliminate outliers such as clouds, shadows, and other
temporary features that might be present in individual acquisitions.

Usage:
  python compute_median.py input_stack.tif output_median.tif
  
Arguments:
  input_stack.tif - Multi-band GeoTIFF where each band is a different time acquisition
  output_median.tif - Output file containing the median value across all time points
"""
import sys
import numpy as np
import rasterio

# Parse command line arguments
input_path = sys.argv[1]  # Path to multi-temporal stack
output_path = sys.argv[2]  # Path to save median composite

# Open the input raster
with rasterio.open(input_path) as src:
    # Read all bands as a 3D array (bands, rows, cols)
    data = src.read()
    
    # Compute the median value across the temporal dimension (axis 0)
    # This creates a 2D array (rows, cols) with the median value at each pixel
    median = np.median(data, axis=0).astype(src.dtypes[0])
    
    # Copy the metadata from the input file
    profile = src.profile
    # Update the count to 1 since we're only writing one band
    profile.update(count=1)
    
    # Write the median values to a new GeoTIFF file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(median, 1)  # Write as band 1