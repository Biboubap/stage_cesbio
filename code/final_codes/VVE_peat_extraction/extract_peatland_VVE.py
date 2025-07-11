#!/usr/bin/env python3
"""
Peatland Extraction Tool for Sentinel-2 Imagery

This script creates a binary mask of potential peatland areas by applying
spectral thresholds to Sentinel-2 bands. It processes very large rasters
(~9GB per band) efficiently by using GDAL block-based reading/writing.

Usage:
  python extract_peatland_VVE.py --ir path/to/infrared.tif --red path/to/red.tif 
                                --green path/to/green.tif 
                                --output path/to/output_mask.tif
"""

import os
import argparse
import numpy as np
from osgeo import gdal
import sys
from tqdm import tqdm
from scipy import ndimage

# Define spectral thresholds for peatland identification
IR_MIN = 2000    # Infrared band min threshold
IR_MAX = 3000    # Infrared band max threshold

RED_MIN = 750  # Red band min threshold
RED_MAX = 1500

GREEN_MIN = 600  # Green band min threshold
GREEN_MAX = 1500

# Post-processing parameters
DISTANCE_INCLUSION = 0  # Pixels to expand the mask boundaries
GROUPE_INCLUSION = 10    # Maximum size of holes to fill

# Block size for processing (adjust based on available memory)
BLOCK_SIZE = 2048
# Overlap between blocks for correct boundary processing
BLOCK_OVERLAP = max(DISTANCE_INCLUSION * 2, 3)  # Ensure enough overlap for operations

def create_parser():
    """Create command line parser"""
    parser = argparse.ArgumentParser(description='Extract peatland areas from Sentinel-2 bands')
    parser.add_argument('--ir', required=True, help='Path to infrared band (B8)')
    parser.add_argument('--red', required=True, help='Path to red band (B4)')
    parser.add_argument('--green', required=True, help='Path to green band (B3)')
    parser.add_argument('--output', required=True, help='Path to save output mask')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, 
                        help=f'Block size for processing (default: {BLOCK_SIZE})')
    parser.add_argument('--distance-inclusion', type=int, default=DISTANCE_INCLUSION,
                        help=f'Distance to expand mask boundaries (default: {DISTANCE_INCLUSION})')
    parser.add_argument('--groupe-inclusion', type=int, default=GROUPE_INCLUSION,
                        help=f'Maximum size of holes to fill (default: {GROUPE_INCLUSION})')
    return parser

def expand_mask_boundaries(mask):
    """
    Expand the mask boundaries by converting 0's adjacent to 1's to 1's.
    
    Args:
        mask: Binary mask (0's and 1's)
        
    Returns:
        Expanded mask
    """
    # Create a structuring element for dilation (8-connectivity)
    struct = np.ones((3, 3), dtype=bool)
    # Dilate the mask
    dilated = ndimage.binary_dilation(mask.astype(bool), structure=struct)
    print(f"Mask expanded from {np.sum(mask)} to {np.sum(dilated)} pixels")
    return dilated.astype(np.uint8)

def fill_small_holes(mask, max_size=8):
    """
    Fill small holes (groups of 0's surrounded by 1's) in the mask.
    
    Args:
        mask: Binary mask (0's and 1's)
        max_size: Maximum size of holes to fill
        
    Returns:
        Mask with small holes filled
    """
    # Make sure mask is boolean for proper hole detection
    mask_bool = mask.astype(bool)
    
    # Find all connected components of background (0's)
    # Structure is for 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    labeled_holes, num_holes = ndimage.label(~mask_bool, structure=structure)
    
    if num_holes == 0:
        return mask
    
    # Count pixels in each labeled component
    component_sizes = np.bincount(labeled_holes.flatten())[1:]  # Skip background label 0
    
    # Find hole labels to fill (those smaller than or equal to max_size)
    holes_to_fill = np.where(component_sizes <= max_size)[0] + 1  # +1 because labels start at 1
    
    # Count how many small holes were found
    small_holes_count = len(holes_to_fill)
    small_pixels_count = sum(component_sizes[i-1] for i in holes_to_fill)
    
    # Create output mask by filling small holes
    filled_mask = mask.copy()
    if small_holes_count > 0:
        # Create mask of all small holes
        holes_mask = np.isin(labeled_holes, holes_to_fill)
        # Fill these holes in the output mask
        filled_mask[holes_mask] = 1

    return filled_mask

def verify_rasters_consistency(raster_paths):
    """
    Verify that all input rasters have the same dimensions and projection.
    
    Args:
        raster_paths: List of paths to raster files
    
    Returns:
        Tuple of (width, height) if consistent, raises ValueError otherwise
    """
    print("Verifying raster consistency...")
    
    # Open first raster to get reference dimensions
    ds_ref = gdal.Open(raster_paths[0], gdal.GA_ReadOnly)
    if ds_ref is None:
        raise ValueError(f"Could not open reference raster: {raster_paths[0]}")
    
    width = ds_ref.RasterXSize
    height = ds_ref.RasterYSize
    projection = ds_ref.GetProjection()
    ds_ref = None  # Close dataset
    
    print(f"Reference dimensions: {width}x{height} pixels")
    
    # Check all other rasters
    for path in raster_paths[1:]:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError(f"Could not open raster: {path}")
        
        if ds.RasterXSize != width or ds.RasterYSize != height:
            raise ValueError(
                f"Inconsistent dimensions: {path} is {ds.RasterXSize}x{ds.RasterYSize}, "
                f"expected {width}x{height}"
            )
        
        if ds.GetProjection() != projection:
            print("Warning: Projection mismatch between rasters")
        
        ds = None  # Close dataset
    
    return width, height

def process_rasters(ir_path, red_path, green_path, output_path, block_size, 
                   distance_inclusion=DISTANCE_INCLUSION, groupe_inclusion=GROUPE_INCLUSION):
    """
    Process input rasters in blocks and create a binary mask based on spectral thresholds.
    Then apply post-processing: boundary expansion and small hole filling.
    
    Args:
        ir_path: Path to infrared band raster
        red_path: Path to red band raster
        green_path: Path to green band raster
        output_path: Path to save output mask
        block_size: Size of blocks for processing
        distance_inclusion: Distance to expand mask boundaries
        groupe_inclusion: Maximum size of holes to fill
    """
    # Verify raster consistency
    width, height = verify_rasters_consistency([ir_path, red_path, green_path])
    
    # Open input rasters
    ir_ds = gdal.Open(ir_path, gdal.GA_ReadOnly)
    red_ds = gdal.Open(red_path, gdal.GA_ReadOnly)
    green_ds = gdal.Open(green_path, gdal.GA_ReadOnly)
    
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES']
    output_ds = driver.Create(
        output_path, width, height, 1, gdal.GDT_Byte, options=options
    )
    
    # Copy georeferencing information from input
    output_ds.SetGeoTransform(ir_ds.GetGeoTransform())
    output_ds.SetProjection(ir_ds.GetProjection())
    
    # Get output band
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(0)
    output_band.Fill(0)  # Initialize with zeros
    
    # Calculate effective block size (accounting for overlap)
    effective_block_size = block_size - 2 * BLOCK_OVERLAP
    effective_block_size = max(effective_block_size, 128)  # Ensure minimum block size
    
    # Calculate number of blocks
    n_blocks_x = int(np.ceil(width / effective_block_size))
    n_blocks_y = int(np.ceil(height / effective_block_size))
    total_blocks = n_blocks_x * n_blocks_y
    
    print(f"Processing {total_blocks} blocks ({n_blocks_x} x {n_blocks_y})...")
    print(f"Using effective block size: {effective_block_size} with overlap: {BLOCK_OVERLAP}")
    print(f"Post-processing: Boundary expansion = {distance_inclusion}, Hole filling <= {groupe_inclusion} pixels")
    
    # Process blocks
    with tqdm(total=total_blocks, desc="Processing blocks") as pbar:
        for block_y in range(n_blocks_y):
            for block_x in range(n_blocks_x):
                # Calculate block coordinates with overlap
                x_offset = block_x * effective_block_size
                y_offset = block_y * effective_block_size
                
                # Add overlap, but don't go negative
                x_offset_with_overlap = max(0, x_offset - BLOCK_OVERLAP)
                y_offset_with_overlap = max(0, y_offset - BLOCK_OVERLAP)
                
                # Calculate block dimensions with overlap
                x_block_size = min(effective_block_size + 2 * BLOCK_OVERLAP, width - x_offset_with_overlap)
                y_block_size = min(effective_block_size + 2 * BLOCK_OVERLAP, height - y_offset_with_overlap)
                
                # Read data from input rasters
                ir_data = ir_ds.GetRasterBand(1).ReadAsArray(
                    x_offset_with_overlap, y_offset_with_overlap, x_block_size, y_block_size
                )
                red_data = red_ds.GetRasterBand(1).ReadAsArray(
                    x_offset_with_overlap, y_offset_with_overlap, x_block_size, y_block_size
                )
                green_data = green_ds.GetRasterBand(1).ReadAsArray(
                    x_offset_with_overlap, y_offset_with_overlap, x_block_size, y_block_size
                )
               
                # Apply spectral thresholds
                mask = np.logical_and.reduce([
                    ir_data > IR_MIN,
                    ir_data < IR_MAX,
                    red_data > RED_MIN,
                    red_data < RED_MAX,
                    green_data > GREEN_MIN,
                    green_data < GREEN_MAX,
                ]).astype(np.uint8)
                
                # Apply post-processing: expand boundaries
                for _ in range(distance_inclusion):
                    mask = expand_mask_boundaries(mask)
                
                # Apply post-processing: fill small holes
                mask = fill_small_holes(mask, max_size=groupe_inclusion)
                
                # Calculate coordinates of the block's interior (without overlap)
                interior_x_start = 0
                interior_y_start = 0
                interior_x_size = x_block_size
                interior_y_size = y_block_size
                
                # Adjust if this block has overlap
                if x_offset > 0:
                    interior_x_start = BLOCK_OVERLAP
                    interior_x_size -= BLOCK_OVERLAP
                if y_offset > 0:
                    interior_y_start = BLOCK_OVERLAP
                    interior_y_size -= BLOCK_OVERLAP
                    
                # Adjust for the right and bottom edges
                if x_offset + effective_block_size > width:
                    interior_x_size = width - x_offset_with_overlap - interior_x_start
                else:
                    interior_x_size = effective_block_size
                
                if y_offset + effective_block_size > height:
                    interior_y_size = height - y_offset_with_overlap - interior_y_start
                else:
                    interior_y_size = effective_block_size
                
                # Extract the interior of the processed block
                interior_mask = mask[
                    interior_y_start:interior_y_start + interior_y_size,
                    interior_x_start:interior_x_start + interior_x_size
                ]
                
                # Write to output
                output_band.WriteArray(interior_mask, x_offset, y_offset)
                
                # Update progress bar
                pbar.update(1)
    
    # Flush cache and calculate statistics
    output_band.FlushCache()
    output_band.ComputeStatistics(False)
    
    # Close datasets
    ir_ds = None
    red_ds = None
    green_ds = None
    output_ds = None
    
    print(f"Peatland mask saved to {output_path}")
    
    # Report mask statistics
    ds = gdal.Open(output_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    
    
    ds = None

def main():
    """Main entry point for the script."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Register GDAL exceptions
    gdal.UseExceptions()
    
    try:
        # Process rasters
        process_rasters(
            args.ir, args.red, args.green,
            args.output, args.block_size,
            args.distance_inclusion, args.groupe_inclusion
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


"""

python /home/lcousin/stage_cesbio/code/final_codes/regression/extract_peatland_VVE.py \
    --ir /media/lcousin/FASTBOYSLIM/Churchill/DataCubeS2/Im_15VVE_B8Amean.tif \
    --red /media/lcousin/FASTBOYSLIM/Churchill/DataCubeS2/Im_15VVE_B4mean.tif \
    --green /media/lcousin/FASTBOYSLIM/Churchill/DataCubeS2/Im_15VVE_B3mean.tif \
    --output /media/lcousin/FASTBOYSLIM/Loris/VVE_mask/peat_plateau_mask_VVE_group15.tif \
    --distance-inclusion 0 \
    --groupe-inclusion 15

"""