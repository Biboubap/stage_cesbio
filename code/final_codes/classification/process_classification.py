#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import joblib
from osgeo import gdal
import dask
from dask.distributed import Client, LocalCluster
import psutil
import logging
import time
from pathlib import Path

# Use relative imports for modules in the same directory tree
from utils.block_rasters_manager import BlockRastersManager
from utils.block_processor import process_block_with_overlap
from utils.block_sample import BlockSample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hard-coded paths to the model files
MODEL1_PATH = "/home/lcousin/stage_cesbio/data/samples/selection14/model_wap32_no_chicoutai.joblib"
MODEL2_PATH = "/home/lcousin/stage_cesbio/data/samples/selection16/classifs/model_16_7/model_16_7.joblib"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process classification on large raster images')
    parser.add_argument('--rgb', required=True, help='Path to RGB raster')
    parser.add_argument('--dsm', required=True, help='Path to DSM raster')
    parser.add_argument('--out', required=True, help='Output path for classification result')
    
    # Processing parameters
    parser.add_argument('--patch-size', type=int, default=None, 
                        help='Size of patches for classification (default: calculated from resolution)')
    parser.add_argument('--block-size', type=int, default=1024, 
                        help='Block size for processing (default: 1024)')
    parser.add_argument('--overlap', type=int, default=48, 
                        help='Overlap size between blocks (default: 48)')
    
    # Resource management
    parser.add_argument('--memory-limit', type=float, default=None, 
                        help='Memory limit per worker in GB (default: auto)')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of workers (default: auto)')
    parser.add_argument('--threads-per-worker', type=int, default=1, 
                        help='Threads per worker (default: 1)')
    
    return parser.parse_args()

def setup_dask_client(n_workers=None, threads_per_worker=1, memory_limit=None):
    """Set up a Dask client for parallel processing."""
    if n_workers is None:
        n_workers = max(1, psutil.cpu_count() - 1)  # Leave one CPU for system
    
    if memory_limit is None:
        total_mem = psutil.virtual_memory().total / (1024**3)  # GB
        memory_limit = f"{max(2, total_mem / n_workers * 0.7):.1f}GB"
    else:
        memory_limit = f"{memory_limit}GB"
    
    # Fix the daemon warning by configuring dask first
    import dask
    dask.config.set({'distributed.worker.daemon': False})
    
    logger.info(f"Setting up Dask with {n_workers} workers, "
                f"{threads_per_worker} threads per worker, "
                f"and {memory_limit} memory per worker")
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client, cluster

def create_output_raster(rgb_path, output_path, dtype=gdal.GDT_Byte):
    """Create an output raster with the same dimensions and georeferencing as the input."""
    src_ds = gdal.Open(rgb_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, width, height, 1, dtype,
                           options=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    
    # Copy georeferencing information
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    
    # Initialize with nodata
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)
    out_band.Fill(0)
    
    out_ds.FlushCache()
    return out_ds

def calculate_blocks(raster_width, raster_height, block_size, overlap):
    """Calculate block coordinates with overlap."""
    blocks = []
    
    # Calculate effective block sizes (block_size - overlap on each side)
    effective_block = block_size - 2 * overlap
    
    # Calculate number of blocks needed
    num_blocks_x = (raster_width + effective_block - 1) // effective_block
    num_blocks_y = (raster_height + effective_block - 1) // effective_block
    
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # Calculate block coordinates with overlap
            x_start = max(0, x * effective_block - overlap)
            y_start = max(0, y * effective_block - overlap)
            
            # Adjust for edges of the raster
            x_end = min(raster_width, (x + 1) * effective_block + overlap)
            y_end = min(raster_height, (y + 1) * effective_block + overlap)
            
            # Calculate valid data region (without overlap)
            valid_x_start = max(x * effective_block, overlap)
            valid_y_start = max(y * effective_block, overlap)
            valid_x_end = min((x + 1) * effective_block, raster_width)
            valid_y_end = min((y + 1) * effective_block, raster_height)
            
            # Handle the first and last blocks correctly
            if x == 0:
                valid_x_start = 0
            if y == 0:
                valid_y_start = 0
                
            blocks.append({
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'valid_x_start': valid_x_start - x_start,  # Relative to block
                'valid_y_start': valid_y_start - y_start,  # Relative to block
                'valid_x_end': valid_x_end - x_start,     # Relative to block
                'valid_x_end': valid_x_end - x_start,     # Relative to block
                'valid_y_end': valid_y_end - y_start,     # Relative to block
                'out_x_start': valid_x_start,             # Absolute coordinates for output
                'out_y_start': valid_y_start,             # Absolute coordinates for output
            })
    
    return blocks

def calculate_patch_size(rgb_path, target_cm=16.8):
    """
    Calculate the appropriate patch size based on the RGB raster resolution.
    
    Args:
        rgb_path: Path to RGB raster
        target_cm: Target ground resolution in centimeters for each patch
        
    Returns:
        Calculated patch size (in pixels)
    """
    try:
        # Open the raster to get its resolution
        ds = gdal.Open(rgb_path)
        if ds is None:
            logger.error(f"Failed to open raster {rgb_path}")
            return 16  # Default to 16 if we can't determine resolution
        
        # Get the geotransform which contains the pixel size/resolution
        gt = ds.GetGeoTransform()
        
        # Resolution in map units (usually meters)
        res_x = abs(gt[1])
        res_y = abs(gt[5])
        
        # Use the average resolution
        resolution_m = (res_x + res_y) / 2
        
        # Convert resolution from meters to centimeters
        resolution_cm = resolution_m * 100
        
        # Calculate how many pixels needed to cover target_cm
        patch_size = max(1, round(target_cm / resolution_cm))
        
        logger.info(f"Raster resolution: {resolution_cm:.2f} cm/pixel")
        logger.info(f"Calculated patch size: {patch_size} pixels to achieve ~{target_cm} cm ground coverage")
        
        # Clean up
        ds = None
        
        return patch_size
    except Exception as e:
        logger.error(f"Error calculating patch size: {e}")
        return 16  # Default value

def process_classification(args):
    """Main function to process the classification on a large raster."""
    start_time = time.time()
    
    # Set up Dask client first (moved up in the function)
    client, cluster = setup_dask_client(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit
    )
    
    try:
        # Determine appropriate patch size
        patch_size = args.patch_size
        if patch_size is None:
            logger.info("Patch size not specified, calculating from raster resolution")
            patch_size = calculate_patch_size(args.rgb)
            logger.info(f"Using calculated patch size: {patch_size}")
        else:
            logger.info(f"Using user-specified patch size: {patch_size}")
        
        # Load models
        logger.info("Loading classification models...")
        model1_data = joblib.load(MODEL1_PATH)
        model1 = model1_data["model"]
        feature_names_model1 = model1_data["feature_names"]
        
        model2_data = joblib.load(MODEL2_PATH)
        model2 = model2_data["model"]
        feature_names_model2 = model2_data["feature_names"]
        logger.info("Models loaded successfully.")
        
        # Scatter models to workers
        logger.info("Distributing models to workers...")
        model1_future = client.scatter(model1)
        model2_future = client.scatter(model2)
        feature_names_model1_future = client.scatter(feature_names_model1)
        feature_names_model2_future = client.scatter(feature_names_model2)
        logger.info("Models distributed successfully.")
        
        # Open input rasters to get dimensions
        rgb_ds = gdal.Open(args.rgb)
        raster_width = rgb_ds.RasterXSize
        raster_height = rgb_ds.RasterYSize
        rgb_ds = None  # Close the dataset
        
        # Create output raster
        logger.info(f"Creating output raster at {args.out}")
        out_ds = create_output_raster(args.rgb, args.out)
        
        # Calculate blocks
        blocks = calculate_blocks(
            raster_width, 
            raster_height,
            args.block_size,
            args.overlap
        )
        logger.info(f"Processing raster in {len(blocks)} blocks with {args.overlap}px overlap")
        
        # Create delayed tasks for each block
        delayed_tasks = []
        for i, block in enumerate(blocks):
            task = dask.delayed(process_block_with_overlap)(
                rgb_path=args.rgb,
                dsm_path=args.dsm,
                model1=model1_future,                   # Now using future
                model2=model2_future,                   # Now using future
                feature_names_model1=feature_names_model1_future,  # Now using future
                feature_names_model2=feature_names_model2_future,  # Now using future
                block=block,
                patch_size=patch_size,
                block_index=i
            )
            delayed_tasks.append(task)
        
        # Execute tasks in parallel
        logger.info(f"Starting parallel processing of {len(delayed_tasks)} blocks...")
        results = dask.compute(*delayed_tasks)
        
        # Write results to output raster
        out_band = out_ds.GetRasterBand(1)
        for i, result in enumerate(results):
            if result is not None:
                # Unpack the result
                block_data, block = result
                
                # Write only the valid (non-overlapping) part to the output
                # Ensure correct coordinate ordering for GDAL
                x_offset = block['out_x_start']
                y_offset = block['out_y_start']
                x_size = block['valid_x_end'] - block['valid_x_start']
                y_size = block['valid_y_end'] - block['valid_y_start']
                
                # Extract valid data from the block result
                valid_data = block_data[
                    block['valid_y_start']:block['valid_y_end'],
                    block['valid_x_start']:block['valid_x_end']
                ]
                
                # Write to the output raster - ensure correct coordinate ordering for WriteArray
                out_band.WriteArray(valid_data, xoff=x_offset, yoff=y_offset)
        
        
        # Clean up
        out_ds.FlushCache()
        out_ds = None
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to {args.out}")
    
    finally:
        # Clean up Dask resources
        client.close()
        cluster.close()
        logger.info("Dask resources released")

if __name__ == "__main__":
    args = parse_arguments()
    process_classification(args)


# python code/final_codes/classification/process_classification.py --rgb drone_treated/WAP32_tiles/rgb/WAP32_full_transparent_mosaic_group1_08_05.tif --dsm drone_treated/WAP32_tiles/dsm/WAP32_full_dsm_08_05.tif --out drone_treated/WAP32_tiles/classif_08_05_test.tif
# python home/lcousin/stage_cesbio/code/final_codes/classification/process_classification.py --rgb home/lcousin/stage_cesbio/drone_treated/WAP32_tiles/rgb/WAP32_full_transparent_mosaic_group1_05_12.tif --dsm home/lcousin/stage_cesbio/drone_treated/WAP32_tiles/dsm/WAP32_full_dsm_05_12.tif --out media/lcousin/FASTBOYSLIM/Loris/test.tif



# python home/lcousin/stage_cesbio/code/final_codes/classification/process_classification.py --rgb home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Lamprey/LampreyAugust2023_ortho_export_MonJun16161722231719_32615.tif --dsm home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Lamprey/Lamprey_DSM_Resampled.tif --out media/lcousin/FASTBOYSLIM/Loris/KonstantinClassif/Lamprey_classif.tif