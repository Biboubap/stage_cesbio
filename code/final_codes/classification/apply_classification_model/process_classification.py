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
# Add parent directory to path to import utility modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.block_rasters_manager import BlockRastersManager
from utils.block_processor import process_block_with_overlap
from utils.block_sample import BlockSample

# Configure logging - reduce verbosity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbosity of block processor logs
logging.getLogger('utils.block_processor').setLevel(logging.WARNING)


def parse_arguments():
    """
    Parse command line arguments for the classification process.
    
    This function defines and processes all command line arguments that control
    how the classification is performed, including input/output paths, processing
    parameters, and resource allocation settings.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process classification on large raster images')
    parser.add_argument('--rgb', required=True, help='Path to RGB raster')
    parser.add_argument('--dsm', required=True, help='Path to Digital Surface Model (DSM) raster')
    parser.add_argument('--out', required=True, help='Output path for classification result')
    
    # Processing parameters
    parser.add_argument('--patch-size', type=int, default=None, 
                        help='Size of patches for classification (default: calculated from resolution)')
    parser.add_argument('--block-size', type=int, default=1024, 
                        help='Block size for processing in pixels (default: 1024)')
    parser.add_argument('--overlap', type=int, default=48, 
                        help='Overlap size between blocks in pixels (default: 48)')
    
    # Resource management
    parser.add_argument('--memory-limit', type=float, default=None, 
                        help='Memory limit per worker in GB (default: auto)')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--threads-per-worker', type=int, default=1, 
                        help='Threads per worker (default: 1)')
    
    return parser.parse_args()

def setup_dask_client(n_workers=None, threads_per_worker=1, memory_limit=None):
    """
    Set up a Dask client for parallel processing.
    
    This function configures a Dask LocalCluster and Client for distributed processing
    of the classification tasks. It automatically determines appropriate resource
    allocation if not specified by the user.
    
    Args:
        n_workers: Number of worker processes to use (default: auto-detect)
        threads_per_worker: Number of threads per worker (default: 1)
        memory_limit: Memory limit per worker in GB (default: auto-detect)
        
    Returns:
        Tuple of (client, cluster) for the Dask distributed system
    """
    # Auto-detect number of workers if not specified
    if n_workers is None:
        n_workers = max(1, psutil.cpu_count() - 1)  # Leave one CPU for system
    
    # Auto-determine memory limit if not specified
    if memory_limit is None:
        total_mem = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        memory_limit = f"{max(2, total_mem / n_workers * 0.7):.1f}GB"  # 70% of available memory per worker
    else:
        memory_limit = f"{memory_limit}GB"
    
    # Fix the daemon warning by configuring dask first
    import dask
    dask.config.set({'distributed.worker.daemon': False})
    
    logger.info(f"Setting up Dask with {n_workers} workers, "
                f"{threads_per_worker} threads per worker, "
                f"and {memory_limit} memory per worker")
    
    # Create the Dask cluster and client
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True  # Use separate processes for better isolation
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client, cluster

def create_output_raster(rgb_path, output_path, patch_size=1, dtype=gdal.GDT_Byte):
    """
    Create an output raster with dimensions adjusted by patch_size while preserving georeferencing.
    
    This function creates a new GeoTIFF at a lower resolution (based on patch_size) than the
    input RGB file, but with proper georeferencing so that each pixel in the output corresponds
    to a patch in the input.
    
    Args:
        rgb_path: Path to the input RGB raster
        output_path: Path where the output raster will be saved
        patch_size: Size of classification patches in pixels
        dtype: GDAL data type for the output raster (default: Byte for class labels)
        
    Returns:
        GDAL dataset object for the output raster
    """
    # Open the input raster to get its dimensions and geotransform
    src_ds = gdal.Open(rgb_path)
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    
    # Calculate dimensions of the output raster (at patch resolution)
    out_width = width // patch_size
    out_height = height // patch_size
    logger.info(f"Creating output raster with dimensions {out_width}x{out_height} (patch size: {patch_size})")
    
    # Get the original geotransform
    gt = src_ds.GetGeoTransform()
    
    # Create a new geotransform with adjusted pixel size
    # [0]: top-left x, [1]: pixel width, [2]: rotation, [3]: top-left y, [4]: rotation, [5]: pixel height
    new_gt = (
        gt[0],                  # Same top-left x coordinate
        gt[1] * patch_size,     # Pixel width multiplied by patch_size
        gt[2],                  # Same rotation (typically 0)
        gt[3],                  # Same top-left y coordinate
        gt[4],                  # Same rotation (typically 0)
        gt[5] * patch_size      # Pixel height multiplied by patch_size (note: typically negative)
    )
    
    # Create the output raster with compression and tiling options
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, out_width, out_height, 1, dtype,
                           options=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    
    # Set the adjusted geotransform and projection
    out_ds.SetGeoTransform(new_gt)
    out_ds.SetProjection(src_ds.GetProjection())
    
    # Initialize with nodata (0 = background/no classification)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)
    out_band.Fill(0)
    
    # Write changes to disk
    out_ds.FlushCache()
    return out_ds

def calculate_blocks(raster_width, raster_height, block_size, overlap, patch_size=1):
    """
    Calculate block coordinates with overlap for processing large rasters in chunks.
    
    Adjusted to account for patch_size, so that block coordinates align with patch boundaries.
    
    Args:
        raster_width: Width of the entire raster in pixels
        raster_height: Height of the entire raster in pixels
        block_size: Size of each processing block in pixels
        overlap: Size of the overlap between adjacent blocks in pixels
        patch_size: Size of classification patches in pixels
        
    Returns:
        List of dictionaries containing block coordinates and valid regions
    """
    blocks = []
    
    # Adjust block_size and overlap to be multiples of patch_size if needed
    block_size = (block_size // patch_size) * patch_size
    overlap = (overlap // patch_size) * patch_size
    
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
                'valid_x_end': valid_x_end - x_start,      # Relative to block
                'valid_y_end': valid_y_end - y_start,      # Relative to block
                'out_x_start': valid_x_start,              # Absolute coordinates for output
                'out_y_start': valid_y_start,              # Absolute coordinates for output
            })
    
    return blocks

def calculate_patch_size(rgb_path, target_cm=16.8):
    """
    Calculate the appropriate patch size based on the RGB raster resolution.
    
    This function determines the optimal patch size to use for classification based on
    the spatial resolution of the input raster. It aims to maintain a consistent
    ground sample area regardless of the input resolution.
    
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
    """
    Main function to process the classification on a large raster.
    
    This function orchestrates the entire classification process and now produces
    a classification map at patch resolution instead of the full RGB resolution.
    
    Args:
        args: Command line arguments containing all processing parameters
    """
    start_time = time.time()
    
    # Set up Dask client for parallel processing
    client, cluster = setup_dask_client(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit
    )
    
    try:
        # Determine appropriate patch size based on raster resolution
        patch_size = args.patch_size
        if patch_size is None:
            logger.info("Patch size not specified, calculating from raster resolution")
            patch_size = calculate_patch_size(args.rgb)
            logger.info(f"Using calculated patch size: {patch_size}")
        else:
            logger.info(f"Using user-specified patch size: {patch_size}")
        
        # Load classification models - these contain both the trained models and
        # the feature names required by each model
        logger.info("Loading classification models...")
        model1_data = joblib.load(MODEL1_PATH)
        model1 = model1_data["model"]
        feature_names_model1 = model1_data["feature_names"]
        
        model2_data = joblib.load(MODEL2_PATH)
        model2 = model2_data["model"]
        feature_names_model2 = model2_data["feature_names"]
        logger.info("Models loaded successfully.")
        
        # Distribute models to workers to avoid reloading for each block
        # This improves efficiency by making the models available to all workers
        logger.info("Distributing models to workers...")
        model1_future = client.scatter(model1)
        model2_future = client.scatter(model2)
        feature_names_model1_future = client.scatter(feature_names_model1)
        feature_names_model2_future = client.scatter(feature_names_model2)
        logger.info("Models distributed successfully.")
        
        # Open input raster to get dimensions
        rgb_ds = gdal.Open(args.rgb)
        raster_width = rgb_ds.RasterXSize
        raster_height = rgb_ds.RasterYSize
        rgb_ds = None  # Close the dataset to free resources
        
        # Create output raster with the adjusted patch-level resolution
        logger.info(f"Creating output raster at {args.out} with patch size {patch_size}")
        out_ds = create_output_raster(args.rgb, args.out, patch_size=patch_size)
        
        # Calculate processing blocks with overlap, aligned with patch boundaries
        blocks = calculate_blocks(
            raster_width, 
            raster_height,
            args.block_size,
            args.overlap,
            patch_size=patch_size
        )
        total_blocks = len(blocks)
        logger.info(f"Processing raster in {total_blocks} blocks with {args.overlap}px overlap")
        
        # Create delayed tasks for processing each block in parallel
        delayed_tasks = []
        for i, block in enumerate(blocks):
            task = dask.delayed(process_block_with_overlap)(
                rgb_path=args.rgb,
                dsm_path=args.dsm,
                model1=model1_future,
                model2=model2_future,
                feature_names_model1=feature_names_model1_future,
                feature_names_model2=feature_names_model2_future,
                block=block,
                patch_size=patch_size,
                block_index=i
            )
            delayed_tasks.append(task)
        
        # Execute all block processing tasks in parallel with progress tracking
        logger.info(f"Starting parallel processing of {total_blocks} blocks...")
        
        # Set up progress tracking
        processed_blocks = 0
        results = []
        
        # Group tasks for batch processing
        batch_size = min(20, total_blocks)  # Process 20 blocks at a time, or fewer if total_blocks < 20
        for i in range(0, total_blocks, batch_size):
            batch_end = min(i + batch_size, total_blocks)
            batch_tasks = delayed_tasks[i:batch_end]
            
            # Process batch
            batch_start_time = time.time()
            batch_results = dask.compute(*batch_tasks)
            batch_elapsed = time.time() - batch_start_time
            
            # Update progress
            results.extend(batch_results)
            processed_blocks += len(batch_results)
            percentage = (processed_blocks / total_blocks) * 100
            elapsed_total = time.time() - start_time
            
            # Print progress similar to train_regression_model.py
            logger.info(f"Classified {processed_blocks}/{total_blocks} blocks ({percentage:.1f}%) in {elapsed_total:.1f}s")
        
        # Write results to output raster
        logger.info("Writing classification results to output raster...")
        out_band = out_ds.GetRasterBand(1)
        blocks_written = 0
        
        for i, result in enumerate(results):
            if result is not None:
                # Unpack the result (now at patch resolution)
                block_data, block = result
                
                # Calculate valid region indices in patch units
                valid_x_start_patches = block['valid_x_start'] // patch_size
                valid_y_start_patches = block['valid_y_start'] // patch_size
                valid_x_size_patches = (block['valid_x_end'] - block['valid_x_start']) // patch_size
                valid_y_size_patches = (block['valid_y_end'] - block['valid_y_start']) // patch_size
                
                # Calculate output positions in patch units
                out_x_start_patches = block['out_x_start'] // patch_size
                out_y_start_patches = block['out_y_start'] // patch_size
                
                # Extract valid data from the block result (already at patch resolution)
                valid_data = block_data[
                    valid_y_start_patches:valid_y_start_patches + valid_y_size_patches,
                    valid_x_start_patches:valid_x_start_patches + valid_x_size_patches
                ]
                
                # Write to the output raster (already at patch resolution)
                out_band.WriteArray(valid_data, xoff=out_x_start_patches, yoff=out_y_start_patches)
                blocks_written += 1
        
        # Clean up and save the final output
        out_ds.FlushCache()
        out_ds = None
        
        elapsed_time = time.time() - start_time
        logger.info(f"Classification completed: {blocks_written}/{total_blocks} blocks processed in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to {args.out} at patch resolution ({patch_size}x{patch_size} pixels per patch)")
    
    finally:
        # Clean up Dask resources
        client.close()
        cluster.close()
        logger.info("Dask resources released")

# Hard-coded paths to the model files
MODEL1_PATH = "/home/lcousin/stage_cesbio/data/samples/selection14/model_wap32_no_chicoutai.joblib"
MODEL2_PATH = "/home/lcousin/stage_cesbio/data/samples/selection16/classifs/model_16_7/model_16_7.joblib"

if __name__ == "__main__":
    args = parse_arguments()
    process_classification(args)
"""

# Example using multi-line command (Linux):
python home/lcousin/stage_cesbio/code/final_codes/classification/apply_classification_model/process_classification.py\
 --rgb home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Chesnay/ChesnayAugust2023_ortho_export_MonJun16161612078476_32615.tif\
 --dsm home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Chesnay/Chesnay_DSM_Resampled.tif \
 --out media/lcousin/FASTBOYSLIM/Loris/KonstantinClassif/Chesnay_classif.tif

python home/lcousin/stage_cesbio/code/final_codes/classification/apply_classification_model/process_classification.py \
 --rgb /home/lcousin/stage_cesbio/drone_treated/Wap23_main_transparent_mosaic_group1.tif\
 --dsm /home/lcousin/stage_cesbio/drone_treated/Wap23_main_dsm.tif\
 --out /home/lcousin/stage_cesbio/drone_treated/WAP23_tiles/classif_well_16.tif



python home/lcousin/stage_cesbio/code/final_codes/classification/apply_classification_model/process_classification.py\
 --rgb home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Belcher/BelcherAugust2023_ortho_export_TueJun17212958144821_32615.tif\
 --dsm home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Belcher/Belcher_DSM_Resampled.tif \
 --out media/lcousin/FASTBOYSLIM/Loris/KonstantinClassif/Belcher_classif_16px.tif \
; python home/lcousin/stage_cesbio/code/final_codes/classification/apply_classification_model/process_classification.py\
 --rgb home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Chesnay/ChesnayAugust2023_ortho_export_MonJun16161612078476_32615.tif\
 --dsm home/lcousin/stage_cesbio/Konstantin/UAV_Konstantin_Tabatha/Chesnay/Chesnay_DSM_Resampled.tif \
 --out media/lcousin/FASTBOYSLIM/Loris/KonstantinClassif/Chesnay_classif_16px.tif

 python home/lcousin/stage_cesbio/code/final_codes/classification/apply_classification_model/process_classification.py \
 --rgb /home/lcousin/stage_cesbio/drone_treated/Wap12_Main_transparent_mosaic_group1.tif\
 --dsm /home/lcousin/stage_cesbio/drone_treated/Wap12_Main_dsm.tif\
 --out /home/lcousin/stage_cesbio/drone_treated/WAP12_tiles/wap12_classif_16px.tif


"""