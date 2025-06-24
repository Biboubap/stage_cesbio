"""
Create a multi-band TIFF file containing the proportions of each class on Sentinel-2 samples.
Each band represents a different class proportion, scaled from 0-100 (percentage).
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import argparse
from tqdm import tqdm

def load_mask(mask_path, ref_transform, ref_projection, ref_width, ref_height):
    """
    Load a mask file and reproject it to match the reference dataset if needed
    
    Args:
        mask_path: Path to the mask file (should be binary, 1 for valid pixels)
        ref_transform: GeoTransform of the reference dataset
        ref_projection: Projection of the reference dataset
        ref_width: Width of the reference dataset
        ref_height: Height of the reference dataset
        
    Returns:
        mask_array: Binary mask array (1 for valid pixels, 0 for masked pixels)
    """
    print(f"Loading mask from {mask_path}...")
    
    # Open the mask file
    mask_ds = gdal.Open(mask_path)
    if mask_ds is None:
        raise ValueError(f"Could not open mask file: {mask_path}")
    
    # Check if mask needs reprojection
    mask_transform = mask_ds.GetGeoTransform()
    mask_projection = mask_ds.GetProjection()
    mask_width = mask_ds.RasterXSize
    mask_height = mask_ds.RasterYSize
    
    # If mask dimensions and projection match reference, read directly
    if (mask_width == ref_width and mask_height == ref_height and 
        mask_transform == ref_transform and mask_projection == ref_projection):
        mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
    else:
        # Reproject mask to match reference dataset
        print("Reprojecting mask to match reference dataset...")
        mem_driver = gdal.GetDriverByName('MEM')
        mask_reprojected = mem_driver.Create('', ref_width, ref_height, 1, gdal.GDT_Byte)
        mask_reprojected.SetGeoTransform(ref_transform)
        mask_reprojected.SetProjection(ref_projection)
        
        # Reproject
        gdal.ReprojectImage(mask_ds, mask_reprojected, mask_projection, ref_projection, 
                           gdal.GRA_NearestNeighbour)
        
        # Read reprojected mask
        mask_array = mask_reprojected.GetRasterBand(1).ReadAsArray()
    
    # Ensure mask is binary (0 or 1)
    mask_array = (mask_array > 0).astype(np.uint8)
    
    print(f"Mask loaded. Valid pixels: {np.sum(mask_array)}/{mask_array.size} ({np.sum(mask_array)/mask_array.size*100:.2f}%)")
    
    return mask_array

def create_proportion_tiff(input_csv, sentinel_path, output_path, mask_path=None):
    """
    Create a multi-band TIFF from class proportion data from a CSV file.
    
    Args:
        input_csv: Path to CSV containing class proportions (from sentinel_proportion.py)
        sentinel_path: Path to a Sentinel-2 raster for georeference
        output_path: Path to save the output multi-band TIFF
        mask_path: Path to a mask file (optional). Only pixels where mask is 1 will be processed
    """
    print(f"Creating proportion TIFF from {input_csv}")
    
    # Load class proportions data
    df = pd.read_csv(input_csv)
    
    # Define classes and get max row/col coordinates
    classes = [
        "Pure_Lichen",
        "Degraded_Lichen",
        "Green",
        "Sphagnum",
        "Depression",
        "Water",
        "all_lichen",
        "through_proportion",
    ]
    
    all_classes = classes
    
    # Get dimensions of Sentinel raster
    ds_sentinel = gdal.Open(sentinel_path)
    if ds_sentinel is None:
        raise ValueError(f"Could not open Sentinel raster: {sentinel_path}")
        
    sentinel_width = ds_sentinel.RasterXSize
    sentinel_height = ds_sentinel.RasterYSize
    geo_transform = ds_sentinel.GetGeoTransform()
    projection = ds_sentinel.GetProjection()

    # Read NoData mask from Sentinel raster (assume first band)
    sentinel_band = ds_sentinel.GetRasterBand(1)
    sentinel_nodata = sentinel_band.GetNoDataValue()
    sentinel_data = sentinel_band.ReadAsArray()
    if sentinel_nodata is not None:
        sentinel_mask = (sentinel_data == sentinel_nodata)
    else:
        sentinel_mask = np.zeros((sentinel_height, sentinel_width), dtype=bool)
    
    # Load mask if provided
    mask = None
    if mask_path:
        mask = load_mask(mask_path, geo_transform, projection, sentinel_width, sentinel_height)

    # Create output raster (multi-band)
    driver = gdal.GetDriverByName('GTiff')
    num_bands = len(all_classes)
    out_ds = driver.Create(output_path, sentinel_width, sentinel_height, num_bands, gdal.GDT_Int16, 
                          options=['COMPRESS=DEFLATE', 'TILED=YES'])
    
    if out_ds is None:
        raise ValueError(f"Could not create output file {output_path}")
    
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Initialize arrays with NoData values (-1) for each band
    proportion_arrays = {class_name: np.full((sentinel_height, sentinel_width), -1, dtype=np.int16) 
                        for class_name in all_classes}
    
    # Fill arrays with proportion values (scaled to 0-100)
    print("Filling proportion arrays...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        col_s = int(row['col_s'])
        row_s = int(row['row_s'])

        # Skip if out of bounds
        if row_s >= sentinel_height or col_s >= sentinel_width:
            continue

        # Skip if pixel is NoData in Sentinel raster
        if sentinel_mask[row_s, col_s]:
            continue
            
        # Skip if pixel is outside the mask (if mask is provided)
        if mask is not None and mask[row_s, col_s] == 0:
            continue

        # Set proportion values for each class (scaled to 0-100)
        for class_name in all_classes:
            if class_name in row:
                # Multiply by 100 to get percentage (0-100) and convert to int16
                proportion_arrays[class_name][row_s, col_s] = int(row[class_name] * 100)

    # Write arrays to bands
    print("Writing bands to output TIFF...")
    for i, class_name in enumerate(all_classes, start=1):
        band = out_ds.GetRasterBand(i)
        band.WriteArray(proportion_arrays[class_name])
        band.SetDescription(class_name)
        band.SetNoDataValue(-1)  # Use -1 as NoData value
        band.FlushCache()
    
    # Add band descriptions as metadata
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": class_name for i, class_name in enumerate(all_classes)})
    
    # Close dataset
    out_ds = None
    print(f"Multi-band proportion TIFF created at {output_path}")

if __name__ == "__main__":
    wap = 23
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""
    mediane_dir = "mediane" if not moy5m else "mediane_10m"
    file_prefix = "" if not moy5m else "10m_"
    
    # Input paths
    input_csv = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/class_proportions_WAP{wap}_filtered.csv"
    sentinel_path = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_bands/{file_prefix}mediane_clipped_STACK_2023_BandB4_WAP{wap}_deflate.tif"
    
    # Mask path (optional)
    mask_path = f"drone_treated/WAP{wap}_tiles/mask_WAP{wap}_peat.tif"
    # Uncomment the line below to disable mask
    # mask_path = None
    
    # Output path
    output_path = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/proportions_WAP{wap}.tif"
    
    # Create proportion TIFF
    create_proportion_tiff(input_csv, sentinel_path, output_path, mask_path)

