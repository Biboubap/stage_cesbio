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

def create_proportion_tiff(input_csv, sentinel_path, output_path):
    """
    Create a multi-band TIFF from class proportion data from a CSV file.
    
    Args:
        input_csv: Path to CSV containing class proportions (from sentinel_proportion.py)
        sentinel_path: Path to a Sentinel-2 raster for georeference
        output_path: Path to save the output multi-band TIFF
    """
    print(f"Creating proportion TIFF from {input_csv}")
    
    # Load class proportions data
    df = pd.read_csv(input_csv)
    
    # Define classes and get max row/col coordinates
    classes = [
        "chicoutai",
        "dry_depression",
        "green_depression", 
        "lichen",
        "sphaignes",
        "watered_depression",
        "black_depression",
        "none"
    ]
    
    # Calculate derived classes
    df['chicoutai_green'] = df['chicoutai'] + df['green_depression']
    df['through_proportion'] = df['sphaignes'] + df['dry_depression'] + df['black_depression'] + df['watered_depression']
    
    all_classes = classes + ['chicoutai_green', 'through_proportion']
    
    # Get dimensions of Sentinel raster
    ds_sentinel = gdal.Open(sentinel_path)
    if ds_sentinel is None:
        raise ValueError(f"Could not open Sentinel raster: {sentinel_path}")
        
    sentinel_width = ds_sentinel.RasterXSize
    sentinel_height = ds_sentinel.RasterYSize
    geo_transform = ds_sentinel.GetGeoTransform()
    projection = ds_sentinel.GetProjection()
    
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
    superresolution = True  # Use 5m resolution (True) or 10m resolution (False)
    use_peat = True
    peat_suffix = "_peat" if use_peat else ""
    resolution_suffix = "" if superresolution else "_10m"
    
    # Set path modifiers based on superresolution flag
    mediane_dir = "mediane" if superresolution else "mediane_10m"
    file_prefix = "" if superresolution else "10m_"
    
    input_csv = f"data/samples/selection15/regression_wap{wap}{peat_suffix}{resolution_suffix}/class_proportions_WAP{wap}_filtered.csv"
    sentinel_path = f"DataCubeS2/BandsS22023_WAP{wap}{peat_suffix}/{mediane_dir}/{file_prefix}mediane_clipped_STACK_2023_BandB2_WAP{wap}_deflate.tif"
    output_path = f"data/samples/selection15/regression_wap{wap}{peat_suffix}{resolution_suffix}/proportions_WAP{wap}.tif"
    create_proportion_tiff(input_csv, sentinel_path, output_path)

