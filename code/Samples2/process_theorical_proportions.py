"""
Process theorical proportions from classification data into a multi-band TIFF file.
This script combines functionalities from sentinel_proportion_2.py and create_proportion_tif_2.py
to create a more streamlined workflow.
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from tqdm import tqdm

def pixel_to_geo(transform, px, py):
    """
    Convert pixel coordinates (col, row) to geo coordinates (x, y) of the top-left corner
    """
    x = transform[0] + px * transform[1] + py * transform[2]
    y = transform[3] + px * transform[4] + py * transform[5]
    return x, y

def geo_to_pixel(transform, x, y):
    """
    Convert geo coordinates (x, y) to pixel coordinates (col, row)
    """
    inv_det = 1 / (transform[1] * transform[5] - transform[2] * transform[4])
    px = inv_det * (transform[5] * (x - transform[0]) - transform[2] * (y - transform[3]))
    py = inv_det * (-transform[4] * (x - transform[0]) + transform[1] * (y - transform[3]))
    return int(round(px)), int(round(py))

def sentinel_to_drone_bounds(col_s, row_s, sentinel_ds, drone_ds):
    """
    For a Sentinel pixel (col_s, row_s), return the bounds xmin, xmax, ymin, ymax
    of drone pixels covered by this Sentinel pixel.
    
    Args:
        col_s, row_s: Column and row indices of the Sentinel pixel
        sentinel_ds: GDAL dataset for Sentinel data
        drone_ds: GDAL dataset for drone data
        
    Returns:
        xmin, xmax, ymin, ymax: Bounds of drone pixels
    """
    gt_sentinel = sentinel_ds.GetGeoTransform()
    gt_drone = drone_ds.GetGeoTransform()
    
    # Get geo coordinates of the top-left corner of the Sentinel pixel
    x_min, y_max = pixel_to_geo(gt_sentinel, col_s, row_s)
    # Get geo coordinates of the bottom-right corner of the Sentinel pixel
    x_max, y_min = pixel_to_geo(gt_sentinel, col_s + 1, row_s + 1)
    
    # Convert to pixel coordinates in the drone image
    col_min, row_min = geo_to_pixel(gt_drone, x_min, y_min)  # bottom-left
    col_max, row_max = geo_to_pixel(gt_drone, x_max, y_max)  # top-right
    
    # Ensure xmin < xmax and ymin < ymax
    xmin = min(col_min, col_max)
    xmax = max(col_min, col_max)
    ymin = min(row_min, row_max)
    ymax = max(row_min, row_max)
    
    return xmin, xmax, ymin, ymax

def load_mask(mask_path, ref_ds):
    """
    Load a mask file and reproject it to match the reference dataset if needed
    
    Args:
        mask_path: Path to the mask file (should be binary, 1 for valid pixels)
        ref_ds: Reference GDAL dataset
        
    Returns:
        mask_array: Binary mask array (1 for valid pixels, 0 for masked pixels)
    """
    print(f"Loading mask from {mask_path}...")
    
    # Open the mask file
    mask_ds = gdal.Open(mask_path)
    if mask_ds is None:
        raise ValueError(f"Could not open mask file: {mask_path}")
    
    # Get reference parameters
    ref_transform = ref_ds.GetGeoTransform()
    ref_projection = ref_ds.GetProjection()
    ref_width = ref_ds.RasterXSize
    ref_height = ref_ds.RasterYSize
    
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

def process_theorical_proportions(classification_path, sentinel_path, output_path, mask_path=None, class_names=None, through_class_names=None):
    """
    Process classification data to create a multi-band TIFF file with class proportions.
    
    Args:
        classification_path: Path to the classification raster (from RF model)
        sentinel_path: Path to the Sentinel-2 raster (for georeference)
        output_path: Path to save the output multi-band TIFF
        mask_path: Optional path to a mask file (only pixels where mask is 1 will be processed)
        class_names: Dictionary mapping class values to names (e.g., {1: "Pure_Lichen", 2: "Degraded_Lichen"})
        through_class_names: List of class names to be considered for through_proportion calculation
    """
    print(f"Processing theoretical proportions from {classification_path} using {sentinel_path} as reference")
    
    # Default class mapping if not provided
    if class_names is None:
        class_names = {
            1: "Pure_Lichen",
            2: "Degraded_Lichen",
            3: "Green",
            4: "Sphagnum",
            5: "Depression",
            6: "Water",
            0: "No Data",
        }
        
    if through_class_names is None:
        through_class_names = ["Sphagnum", "Depression", "Water"]
    
    # Open datasets
    ds_sentinel = gdal.Open(sentinel_path)
    ds_class = gdal.Open(classification_path)
    
    if ds_sentinel is None:
        raise ValueError(f"Could not open Sentinel raster: {sentinel_path}")
    
    if ds_class is None:
        raise ValueError(f"Could not open classification raster: {classification_path}")
    
    # Get Sentinel dimensions and georeference info
    sentinel_width = ds_sentinel.RasterXSize
    sentinel_height = ds_sentinel.RasterYSize
    geo_transform = ds_sentinel.GetGeoTransform()
    projection = ds_sentinel.GetProjection()
    
    # Read classification data
    class_data = ds_class.GetRasterBand(1).ReadAsArray()
    
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
        mask = load_mask(mask_path, ds_sentinel)
    
    # Create output raster (multi-band)
    # Define classes to include in the output TIFF
    classes_to_include = [class_name for val, class_name in class_names.items() if val != 0]
    # Add additional combined classes
    classes_to_include.append("all_lichen")
    classes_to_include.append("through_proportion")
    
    # Create the output raster
    driver = gdal.GetDriverByName('GTiff')
    num_bands = len(classes_to_include)
    out_ds = driver.Create(output_path, sentinel_width, sentinel_height, num_bands, gdal.GDT_Int16, 
                          options=['COMPRESS=DEFLATE', 'TILED=YES'])
    
    if out_ds is None:
        raise ValueError(f"Could not create output file {output_path}")
    
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Initialize arrays with NoData values (-1) for each band
    proportion_arrays = {class_name: np.full((sentinel_height, sentinel_width), -1, dtype=np.int16) 
                        for class_name in classes_to_include}
    
    # Process each Sentinel pixel
    print("Computing class proportions and filling arrays...")
    for row_s in tqdm(range(sentinel_height)):
        for col_s in range(sentinel_width):
            # Skip if pixel is NoData in Sentinel raster
            if sentinel_mask[row_s, col_s]:
                continue
                
            # Skip if pixel is outside the mask (if mask is provided)
            if mask is not None and mask[row_s, col_s] == 0:
                continue
                
            # Get drone pixel bounds for this Sentinel pixel
            xmin, xmax, ymin, ymax = sentinel_to_drone_bounds(col_s, row_s, ds_sentinel, ds_class)
            
            # Skip if out of bounds
            if (xmin < 0 or ymin < 0 or 
                xmax >= class_data.shape[1] or 
                ymax >= class_data.shape[0]):
                continue
            
            # Extract classification data for this region
            subclass = class_data[ymin:ymax, xmin:xmax]
            
            # Skip if empty
            if subclass.size == 0:
                continue
            
            # Count total valid pixels (non-nodata)
            valid_pixels = np.sum(subclass != 0)
            if valid_pixels == 0:
                continue
            
            # Compute proportions for each class
            class_proportions = {}
            for class_val, class_name in class_names.items():
                if class_val == 0:  # Skip nodata
                    continue
                count = np.sum(subclass == class_val)
                prop = count / valid_pixels if valid_pixels > 0 else 0
                
                # Store proportion for this class (scaled to percentage 0-100)
                if class_name in proportion_arrays:
                    proportion_arrays[class_name][row_s, col_s] = int(prop * 100)
            
            # Calculate additional metrics (all_lichen and through_proportion)
            # All lichen (sum of Pure_Lichen and Degraded_Lichen)
            all_lichen = (
                (proportion_arrays["Pure_Lichen"][row_s, col_s] if "Pure_Lichen" in proportion_arrays else 0) +
                (proportion_arrays["Degraded_Lichen"][row_s, col_s] if "Degraded_Lichen" in proportion_arrays else 0)
            )
            if "all_lichen" in proportion_arrays and all_lichen >= 0:
                proportion_arrays["all_lichen"][row_s, col_s] = all_lichen
            
            # through_proportion (sum of specified through_class_names)
            through_proportion = 0
            for class_name in through_class_names:
                if class_name in proportion_arrays and proportion_arrays[class_name][row_s, col_s] >= 0:
                    through_proportion += proportion_arrays[class_name][row_s, col_s]
                    
            if "through_proportion" in proportion_arrays:
                proportion_arrays["through_proportion"][row_s, col_s] = through_proportion
    
    # Write arrays to bands
    print("Writing bands to output TIFF...")
    for i, class_name in enumerate(classes_to_include, start=1):
        band = out_ds.GetRasterBand(i)
        band.WriteArray(proportion_arrays[class_name])
        band.SetDescription(class_name)
        band.SetNoDataValue(-1)  # Use -1 as NoData value
        band.FlushCache()
    
    # Add band descriptions as metadata
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": class_name for i, class_name in enumerate(classes_to_include)})
    
    # Close dataset
    out_ds = None
    print(f"Multi-band proportion TIFF created at {output_path}")
    return output_path

if __name__ == "__main__":
    # WAP configuration parameters
    wap = 23  # WAP site number (e.g., 23 or 32)
    use_peat = False  # Whether to use the peat dataset paths
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False  # Whether to use moy5m data
    moy5m_suffix = "_moy5m" if moy5m else ""
    mediane_dir = "mediane" if not moy5m else "mediane_10m"
    file_prefix = "" if not moy5m else "10m_"
    
    # Input paths
    classification_path = f"drone_treated/WAP{wap}_tiles/WAP{wap}_classif_merged.tif"
    sentinel_path = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_bands/{file_prefix}mediane_clipped_STACK_2023_BandB4_WAP{wap}_deflate.tif"
    
    # Mask path (optional)
    mask_path = f"drone_treated/WAP{wap}_tiles/mask_WAP{wap}_peat.tif"
    # Uncomment the line below to disable mask
    # mask_path = None
    
    # Output path
    output_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/proportions_WAP{wap}.tif"
    
    # Define class labels
    class_labels = {
        1: "Pure_Lichen",
        2: "Degraded_Lichen",
        3: "Green",
        4: "Sphagnum",
        5: "Depression",
        6: "Water",
        0: "No Data",
    }
    
    through_class_labels = ["Sphagnum", "Depression", "Water"]
    
    print(f"Processing WAP{wap} data with {'peat' if use_peat else 'standard'} dataset at {resolution}m resolution...")
    
    # Process the data
    process_theorical_proportions(
        classification_path=classification_path,
        sentinel_path=sentinel_path,
        output_path=output_path,
        mask_path=mask_path,
        class_names=class_labels,
        through_class_names=through_class_labels
    )
    
    print(f"Processed theoretical proportions saved to {output_path}")
