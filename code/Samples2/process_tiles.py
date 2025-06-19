import os
import re
import glob
import numpy as np
import joblib
from osgeo import gdal
from rasters_manager import RastersManager

# Import necessary functions from evaluation5
from evaluation5_previous import (
    create_samples_and_compute,
    extract_features,
    filter_features,
    predict_samples_2,
    save_classification_to_tif,
    filter_isolated_samples,  # Import the filter function
    SamplesSet2
)

def process_all_tiles(rgb_folder, dsm_folder, out_folder, model_path, max_tiles=None, size_patch=16, 
                     start_row=None, start_column=None, nb_wap=32):
    
    """
    Process all corresponding RGB and DSM tiles in the given folders.
    
    Args:
        rgb_folder: Path to the folder containing RGB tiles
        dsm_folder: Path to the folder containing DSM tiles
        out_folder: Path to save classification results
        model_path: Path to the trained model joblib file
        max_tiles: Maximum number of tiles to process (None for all)
        size_patch: Size of the patch for classification
        start_x: Optional starting X coordinate (e.g., '09')
        start_y: Optional starting Y coordinate (e.g., '04')
    """
    start_y = start_column if start_column is not None else None
    start_x = start_row if start_row is not None else None
    assert nb_wap in [32, 23, 12, 99], "nb_wap must be either '32' or '23' to match the tile naming conventions."

    # Create output directory if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)
    
    # Get all RGB tiles
    rgb_files = glob.glob(os.path.join(rgb_folder, "*.tif"))
    
    # Extract tile coordinates and create a mapping
    # Modified pattern to match WAP32_full_transparent_mosaic_group1_XX_YY.tif
    if nb_wap == 32:
        rgb_pattern = re.compile(r'WAP32_full_transparent_mosaic_group1_(\d+)_(\d+)\.tif')
    elif nb_wap == 23:
        rgb_pattern = re.compile(r'Wap23_main_transparent_mosaic_group1_(\d+)_(\d+)\.tif')
    elif nb_wap == 12:
        rgb_pattern = re.compile(r'Wap12_Main_transparent_mosaic_group1_(\d+)_(\d+)\.tif')
    elif nb_wap == 99:
        rgb_pattern = re.compile(r'twin_lake_mosaïc_(\d+)_(\d+)\.tif')
    tiles_info = []

    for rgb_file in rgb_files:
        basename = os.path.basename(rgb_file)
        match = rgb_pattern.search(basename)
        if match:
            x_coord, y_coord = match.groups()
            # DSM file follows pattern WAP32_full_dsm_XX_YY.tif
            if nb_wap == 32:
                dsm_file = os.path.join(dsm_folder, f"WAP32_full_dsm_{x_coord}_{y_coord}.tif")
            elif nb_wap == 23:
                dsm_file = os.path.join(dsm_folder, f"Wap23_main_dsm_{x_coord}_{y_coord}.tif")
            elif nb_wap == 12:
                dsm_file = os.path.join(dsm_folder, f"Wap12_Main_dsm_{x_coord}_{y_coord}.tif")
            elif nb_wap == 99:
                dsm_file = os.path.join(dsm_folder, f"twin_lake_dsm_{x_coord}_{y_coord}.tif")

            # Check if corresponding DSM file exists
            if os.path.exists(dsm_file):
                if nb_wap == 32:
                    out_file = os.path.join(out_folder, f"WAP32_classif_{x_coord}_{y_coord}.tif")
                elif nb_wap == 23:
                    out_file = os.path.join(out_folder, f"WAP23_classif_{x_coord}_{y_coord}.tif")
                elif nb_wap == 12:
                    out_file = os.path.join(out_folder, f"WAP12_classif_{x_coord}_{y_coord}.tif")
                elif nb_wap == 99:
                    out_file = os.path.join(out_folder, f"twin_lake_classif_{x_coord}_{y_coord}.tif")

                tiles_info.append((rgb_file, dsm_file, out_file, x_coord, y_coord))

    # First sort tiles by coordinates
    tiles_info = sort_tiles_by_coordinates(tiles_info)
    
    # Filter tiles by starting coordinates if specified
    if start_x is not None or start_y is not None:
        start_x_int = int(start_x) if start_x is not None else 0
        start_y_int = int(start_y) if start_y is not None else 0
        
        # Since tiles are sorted, find the starting index
        filtered_tiles = []
        for rgb_file, dsm_file, out_file, x_coord, y_coord in tiles_info:
            x_int = int(x_coord)
            y_int = int(y_coord)
            
            # Keep tiles with coordinates >= the starting point
            if (x_int > start_x_int) or (x_int == start_x_int and y_int >= start_y_int):
                filtered_tiles.append((rgb_file, dsm_file, out_file, x_coord, y_coord))
        
        original_count = len(tiles_info)
        tiles_info = filtered_tiles
        print(f"Filtered from {original_count} to {len(tiles_info)} tiles starting from coordinates ({start_x_int}, {start_y_int})")
    
    print(f"Found {len(tiles_info)} matching RGB/DSM tile pairs to process")
    
    # Limit to max_tiles if specified
    if max_tiles is not None and max_tiles < len(tiles_info):
        tiles_info = tiles_info[:max_tiles]
        print(f"Processing {len(tiles_info)} tiles (limited by max_tiles)")
    
    # Load the model once
    model_data = joblib.load(model_path)
    clf = model_data["model"]
    feature_names_model = model_data["feature_names"]
    print(f"Model loaded from {model_path}")
    
    # Process each tile
    for i, (rgb_file, dsm_file, out_file, x_coord, y_coord) in enumerate(tiles_info):
        print(f"\nProcessing tile {i+1}/{len(tiles_info)}: {os.path.basename(rgb_file)} (coordinates: {x_coord}_{y_coord})")
        
        # Get raster dimensions
        ds = gdal.Open(rgb_file)
        width = ds.RasterXSize
        height = ds.RasterYSize
        ds = None
        
        # Process the tile
        process_single_tile(
            rgb_file, 
            dsm_file, 
            out_file,
            clf,
            feature_names_model,
            x_start=0, 
            y_start=0,
            x_end=width, 
            y_end=height,
            size_patch=size_patch
        )
        
        # Clear the rasters manager to free memory
        rm = RastersManager()
        rm.clear_rasters()
        print(f"Cleared RastersManager for tile {i+1}")


def process_single_tile(rgb_path, dsm_path, out_path, clf, feature_names_model, 
                        x_start=0, y_start=0, x_end=None, y_end=None, size_patch=16):
    """
    Process a single tile and save the classification result.
    
    Args:
        rgb_path: Path to RGB raster
        dsm_path: Path to DSM raster
        out_path: Output path for classification result
        clf: Trained classifier model
        feature_names_model: List of feature names required by the model
        x_start, y_start, x_end, y_end: Boundaries for processing
        size_patch: Size of patches for classification
    """
    # Set default bounds if not provided
    if x_end is None or y_end is None:
        ds = gdal.Open(rgb_path)
        if x_end is None:
            x_end = ds.RasterXSize
        if y_end is None:
            y_end = ds.RasterYSize
        ds = None
    
    # Create samples separately from computing features to handle NaNs
    n_samples_x = (x_end - x_start) // size_patch
    n_samples_y = (y_end - y_start) // size_patch
    
    # Create samples set
    samples_set = SamplesSet2(
        ds_path=rgb_path,
        dz_path=dsm_path,
        dt_path=None,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y
    )
    
    # Create samples grid
    print(f"Creating samples grid for tile: {os.path.basename(rgb_path)}")
    samples_set.create_samples_grid(x_start=x_start, y_start=y_start, size_patch=size_patch)
    
    # Identify samples with NaN values before filling neighbors
    nan_positions = identify_nan_samples(samples_set, n_samples_x, n_samples_y)
    print(f"Found {len(nan_positions)} samples with NaN values")
    
    # Set NaN values to zero in the samples
    fix_nan_values(samples_set, nan_positions)
    
    # Now fill neighbors
    print("Computing neighborhood statistics...")
    samples_set.fill_neighbors_all(distance_large=3)
    
    # Extract features
    print("Extracting features...")
    features, positions, feature_names = extract_features(samples_set, n_samples_x, n_samples_y)
    print(f"Features extracted with shape: {features.shape}")
    
    # Filter features to match the model
    features_filtered = filter_features(features, feature_names, feature_names_model)
    
    # Predict
    print("Running prediction...")
    pred_map = predict_samples_2(
        clf, features_filtered, positions, n_samples_x, n_samples_y, 
        samples_set, mask=None, size_patch=size_patch
    )
    
    # Mark transparent (NaN) areas with class 0
    for i_x, i_y in nan_positions:
        pred_map[i_y, i_x] = 0
    
    if nan_positions:
        print(f"Marked {len(nan_positions)} transparent areas as class 0")
    
    # Apply filter to remove isolated samples
    print("Applying isolated samples filter...")
    pred_map_filtered = pred_map#filter_isolated_samples(pred_map)
    
    # Save classification result as TIF
    print(f"Saving classification to: {out_path}")
    save_classification_to_tif(
        pred_map_filtered,  # Use the filtered map
        ref_tif_path=rgb_path,
        out_tif_path=out_path,
        size_patch=size_patch,
        create_qml=False
    )

def identify_nan_samples(samples_set, n_samples_x, n_samples_y):
    """
    Identify samples containing NaN values in their RGB bands.
    
    Args:
        samples_set: The SamplesSet2 object containing samples
        n_samples_x: Number of samples in X direction
        n_samples_y: Number of samples in Y direction
        
    Returns:
        List of tuples (i_x, i_y) for samples with NaN values
    """
    samples_matrix = samples_set.get_samples_matrix()
    nan_positions = []
    
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            s = samples_matrix[i_y][i_x]
            # Check if sample has NaN values in R, G, or B
            if (s.r_mean is None or np.isnan(s.r_mean) or 
                s.g_mean is None or np.isnan(s.g_mean) or 
                s.b_mean is None or np.isnan(s.b_mean)):
                nan_positions.append((i_x, i_y))
    
    return nan_positions

def fix_nan_values(samples_set, nan_positions):
    """
    Set NaN values in samples to zero.
    
    Args:
        samples_set: The SamplesSet2 object containing samples
        nan_positions: List of (i_x, i_y) tuples identifying samples with NaNs
    """
    if not nan_positions:
        return
        
    samples_matrix = samples_set.get_samples_matrix()
    
    for i_x, i_y in nan_positions:
        s = samples_matrix[i_y][i_x]
        
        # Replace RGB values with zeros if they're NaN
        if s.r_mean is None or np.isnan(s.r_mean):
            s.r_mean = 0.0
        if s.g_mean is None or np.isnan(s.g_mean):
            s.g_mean = 0.0
        if s.b_mean is None or np.isnan(s.b_mean):
            s.b_mean = 0.0
            
        # Replace other statistics with zeros if they're NaN
        for attr in ['r_var', 'g_var', 'b_var', 'z_mean', 'z_var']:
            if hasattr(s, attr):
                val = getattr(s, attr)
                if val is None or (isinstance(val, (float, np.float32, np.float64)) and np.isnan(val)):
                    setattr(s, attr, 0.0)
    
    print(f"Fixed NaN values in {len(nan_positions)} samples")

def sort_tiles_by_coordinates(tiles_info):
    """
    Sort tiles by their coordinates: first by row (y), then by column (x).
    
    Args:
        tiles_info: List of tuples (rgb_file, dsm_file, out_file, x_coord, y_coord)
    
    Returns:
        Sorted list of tiles in order: 01_01, 01_02, ..., 02_01, etc.
    """
    # Convert coordinates to integers for proper numerical sorting
    def get_sort_key(tile_info):
        x_coord = int(tile_info[3])
        y_coord = int(tile_info[4])
        return (x_coord, y_coord)  # Sort by row (y) first, then column (x)
    
    return sorted(tiles_info, key=get_sort_key)

if __name__ == "__main__":
    model = "selection13/merged/model_wap_32_5"

    wap = 32
    process_all_tiles(
        rgb_folder=f"drone_treated/WAP{wap}_tiles/rgb",
        dsm_folder=f"drone_treated/WAP{wap}_tiles/dsm",
        out_folder=f"drone_treated/WAP{wap}_tiles/classification_5wd",
        model_path=f"data/samples/selection13/merged/model_wap_32_5.joblib",  # Update to use the new model
        max_tiles=None,  # Process all tiles, or specify a number to limit
        size_patch=16,
        # start_column="05", 
        # start_row="05"   
        nb_wap = wap
    )

    # wap = 23
    # process_all_tiles(
    #     rgb_folder=f"drone_treated/WAP{wap}_tiles/rgb",
    #     dsm_folder=f"drone_treated/WAP{wap}_tiles/dsm",
    #     out_folder=f"drone_treated/WAP{wap}_tiles/classification_wap23_model_16_5",
    #     model_path=f"data/samples/selection16/classifs/model_16_5_grouped/model_16_5_grouped.joblib",  # Update to use the new model
    #     max_tiles=None,  # Process all tiles, or specify a number to limit
    #     size_patch=16,
    #     # start_column="00", 
    #     # start_row="00"   
    #     nb_wap = wap
    # )

    # wap = 12
    # process_all_tiles(
    #     rgb_folder=f"drone_treated/WAP{wap}_tiles/rgb",
    #     dsm_folder=f"drone_treated/WAP{wap}_tiles/dsm",
    #     out_folder=f"drone_treated/WAP{wap}_tiles/classification_wap32_5wd",
    #         model_path="data/samples/selection13/merged/model_wap_32_5.joblib",  # Update to use the new model
    #     max_tiles=None,  # Process all tiles, or specify a number to limit
    #     size_patch=16,
    #     # start_column="00", 
    #     # start_row="00"   
    #     nb_wap = wap
    # )

    # wap = 99
    # process_all_tiles(
    #     rgb_folder=f"drone_treated/WAP{wap}_tiles/rgb",
    #     dsm_folder=f"drone_treated/WAP{wap}_tiles/dsm",
    #     out_folder=f"drone_treated/WAP{wap}_tiles/classification_wap32_5wd",
    #         model_path="data/samples/selection13/merged/model_wap_32_5.joblib",  # Update to use the new model
    #     max_tiles=None,  # Process all tiles, or specify a number to limit
    #     size_patch=16,
    #     # start_column="00", 
    #     # start_row="00"   
    #     nb_wap = wap
    # )


    
