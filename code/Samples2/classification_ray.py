import os
import re
import glob
import numpy as np
import joblib
from osgeo import gdal
from rasters_manager import RastersManager
import ray
import matplotlib.pyplot as plt
from samples_set2 import SamplesSet2

# Import necessary functions from evaluation5 and evaluation5_previous
from evaluation5 import (
    extract_features, 
    filter_features,
    create_samples_and_compute,
    predict_samples_2 as predict_samples_model16_7,
    filter_isolated_samples
)

from evaluation5_previous import (
    predict_samples_2 as predict_samples_no_chicoutai
)

def process_all_tiles(rgb_folder, dsm_folder, out_folder, model1_path, model2_path, 
                     max_tiles=None, size_patch=16, start_row=None, start_column=None):
    """
    Process all corresponding RGB and DSM tiles in the given folders using two models.
    
    Args:
        rgb_folder: Path to the folder containing RGB tiles
        dsm_folder: Path to the folder containing DSM tiles
        out_folder: Path to save classification results
        model1_path: Path to the trained no_chicoutai model
        model2_path: Path to the trained 16_7 model
        max_tiles: Maximum number of tiles to process (None for all)
        size_patch: Size of the patch for classification
        start_row: Optional starting row coordinate
        start_column: Optional starting column coordinate
        nb_wap: WAP number (32, 23, 12, or 99)
    """
    start_y = start_column if start_column is not None else None
    start_x = start_row if start_row is not None else None
    

    # Create output directory if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)
    
    # Get all RGB tiles
    rgb_files = glob.glob(os.path.join(rgb_folder, "*.tif"))
    
    # Extract tile coordinates using a generic pattern that matches any prefix
    rgb_pattern = re.compile(r'.*_(\d+)_(\d+)\.tif$')

    tiles_info = []

    for rgb_file in rgb_files:
        basename = os.path.basename(rgb_file)
        match = rgb_pattern.search(basename)
        if match:
            x_coord, y_coord = match.groups()
            
            # Look for any DSM file with the same coordinates
            dsm_pattern = f'*_{x_coord}_{y_coord}.tif'
            dsm_matches = glob.glob(os.path.join(dsm_folder, dsm_pattern))
            
            if dsm_matches:
                dsm_file = dsm_matches[0]  # Take the first match if multiple exist
                out_file = os.path.join(out_folder, f"merged_classif_{x_coord}_{y_coord}.tif")
                tiles_info.append((rgb_file, dsm_file, out_file, x_coord, y_coord))
            else:
                print(f"Warning: No matching DSM file found for {basename} with coordinates {x_coord}_{y_coord}")

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
    
    # Load both models once
    model1_data = joblib.load(model1_path)
    model1 = model1_data["model"]
    feature_names_model1 = model1_data["feature_names"]
    print(f"Model 1 (no_chicoutai) loaded from {model1_path}")
    
    model2_data = joblib.load(model2_path)
    model2 = model2_data["model"]
    feature_names_model2 = model2_data["feature_names"]
    print(f"Model 2 (16_7) loaded from {model2_path}")
    
    # Process each tile in parallel using Ray
    tiles = []
    for i, (rgb_file, dsm_file, out_file, x_coord, y_coord) in enumerate(tiles_info):
        tiles.append(process_single_tile.remote(
            rgb_file, 
            dsm_file, 
            out_file,
            model1,
            model2,
            feature_names_model1,
            feature_names_model2,
            x_start=0, 
            y_start=0,
            size_patch=size_patch,
        ))
    ray.get(tiles)
    
    # Create a single QML file after all tiles have been processed
    qml_path = os.path.join(out_folder, f"merged_classification_colormap.qml")
    create_qgis_colormap(qml_path)
    print(f"Created a single QML file at {qml_path} for all processed tiles")

@ray.remote
def process_single_tile(rgb_path, dsm_path, out_path, model1, model2, feature_names_model1, feature_names_model2,
                        x_start=0, y_start=0, x_end=None, y_end=None, size_patch=16):
    """
    Process a single tile using both models and merge their results.
    
    Args:
        rgb_path: Path to RGB raster
        dsm_path: Path to DSM raster
        out_path: Output path for classification result
        model1: Trained no_chicoutai model
        model2: Trained 16_7 model
        feature_names_model1: Feature names for model1
        feature_names_model2: Feature names for model2
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
    
    # Filter features for each model
    features_filtered1 = filter_features(features, feature_names, feature_names_model1)
    features_filtered2 = filter_features(features, feature_names, feature_names_model2)
    
    # Predict with both models
    print(f"Running prediction with no_chicoutai model...")
    pred_map1 = predict_samples_no_chicoutai(
        model1, features_filtered1, positions, n_samples_x, n_samples_y, 
        samples_set, mask=None, size_patch=size_patch
    )
    
    print(f"Running prediction with 16_7 model...")
    pred_map2 = predict_samples_model16_7(
        model2, features_filtered2, positions, n_samples_x, n_samples_y, 
        samples_set, mask=None, size_patch=size_patch, grouped=False
    )
    
    # Mark transparent (NaN) areas with class 0
    for i_x, i_y in nan_positions:
        pred_map1[i_y, i_x] = 0
        pred_map2[i_y, i_x] = 0
    
    if nan_positions:
        print(f"Marked {len(nan_positions)} transparent areas as class 0")
    
    # Merge predictions
    print("Merging predictions from both models...")
    merged_map = merge_predictions(pred_map1, pred_map2)
    
    # Apply filter to remove isolated samples
    print("Applying isolated samples filter...")
    merged_map_filtered = filter_isolated_samples(merged_map)
    
    # Save classification result as TIF
    print(f"Saving merged classification to: {out_path}")
    save_classification_to_tif(
        merged_map_filtered,
        ref_tif_path=rgb_path,
        out_tif_path=out_path,
        size_patch=size_patch,
        create_qml=False  # Don't create a QML for each tile, we'll create a single one at the end
    )

    # Release resources
    rm = RastersManager()
    rm.clear_rasters()
    
    return out_path

def merge_predictions(pred_map1, pred_map2):
    """
    Merge predictions from the two models according to the specified rules.
    
    pred_map1: Predictions from the no_chicoutai model
    pred_map2: Predictions from the 16_7 model
    
    Returns: Merged prediction map with new class IDs
    """
    # Define the merged classes
    MERGED_CLASSES = {
        1: "Pure_Lichen",      # from class 8 of 16_7
        2: "Degraded_Lichen",  # from class 6 of 16_7
        3: "Green",            # from classes 2,7,9 of 16_7 and class 3 of no_chicoutai
        4: "Sphagnum",         # from classes 4,10 of 16_7 and class 5 of no_chicoutai
        5: "Depression",       # from classes 1,3 of 16_7 and classes 2,7 of no_chicoutai
        6: "Water",            # from class 5 of 16_7 and class 6 of no_chicoutai
        0: "No Data",
        255: "No Data"
    }
    
    # Create the merged prediction map
    n_samples_y, n_samples_x = pred_map1.shape
    merged_map = np.zeros((n_samples_y, n_samples_x), dtype=np.uint8)
    
    # Process all samples
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            val1 = pred_map1[i_y, i_x]
            val2 = pred_map2[i_y, i_x]
            
            # Special cases for 0 and 255 (No Data)
            if val1 == 0 or val1 == 255:
                merged_map[i_y, i_x] = val1
                continue
                
            # For classes 1 (Chicoutai) and 4 (Lichen) from no_chicoutai model,
            # use the predictions from the 16_7 model
            if val1 == 1 or val1 == 4:  # Chicoutai or Lichen
                # Map 16_7 model classes
                if val2 == 8:  # peat_pure_lichen
                    merged_map[i_y, i_x] = 1  # Pure_Lichen
                elif val2 == 6:  # peat_degraded_lichen
                    merged_map[i_y, i_x] = 2  # Degraded_Lichen
                elif val2 in [2, 7, 9]:  # depression_green, peat_green, through_green
                    merged_map[i_y, i_x] = 3  # Green
                elif val2 in [4, 10]:  # depression_sphagnum, through_sphagnum
                    merged_map[i_y, i_x] = 4  # Sphagnum
                elif val2 in [1, 3]:  # depression_fen, depression_peat
                    merged_map[i_y, i_x] = 5  # Depression
                elif val2 == 5:  # depression_water
                    merged_map[i_y, i_x] = 6  # Water
                else:
                    merged_map[i_y, i_x] = 0  # Default to No Data
            else:
                # For other classes from no_chicoutai model, map them directly
                if val1 == 3:  # green_depression
                    merged_map[i_y, i_x] = 3  # Green
                elif val1 == 5:  # sphaignes
                    merged_map[i_y, i_x] = 4  # Sphagnum
                elif val1 in [2, 7]:  # dry_depression, black_depression
                    merged_map[i_y, i_x] = 5  # Depression
                elif val1 == 6:  # watered_depression
                    merged_map[i_y, i_x] = 6  # Water
                else:
                    merged_map[i_y, i_x] = 0  # Default to No Data
    
    return merged_map

def create_qgis_colormap(output_qml):
    """
    Creates a QGIS colormap file (.qml) for the merged classification.
    
    Args:
        output_qml: Path to save the QML file
    """
    # Merged class definitions
    class_labels = {
        1: "Pure_Lichen",
        2: "Degraded_Lichen",
        3: "Green",
        4: "Sphagnum",
        5: "Depression",
        6: "Water",
        0: "No Data",
        255: "No Data"
    }
    
    # Colors for each class
    color_dict = {
        1: [220, 220, 220],   # Pure_Lichen: very light gray
        2: [150, 150, 150],   # Degraded_Lichen: gray
        3: [100, 180, 100],   # Green: green
        4: [139, 69, 19],     # Sphagnum: dark orange
        5: [153, 136, 0],     # Depression: yellow/black
        6: [65, 105, 225],    # Water: blue
        0: [0, 0, 0],         # No Data: black
        255: [0, 0, 0]        # No Data: black
    }

    # QGIS QML template
    qml_template = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis hasScaleBasedVisibilityFlag="0" maxScale="0" version="3.22.4-Białowieża" minScale="1e+08" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" fetchMode="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option type="bool" value="false" name="WMSBackgroundLayer"/>
      <Option type="bool" value="false" name="WMSPublishDataSourceUrl"/>
      <Option type="int" value="0" name="embeddedWidgets/count"/>
      <Option type="QString" value="Value" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" value="" name="name"/>
      <Option name="properties"/>
      <Option type="QString" value="collection" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" nodataColor="" type="paletted" opacity="1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
{color_entries}
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation invertColors="0" saturation="0" colorizeRed="255" colorizeOn="0" colorizeGreen="128" colorizeBlue="128" grayscaleMode="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    
    # Generate color entries for the XML
    color_entries = []
    for class_value, label in sorted(class_labels.items()):
        rgb = color_dict.get(class_value, [0, 0, 0])
        # Special handling for transparency
        alpha = 0 if class_value in [0, 255] else 255
        
        # Format: <paletteEntry color="#RRGGBB" alpha="255" value="1" label="1"/>
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        entry = f'        <paletteEntry color="{hex_color}" alpha="{alpha}" value="{class_value}" label="{label}"/>'
        color_entries.append(entry)
    
    # Insert color entries into template
    qml_content = qml_template.format(color_entries="\n".join(color_entries))
    
    # Write the QML file
    with open(output_qml, 'w') as f:
        f.write(qml_content)
    
    print(f"QGIS color map saved to {output_qml}")

def save_classification_to_tif(pred_map, ref_tif_path, out_tif_path, size_patch=16, create_qml=False):
    """
    Save the classification map (pred_map) as a .tif file,
    using the georeference and size of the reference raster.
    
    Args:
        pred_map: Classification map as a numpy array
        ref_tif_path: Path to the reference raster for georeference
        out_tif_path: Output path to save the classification raster
        size_patch: Patch size (default: 16)
        create_qml: If True, create a QML file for QGIS with the same color palette
    """
    ds = gdal.Open(ref_tif_path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    n_samples_y, n_samples_x = pred_map.shape
    out_img = np.zeros((height, width), dtype=np.uint8)
    
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            x0 = i_x * size_patch
            y0 = i_y * size_patch
            out_img[x0:x0+size_patch, y0:y0+size_patch] = pred_map[i_y, i_x]
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_tif_path, width, height, 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(out_img)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"Classification map saved to {out_tif_path}")
    
    # Create QML file only if requested
    if create_qml:
        qml_path = out_tif_path.replace('.tif', '.qml')
        create_qgis_colormap(qml_path)

def identify_nan_samples(samples_set, n_samples_x, n_samples_y):
    """
    Identify samples containing NaN values in their RGB bands.
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
    Sort tiles by their coordinates: first by row (x), then by column (y).
    """
    # Convert coordinates to integers for proper numerical sorting
    def get_sort_key(tile_info):
        x_coord = int(tile_info[3])
        y_coord = int(tile_info[4])
        return (x_coord, y_coord)  # Sort by row (x) first, then column (y)
    
    return sorted(tiles_info, key=get_sort_key)

if __name__ == "__main__":
    ray.init(dashboard_host="127.0.0.1")
    
    
    # Path to model 1 (no_chicoutai)
    model1_path = "data/samples/selection14/model_wap32_no_chicoutai.joblib"
    
    # Path to model 2 (16_7)
    model2_path = "data/samples/selection16/classifs/model_16_7/model_16_7.joblib"
    
    process_all_tiles(
        rgb_folder=f"Konstantin/Chesnay_tiles/rgb",
        dsm_folder=f"Konstantin/Chesnay_tiles/dsm",
        out_folder=f"Konstantin/Chesnay_tiles/merged_classification",
        model1_path=model1_path,
        model2_path=model2_path,
        max_tiles=None,  # Process all tiles
        size_patch=8,
    )
    
    ray.shutdown()



