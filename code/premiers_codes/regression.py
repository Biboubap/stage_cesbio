"""
regression.py

A comprehensive tool for processing Sentinel-2 data with classification maps to:
1. Generate proportion maps from classification data (theoretical proportions)
2. Generate predictions using pre-trained models
3. Compare and evaluate results with metrics and visualizations

Each function can be used independently or as part of a complete workflow.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import joblib
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# -- UTILITY FUNCTIONS --

def pixel_to_geo(transform, px, py):
    """Convert pixel coordinates to geo coordinates"""
    x = transform[0] + px * transform[1] + py * transform[2]
    y = transform[3] + px * transform[4] + py * transform[5]
    return x, y

def geo_to_pixel(transform, x, y):
    """Convert geo coordinates to pixel coordinates"""
    inv_det = 1 / (transform[1] * transform[5] - transform[2] * transform[4])
    px = inv_det * (transform[5] * (x - transform[0]) - transform[2] * (y - transform[3]))
    py = inv_det * (-transform[4] * (x - transform[0]) + transform[1] * (y - transform[3]))
    return int(round(px)), int(round(py))

def sentinel_to_drone_bounds(col_s, row_s, sentinel_ds, drone_ds):
    """Get drone pixel bounds for a sentinel pixel"""
    gt_sentinel = sentinel_ds.GetGeoTransform()
    gt_drone = drone_ds.GetGeoTransform()
    
    # Get geo coordinates of the sentinel pixel
    x_min, y_max = pixel_to_geo(gt_sentinel, col_s, row_s)
    x_max, y_min = pixel_to_geo(gt_sentinel, col_s + 1, row_s + 1)
    
    # Convert to pixel coordinates in drone image
    col_min, row_min = geo_to_pixel(gt_drone, x_min, y_min)
    col_max, row_max = geo_to_pixel(gt_drone, x_max, y_max)
    
    # Ensure correct ordering
    xmin = min(col_min, col_max)
    xmax = max(col_min, col_max)
    ymin = min(row_min, row_max)
    ymax = max(row_min, row_max)
    
    return xmin, xmax, ymin, ymax

def load_mask(mask_path, ref_ds=None, ref_transform=None, ref_projection=None, 
              ref_width=None, ref_height=None):
    """
    Load a mask file and reproject it to match reference data if needed
    
    Args:
        mask_path: Path to the mask file
        ref_ds: Reference GDAL dataset (optional if other ref_* params provided)
        ref_transform, ref_projection, ref_width, ref_height: Reference parameters
        
    Returns:
        Binary mask array (1 for valid pixels)
    """
    print(f"Loading mask from {mask_path}...")
    
    # Open the mask file
    mask_ds = gdal.Open(mask_path)
    if mask_ds is None:
        raise ValueError(f"Could not open mask file: {mask_path}")
    
    # Get reference parameters
    if ref_ds is not None:
        ref_transform = ref_ds.GetGeoTransform()
        ref_projection = ref_ds.GetProjection()
        ref_width = ref_ds.RasterXSize
        ref_height = ref_ds.RasterYSize
    
    # Get mask parameters
    mask_transform = mask_ds.GetGeoTransform()
    mask_projection = mask_ds.GetProjection()
    mask_width = mask_ds.RasterXSize
    mask_height = mask_ds.RasterYSize
    
    # If dimensions and projection match, read directly
    if (mask_width == ref_width and mask_height == ref_height and 
        mask_transform == ref_transform and mask_projection == ref_projection):
        mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
    else:
        # Reproject mask to match reference dataset
        print(f"Reprojecting mask from {mask_width}x{mask_height} to {ref_width}x{ref_height}...")
        mem_driver = gdal.GetDriverByName('MEM')
        mask_reprojected = mem_driver.Create('', ref_width, ref_height, 1, gdal.GDT_Byte)
        mask_reprojected.SetGeoTransform(ref_transform)
        mask_reprojected.SetProjection(ref_projection)
        
        # Reproject
        gdal.ReprojectImage(mask_ds, mask_reprojected, mask_projection, ref_projection, 
                           gdal.GRA_NearestNeighbour)
        
        # Read reprojected mask
        mask_array = mask_reprojected.GetRasterBand(1).ReadAsArray()
    
    # Ensure mask is binary (0 or 1) - following the approach in process_theoretical_proportions.py
    binary_mask = (mask_array > 0).astype(np.uint8)
    
    print(f"Mask loaded. Valid pixels: {np.sum(binary_mask)}/{binary_mask.size} ({np.sum(binary_mask)/binary_mask.size*100:.2f}%)")
    
    return binary_mask

def load_sentinel_data(bands_dir, indices_dir=None, reference_path=None):
    """
    Load Sentinel bands and indices as features
    
    Args:
        bands_dir: Directory containing band TIF files
        indices_dir: Directory containing indices TIF files (optional)
        reference_path: Path to a reference raster for dimensions (optional)
        
    Returns:
        features: NumPy array of shape (n_features, rows, cols)
        band_names: List of feature names
        geo_transform: GeoTransform of the input raster
        projection: Projection of the input raster
    """
    # List all tif files in bands directory
    band_files = sorted([os.path.join(bands_dir, f) for f in os.listdir(bands_dir) 
                        if f.lower().endswith('.tif') and not f.endswith('.aux.xml')])
    
    # List all tif files in indices directory if provided
    index_files = []
    if indices_dir and os.path.exists(indices_dir):
        index_files = sorted([os.path.join(indices_dir, f) for f in os.listdir(indices_dir) 
                             if f.lower().endswith('.tif') and not f.endswith('.aux.xml')])
    
    print(f"Found {len(band_files)} band files and {len(index_files)} index files")
    
    # Use reference raster if provided, otherwise use first band file
    if reference_path and os.path.exists(reference_path):
        ref_ds = gdal.Open(reference_path)
        print(f"Using dimensions from reference: {reference_path}")
    else:
        ref_ds = gdal.Open(band_files[0])
        print(f"Using dimensions from first band: {band_files[0]}")
        
    if ref_ds is None:
        raise ValueError(f"Could not open reference raster")
    
    width = ref_ds.RasterXSize
    height = ref_ds.RasterYSize
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    bands = []
    band_names = []
    
    # Load band files
    for tif in band_files:
        ds = gdal.Open(tif)
        if ds is None:
            print(f"Warning: Could not open {tif}")
            continue
            
        # Check if reprojection is needed
        if (ds.RasterXSize != width or ds.RasterYSize != height):
            print(f"Reprojecting {tif} to match reference dimensions")
            mem_driver = gdal.GetDriverByName('MEM')
            reprojected_ds = mem_driver.Create('', width, height, 1, ds.GetRasterBand(1).DataType)
            reprojected_ds.SetGeoTransform(geo_transform)
            reprojected_ds.SetProjection(projection)
            gdal.ReprojectImage(ds, reprojected_ds, ds.GetProjection(), projection, gdal.GRA_NearestNeighbour)
            arr = reprojected_ds.GetRasterBand(1).ReadAsArray()
        else:
            arr = ds.GetRasterBand(1).ReadAsArray()
            
        bands.append(arr)
        file_name = os.path.splitext(os.path.basename(tif))[0]
        band_name = file_name.replace('mediane_clipped_STACK_2023_Band', '')
        band_names.append(f"band_{band_name}")
    
    # Load index files if provided
    if indices_dir:
        for tif in index_files:
            ds = gdal.Open(tif)
            if ds is None:
                print(f"Warning: Could not open {tif}")
                continue
                
            # Check if reprojection is needed
            if (ds.RasterXSize != width or ds.RasterYSize != height):
                print(f"Reprojecting {tif} to match reference dimensions")
                mem_driver = gdal.GetDriverByName('MEM')
                reprojected_ds = mem_driver.Create('', width, height, 1, ds.GetRasterBand(1).DataType)
                reprojected_ds.SetGeoTransform(geo_transform)
                reprojected_ds.SetProjection(projection)
                gdal.ReprojectImage(ds, reprojected_ds, ds.GetProjection(), projection, gdal.GRA_NearestNeighbour)
                arr = reprojected_ds.GetRasterBand(1).ReadAsArray()
            else:
                arr = ds.GetRasterBand(1).ReadAsArray()
                
            bands.append(arr)
            file_name = os.path.splitext(os.path.basename(tif))[0]
            index_name = file_name.replace('mediane_clipped_', '')
            band_names.append(f"index_{index_name}")
    
    features = np.stack(bands, axis=0)  # (n_features, rows, cols)
    print(f"Loaded {len(band_names)} features with dimensions {features.shape}")
    return features, band_names, geo_transform, projection

def load_regression_model(model_path):
    """Load a regression model from a joblib file"""
    print(f"Loading model from {model_path}...")
    
    model_data = joblib.load(model_path)
    
    # Check if it's a dictionary containing the model and metadata
    if isinstance(model_data, dict):
        if "model" in model_data:
            model = model_data["model"]
            feature_names = model_data.get("feature_names", None)
            return model, feature_names
    
    # Direct model without metadata
    return model_data, None


# -- MAIN FUNCTIONS --

def process_proportions(classification_path, sentinel_path, output_path, mask_path=None, 
                        class_names=None, through_class_names=None):
    """
    Process classification data to create a multi-band TIFF file with class proportions.
    
    Args:
        classification_path: Path to the classification raster
        sentinel_path: Path to a Sentinel-2 raster (for georeference)
        output_path: Path to save the output multi-band TIFF
        mask_path: Optional path to a mask file (only pixels where mask is 1 will be processed)
        class_names: Dictionary mapping class values to names (e.g., {1: "Pure_Lichen", ...})
        through_class_names: List of class names for through_proportion calculation
        
    Returns:
        Path to the created proportion map
    """
    print(f"\n{'='*80}\nProcessing theoretical proportions\n{'='*80}")
    print(f"Processing classification data from {classification_path}")
    print(f"Using {sentinel_path} as reference for geolocation")
    
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
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
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
    
    # Read NoData mask from Sentinel raster
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
    print(f"Theoretical proportion map created at {output_path}")
    return output_path


def process_prediction(model_paths, target_classes, bands_dir, indices_dir, output_path, 
                      reference_path=None, mask_path=None, normalise=False):
    """
    Create a multi-band TIFF with regression predictions for each target class.
    
    Args:
        model_paths: Dictionary of paths to models for each target class
                    (e.g., {'sqrt_Pure_Lichen': 'path/to/lichen_model.joblib', ...})
        target_classes: Dictionary mapping model names to output band names
                        (e.g., {'sqrt_Pure_Lichen': 'Pure_Lichen'})
        bands_dir: Directory containing Sentinel band TIFs
        indices_dir: Directory containing Sentinel index TIFs
        output_path: Path to save the output TIFF
        reference_path: Optional path to a reference TIF for dimensions
        mask_path: Path to a mask file (optional). Only pixels where mask is 1 will be processed
        normalise: Whether to normalize predictions so their sum equals 100%
        
    Returns:
        Path to the created prediction map
    """
    print(f"\n{'='*80}\nProcessing regression prediction\n{'='*80}")
    
    # Check which models exist and create directory for output
    available_models = {}
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            available_models[model_name] = path
        else:
            print(f"Warning: Model not found: {path}")
    
    if not available_models:
        raise ValueError("No valid regression models found")
    
    # Filter target_classes to only include available models
    available_target_classes = {model: target_classes[model] for model in available_models.keys()}
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    # Load all models
    models = {}
    all_feature_names = None
    
    for target_class, model_path in available_models.items():
        model, feature_names = load_regression_model(model_path)
        models[target_class] = model
        if all_feature_names is None:
            all_feature_names = feature_names
    
    print(f"Creating regression TIFF with {len(available_target_classes)} target classes: {list(available_target_classes.values())}")
    
    # Load Sentinel features - use reference_path if provided
    features, band_names, geo_transform, projection = load_sentinel_data(
        bands_dir, indices_dir, reference_path)
    
    height, width = features[0].shape
    
    # Load mask if provided
    mask = None
    if mask_path:
        mask = load_mask(mask_path, ref_transform=geo_transform, 
                        ref_projection=projection, ref_width=width, ref_height=height)
    
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    num_bands = len(available_target_classes)
    out_ds = driver.Create(output_path, width, height, num_bands, gdal.GDT_Int16,
                          options=['COMPRESS=DEFLATE', 'TILED=YES'])
    
    if out_ds is None:
        raise ValueError(f"Could not create output file {output_path}")
    
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Initialize prediction arrays - one for each output class
    prediction_arrays = {output_class: np.full((height, width), -1, dtype=np.float32)  # Initialize with NoData (-1)
                        for model_name, output_class in available_target_classes.items()}
    
    # Make predictions
    print("Making predictions...")
    for row in tqdm(range(height)):
        row_features = []
        valid_cols = []
        for col in range(width):
            # Only process pixels within the mask (if provided)
            if mask is not None:
                if mask[row, col] != 1:
                    continue  # Skip this pixel
            pixel_features = [band[row, col] for band in features]
            row_features.append(pixel_features)
            valid_cols.append(col)
        if not row_features:
            continue
        X = np.array(row_features)
        # Predict for each target class
        for model_name, output_class in available_target_classes.items():
            model = models[model_name]
            predictions = model.predict(X)
            if model_name.startswith("sqrt_"):
                predictions = predictions ** 2
            for i, col in enumerate(valid_cols):
                prediction_arrays[output_class][row, col] = predictions[i]
    
    if normalise:
        # Normalize predictions to sum to 100%
        print("Normalizing predictions...")
        sum_array = np.zeros((height, width), dtype=np.float32)
        for output_class in available_target_classes.values():
            # Only normalize valid pixels (non-negative values)
            valid_mask = prediction_arrays[output_class] >= 0
            prediction_arrays[output_class][valid_mask] = np.maximum(0, prediction_arrays[output_class][valid_mask])
            sum_array[valid_mask] += prediction_arrays[output_class][valid_mask]
        
        # Avoid division by zero
        valid_sum_mask = sum_array > 0
        
        # Scale to 0-100 range as integers
        for i, (model_name, output_class) in enumerate(available_target_classes.items()):
            # Only normalize pixels that are valid and have positive sum
            valid_pixels = (prediction_arrays[output_class] >= 0) & valid_sum_mask
            prediction_arrays[output_class][valid_pixels] = np.divide(
                prediction_arrays[output_class][valid_pixels], 
                sum_array[valid_pixels]
            ) * 100
            
            # Convert to int16
            normalized_int = np.full((height, width), -1, dtype=np.int16)  # Initialize with NoData
            normalized_int[valid_pixels] = np.clip(prediction_arrays[output_class][valid_pixels], 0, 100).astype(np.int16)
            
            # Write to band
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(normalized_int)
            band.SetDescription(output_class)
            band.SetNoDataValue(-1)
            band.FlushCache()
    else:
        # No normalization: convert proportions to 0-100 percentage range
        print("Writing predictions without normalization...")
        for i, (model_name, output_class) in enumerate(available_target_classes.items()):
            # Create output array with NoData values
            output_array = np.full((height, width), -1, dtype=np.int16)
            
            # Process only valid pixels (non-negative)
            valid_pixels = prediction_arrays[output_class] >= 0
            
            # Convert to percentage (0-100)
            output_array[valid_pixels] = np.clip(
                prediction_arrays[output_class][valid_pixels] * 100,  # Convert to percentage
                0, 100
            ).astype(np.int16)
            
            # Write to band
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(output_array)
            band.SetDescription(output_class)
            band.SetNoDataValue(-1)
            band.FlushCache()
    
    # Add band descriptions
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": name for i, name in enumerate(available_target_classes.values())})
    out_ds = None
    
    print(f"Regression prediction map created at {output_path}")
    return output_path


def compare_results(truth_path, pred_path, output_path, band_mapping=None, mask_path=None):
    """
    Compare theoretical proportions with predicted proportions and generate evaluation plots.
    
    Args:
        truth_path: Path to the theoretical proportion TIFF
        pred_path: Path to the predicted proportion TIFF
        output_path: Path to save the output plots and metrics
        band_mapping: Dictionary mapping prediction band names to truth band names
        mask_path: Path to a mask file (optional). Only pixels where mask is 1 will be evaluated
        
    Returns:
        Dictionary of dictionaries with metrics and arrays for each band
    """
    print(f"\n{'='*80}\nEvaluating regression results\n{'='*80}")
    
    # Load TIFFs
    print(f"Loading ground truth TIF: {truth_path}")
    truth_bands, truth_band_names, truth_width, truth_height, geo_transform, projection = load_tif_bands(truth_path)
    print(f"Found bands: {truth_band_names}")
    
    print(f"Loading prediction TIF: {pred_path}")
    pred_bands, pred_band_names, pred_width, pred_height, _, _ = load_tif_bands(pred_path)
    print(f"Found bands: {pred_band_names}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Check dimensions match
    if truth_width != pred_width or truth_height != pred_height:
        print(f"Warning: Dimension mismatch! Truth: {truth_width}x{truth_height}, Prediction: {pred_width}x{pred_height}")
        print("Will use only common valid pixels")
    
    # Load mask if provided
    mask = None
    if mask_path:
        mask = load_mask(mask_path, ref_transform=geo_transform, ref_projection=projection,
                        ref_width=truth_width, ref_height=truth_height)
    
    # If no band mapping provided, try to match names
    if not band_mapping:
        band_mapping = {}
        for band_name in truth_band_names:
            if band_name in pred_band_names:
                band_mapping[band_name] = band_name
    
    # Get bands to compare
    selected_bands = list(band_mapping.keys())
    print(f"Will compare these bands: {selected_bands}")
    
    # Initialize results dictionary
    results = {}
    
    # Compare each band
    for truth_band_name in selected_bands:
        # Get corresponding band name in prediction
        pred_band_name = band_mapping[truth_band_name]
        
        # Check both bands exist
        if truth_band_name not in truth_bands or pred_band_name not in pred_bands:
            print(f"Warning: Band {truth_band_name} or {pred_band_name} not found in TIFs.")
            continue
        
        # Get band data
        truth_data = truth_bands[truth_band_name]
        pred_data = pred_bands[pred_band_name]
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape == truth_data.shape:
                truth_data = np.where(mask == 1, truth_data, np.nan)
                pred_data = np.where(mask == 1, pred_data, np.nan)
            else:
                print(f"Warning: Mask shape {mask.shape} doesn't match data shape {truth_data.shape}. Skipping mask.")
        
        # Flatten arrays and filter out NaN values
        valid_mask = ~np.isnan(truth_data) & ~np.isnan(pred_data)
        truth_flat = truth_data[valid_mask].flatten()
        pred_flat = pred_data[valid_mask].flatten()
        
        # Skip if no valid pixels
        if len(truth_flat) == 0:
            print(f"Warning: No valid pixels for band {truth_band_name}")
            continue
        
        # Calculate metrics
        r2 = r2_score(truth_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(truth_flat, pred_flat))
        pearson_coef, _ = pearsonr(truth_flat, pred_flat)
        
        results[truth_band_name] = {
            'truth': truth_flat,
            'pred': pred_flat,
            'metrics': {
                'r2': r2,
                'rmse': rmse,
                'pearson': pearson_coef
            }
        }
    
    # Create plots
    plot_evaluation_results(results, output_path)
    
    return results


def load_tif_bands(tif_path):
    """
    Load all bands from a multi-band TIF file
    
    Args:
        tif_path: Path to the TIF file
        
    Returns:
        bands_data: Dictionary with band names as keys and band data as values
        band_names: List of band names in order
        width: Width of the raster
        height: Height of the raster
        geo_transform: GeoTransform of the raster
        projection: Projection of the raster
    """
    ds = gdal.Open(tif_path)
    if ds is None:
        raise ValueError(f"Could not open TIF file: {tif_path}")
    
    width = ds.RasterXSize
    height = ds.RasterYSize
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # Get band descriptions from metadata if available
    band_names = []
    bands_data = {}
    
    # Get metadata
    metadata = ds.GetMetadata()
    
    # Try to get band names from metadata
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        band_desc = band.GetDescription()
        
        # If band has a description, use that; otherwise try metadata or use default
        if band_desc:
            band_name = band_desc
        elif f'BAND_{i}_NAME' in metadata:
            band_name = metadata[f'BAND_{i}_NAME']
        else:
            band_name = f"band_{i}"
        
        # Read band data
        band_data = band.ReadAsArray().astype(np.float32)
        
        # Convert from percentage (0-100) to proportion (0-1)
        band_data = band_data / 100.0
        
        # Replace no data values with NaN
        nodata = band.GetNoDataValue()
        if nodata is not None:
            band_data[band_data == nodata/100.0] = np.nan
        
        band_names.append(band_name)
        bands_data[band_name] = band_data
    
    return bands_data, band_names, width, height, geo_transform, projection


def plot_evaluation_results(results, output_path):
    """
    Plot scatter plots of truth vs prediction for all bands
    
    Args:
        results: Dictionary with comparison results
        output_path: Path to save the output plot
    """
    # Number of bands to plot
    n_bands = len(results)
    
    if n_bands == 0:
        print("No valid bands to plot")
        return
        
    # Calculate grid dimensions for subplots
    cols = min(3, n_bands)  # Maximum 3 columns
    rows = (n_bands + cols - 1) // cols  # Calculate rows needed
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    
    # If there's only one band, axes won't be an array
    if n_bands == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy indexing
    axes = np.array(axes).flatten()
    
    # Plot each band on the combined figure
    for i, (band_name, band_data) in enumerate(results.items()):
        ax = axes[i]
        truth = band_data['truth']
        pred = band_data['pred']
        metrics = band_data['metrics']
        
        # Create scatter plot
        ax.scatter(truth, pred, color='#1F77B4', alpha=0.6, s=10, edgecolor='none')
        
        # Get max value for plot limits
        max_val = max(np.max(truth), np.max(pred))
        
        # Plot identity line (y=x)
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        
        # Set labels and title
        ax.set_xlabel("Actual value", fontsize=10)
        ax.set_ylabel("Predicted value", fontsize=10)
        
        # Add metrics to title
        ax.set_title(f"{band_name}\n"
                    f"R² = {metrics['r2']:.3f}, "
                    f"RMSE = {metrics['rmse']:.3f}, "
                    f"r = {metrics['pearson']:.3f}", 
                    fontsize=12)
        
        # Set axis limits
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        
        # Add grid
        ax.grid(alpha=0.3)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150)
    print(f"Evaluation plot saved to {output_path}")
    plt.close()
    
    # Display metrics summary
    print("\nPerformance metrics summary:")
    print("=" * 60)
    print(f"{'Band':<25} {'R²':>10} {'RMSE':>10} {'Pearson r':>10}")
    print("-" * 60)
    for band_name, band_data in results.items():
        metrics = band_data['metrics']
        print(f"{band_name:<25} {metrics['r2']:>10.4f} {metrics['rmse']:>10.4f} {metrics['pearson']:>10.4f}")
    print("=" * 60)
    
    # Save metrics to CSV
    metrics_path = output_path.replace('.png', '_metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write("band,r2,rmse,pearson\n")
        for band_name, band_data in results.items():
            metrics = band_data['metrics']
            f.write(f"{band_name},{metrics['r2']:.6f},{metrics['rmse']:.6f},{metrics['pearson']:.6f}\n")
    print(f"Metrics saved to {metrics_path}")
    
    # Create individual plots for each band
    for band_name, band_data in results.items():
        truth = band_data['truth']
        pred = band_data['pred']
        metrics = band_data['metrics']

        plt.figure(figsize=(6, 6))
        plt.scatter(truth, pred, color='#1F77B4', alpha=0.6, s=10, edgecolor='none')
        max_val = max(np.max(truth), np.max(pred))
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        plt.xlabel("Actual value", fontsize=12)
        plt.ylabel("Predicted value", fontsize=12)
        plt.title(f"{band_name}\n"
                  f"R² = {metrics['r2']:.3f}, "
                  f"RMSE = {metrics['rmse']:.3f}, "
                  f"r = {metrics['pearson']:.3f}", 
                  fontsize=14)
        plt.xlim(0, max_val * 1.05)
        plt.ylim(0, max_val * 1.05)
        plt.grid(alpha=0.3)
        indiv_path = output_path.replace('.png', f'_{band_name}.png')
        plt.tight_layout()
        plt.savefig(indiv_path, dpi=150)
        plt.close()
        print(f"Individual plot for {band_name} saved to {indiv_path}")


def main():
    """
    Example of using the three main functions
    """
    # wap = 23
    resolution = "_10m"
    # # Example configuration
    # base_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{resolution}/regression"
    # os.makedirs(base_dir, exist_ok=True)
    
    
    # # Input paths
    # classification_path = f"drone_treated/WAP{wap}_tiles/WAP{wap}_classif_merged.tif"
    # sentinel_path = f"DataCubeS2/WAP{wap}{resolution}/mediane_bands/mediane_clipped_STACK_2023_BandB4_WAP{wap}_deflate.tif"
    # bands_dir = f"DataCubeS2/WAP{wap}{resolution}/mediane_bands"
    # indices_dir = f"DataCubeS2/WAP{wap}{resolution}/mediane_indices"

    # if wap == 32 : 
    #     mask_path = "drone_treated/WAP32_tiles/mask_WAP32_2.tif"
    # elif wap == 23 : 
    #     mask_path = f"drone_treated/WAP23_tiles/mask_well_classified_2_WAP23.tif"
    # else:
    #     mask_path = None  # No mask provided

    base_dir = f"data/regressions/regression_merged_model/regression_chesnay{resolution}/regression"
    classification_path = "Konstantin/Chesnay_tiles/merged_classification_8.tif"
    sentinel_path = f"DataCubeS2/Chesnay{resolution}/mediane_bands/mediane_STACK_2023_BandB2_Chesnay_deflate.tif"
    bands_dir = f"DataCubeS2/Chesnay{resolution}/mediane_bands"
    indices_dir = f"DataCubeS2/Chesnay{resolution}/mediane_indices"
    mask_path = "Konstantin/Chesnay_tiles/mask_well_classified.tif"
    
    # Model paths (from a model directory)
    model_dir = f"data/regressions/regression_merged_model/regression_wap32{resolution}/regression_results"
    model_paths = {
        'sqrt_Pure_Lichen': os.path.join(model_dir, 'sqrt_Pure_Lichen_rf.joblib'),
        'sqrt_Degraded_Lichen': os.path.join(model_dir, 'sqrt_Degraded_Lichen_rf.joblib'),
        'sqrt_Green': os.path.join(model_dir, 'sqrt_Green_rf.joblib'),
        'sqrt_all_lichen': os.path.join(model_dir, 'sqrt_all_lichen_rf.joblib'),
        'sqrt_through_proportion': os.path.join(model_dir, 'sqrt_through_proportion_rf.joblib'),
    }
    
    # Target class mapping
    target_classes = {
        'sqrt_Pure_Lichen': 'Pure_Lichen',
        'sqrt_Degraded_Lichen': 'Degraded_Lichen',
        'sqrt_Green': 'Green',
        'sqrt_all_lichen': 'all_lichen',
        'sqrt_through_proportion': 'through_proportion',
    }
    
    # Output paths
    # proportion_path = f"{base_dir}/theoretical_proportions_WAP{wap}.tif"
    # prediction_path = f"{base_dir}/regression_predictions_WAP{wap}.tif"
    proportion_path = f"{base_dir}/theoretical_proportions_chesnay.tif"
    prediction_path = f"{base_dir}/regression_predictions_chesnay.tif"
    evaluation_path = f"{base_dir}/evaluation/evaluation.png"
    
    # 1. Process proportions from classification
    process_proportions(
        classification_path=classification_path,
        sentinel_path=sentinel_path,
        output_path=proportion_path,
        mask_path=mask_path
    )
    
    # 2. Make predictions using models and Sentinel data
    process_prediction(
        model_paths=model_paths,
        target_classes=target_classes,
        bands_dir=bands_dir,
        indices_dir=indices_dir,
        output_path=prediction_path,
        reference_path=proportion_path,  # Use proportions as reference for dimensions
        mask_path=mask_path,
        normalise=False
    )
    
    # 3. Compare results and create evaluation plots
    compare_results(
        truth_path=proportion_path,
        pred_path=prediction_path,
        output_path=evaluation_path,
        mask_path=mask_path
    )
    
    print("Workflow completed successfully!")


if __name__ == "__main__":
    main()
