"""
Process regression prediction and evaluation.

This script combines functionalities from plot_regression_tif_2.py and evaluate_regression_2.py
to create a streamlined workflow for:
1. Generating regression maps from Sentinel data using trained models
2. Evaluating regression results by comparing predicted vs theoretical proportions
3. Ensuring consistent dimensions between maps for proper comparison

All operations use the same reference geometry to avoid dimension mismatches.
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

def load_regression_model(model_path):
    """
    Load a regression model from a joblib file
    
    Args:
        model_path: Path to the joblib file containing the model
        
    Returns:
        Loaded regression model and its feature names if available
    """
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

def load_sentinel_features(bands_dir, indices_dir, reference_path=None):
    """
    Load all Sentinel bands and indices as features
    
    Args:
        bands_dir: Directory containing band TIF files
        indices_dir: Directory containing indices TIF files
        reference_path: Optional path to a reference raster to match dimensions
        
    Returns:
        features: NumPy array of shape (n_features, rows, cols)
        band_names: List of feature names
        geo_transform: GeoTransform of the input raster
        projection: Projection of the input raster
    """
    # List all tif files
    band_files = sorted([os.path.join(bands_dir, f) for f in os.listdir(bands_dir) 
                        if f.lower().endswith('.tif') and not f.endswith('.aux.xml')])
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
        band_name = file_name.replace('mediane_clipped_STACK_2023_Band', '').replace('_WAP32', '').replace('_WAP23', '')
        band_names.append(f"band_{band_name}")
    
    # Load index files
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
        index_name = file_name.replace('mediane_clipped_', '').replace('_WAP32', '').replace('_WAP23', '')
        band_names.append(f"index_{index_name}")
    
    features = np.stack(bands, axis=0)  # (n_features, rows, cols)
    print(f"Loaded {len(band_names)} features with dimensions {features.shape}")
    return features, band_names, geo_transform, projection

def load_mask(mask_path, ref_transform, ref_projection, ref_width, ref_height):
    """
    Load a mask file and reproject it to match the reference dataset if needed
    
    Args:
        mask_path: Path to the prediction map (first band will be used for masking)
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
    
    # Get the first band and its noData value
    first_band = mask_ds.GetRasterBand(1)
    nodata_value = first_band.GetNoDataValue()
    
    print(f"Using noData value: {nodata_value}")
    
    # Check if mask needs reprojection
    mask_transform = mask_ds.GetGeoTransform()
    mask_projection = mask_ds.GetProjection()
    mask_width = mask_ds.RasterXSize
    mask_height = mask_ds.RasterYSize
    
    # If mask dimensions and projection match reference, read directly
    if (mask_width == ref_width and mask_height == ref_height and 
        mask_transform == ref_transform and mask_projection == ref_projection):
        mask_array = first_band.ReadAsArray()
    else:
        # Reproject mask to match reference dataset
        print(f"Reprojecting mask from {mask_width}x{mask_height} to {ref_width}x{ref_height}...")
        mem_driver = gdal.GetDriverByName('MEM')
        mask_reprojected = mem_driver.Create('', ref_width, ref_height, 1, first_band.DataType)
        mask_reprojected.SetGeoTransform(ref_transform)
        mask_reprojected.SetProjection(ref_projection)
        
        # Reproject
        gdal.ReprojectImage(mask_ds, mask_reprojected, mask_projection, ref_projection, 
                           gdal.GRA_NearestNeighbour)
        
        # Read reprojected mask
        mask_array = mask_reprojected.GetRasterBand(1).ReadAsArray()
    
    # Create binary mask (0 where noData, 1 elsewhere)
    if nodata_value is not None:
        binary_mask = (mask_array != nodata_value).astype(np.uint8)
    else:
        # If noData value is not defined, assume all pixels are valid
        binary_mask = np.ones_like(mask_array, dtype=np.uint8)
    
    print(f"Mask created. Valid pixels: {np.sum(binary_mask)}/{binary_mask.size} ({np.sum(binary_mask)/binary_mask.size*100:.2f}%)")
    
    return binary_mask

def process_regression(model_paths, target_classes, bands_dir, indices_dir, output_path, 
                      reference_path=None, mask_path=None, normalise=False):
    """
    Create a multi-band TIFF with regression predictions for each target class.
    Uses the same reference for all operations to ensure dimensional consistency.
    
    Args:
        model_paths: Dictionary of paths to models for each target class
                     (e.g., {'Pure_Lichen': 'path/to/lichen_model.joblib', ...})
        target_classes: Dictionary mapping model names to output band names
                        (e.g., {'sqrt_Pure_Lichen': 'Pure_Lichen'})
        bands_dir: Directory containing Sentinel band TIFs
        indices_dir: Directory containing Sentinel index TIFs
        output_path: Path to save the output TIFF
        reference_path: Optional path to a reference TIF for dimensions and projection
        mask_path: Path to a mask file (optional). Only pixels where mask is not 0 will be processed
        normalise: Whether to normalise predictions so that their sum is 100% for each pixel
    """
    # Load all models
    models = {}
    all_feature_names = None
    
    for target_class, model_path in model_paths.items():
        model, feature_names = load_regression_model(model_path)
        models[target_class] = model
        if all_feature_names is None:
            all_feature_names = feature_names
    
    print(f"Creating regression TIFF with {len(target_classes)} target classes: {list(target_classes.values())}")
    
    # Load Sentinel features - use reference_path if provided
    features, band_names, geo_transform, projection = load_sentinel_features(
        bands_dir, indices_dir, reference_path)
    
    height, width = features[0].shape
    
    # Load mask if provided
    mask = None
    if mask_path:
        mask = load_mask(mask_path, geo_transform, projection, width, height)
    
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    num_bands = len(target_classes)
    out_ds = driver.Create(output_path, width, height, num_bands, gdal.GDT_Int16,
                          options=['COMPRESS=DEFLATE', 'TILED=YES'])
    
    if out_ds is None:
        raise ValueError(f"Could not create output file {output_path}")
    
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Initialize prediction arrays - one for each output class
    prediction_arrays = {output_class: np.full((height, width), -1, dtype=np.float32)  # Initialize with NoData (-1)
                        for model_name, output_class in target_classes.items()}
    
    # Make predictions
    print("Making predictions...")
    for row in tqdm(range(height)):
        # Extract features for the entire row
        row_features = []
        row_mask = []
        
        for col in range(width):
            # If mask is provided, skip pixels outside the mask
            if mask is not None and mask[row, col] == 0:
                row_features.append(None)  # Placeholder
                row_mask.append(False)
                continue
            
            # Extract features for this pixel
            pixel_features = [band[row, col] for band in features]
            row_features.append(pixel_features)
            row_mask.append(True)
        
        # Skip if no valid pixels in this row
        if not any(row_mask):
            continue
        
        # Collect only valid features
        valid_features = [feat for feat, valid in zip(row_features, row_mask) if valid]
        
        # Skip if no valid features
        if not valid_features:
            continue
            
        # Convert to numpy array
        X = np.array(valid_features)
        
        # Get valid column indices
        valid_cols = [col for col, valid in enumerate(row_mask) if valid]
        
        # Predict for each target class
        for model_name, output_class in target_classes.items():
            model = models[model_name]
            
            # Make prediction
            predictions = model.predict(X)
            
            # Apply squared transformation for sqrt_ classes
            if model_name.startswith("sqrt_"):
                predictions = predictions ** 2  # Convert back from sqrt space
            
            # Assign predictions back to the correct columns
            for i, col in enumerate(valid_cols):
                prediction_arrays[output_class][row, col] = predictions[i]
    
    if normalise:
        # Normalize predictions to sum to 1 (100%)
        print("Normalizing predictions...")
        sum_array = np.zeros((height, width), dtype=np.float32)
        for output_class in target_classes.values():
            # Only normalize valid pixels (non-negative values)
            valid_mask = prediction_arrays[output_class] >= 0
            prediction_arrays[output_class][valid_mask] = np.maximum(0, prediction_arrays[output_class][valid_mask])  # Clip negative values
            sum_array[valid_mask] += prediction_arrays[output_class][valid_mask]
        
        # Avoid division by zero
        valid_sum_mask = sum_array > 0
        
        # Scale to 0-100% range as integers
        for i, (model_name, output_class) in enumerate(target_classes.items()):
            # Only normalize pixels that are in mask and have positive sum
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
        for i, (model_name, output_class) in enumerate(target_classes.items()):
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
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": name for i, name in enumerate(target_classes.values())})
    out_ds = None
    
    print(f"Multi-band regression TIFF created at {output_path}")
    return output_path

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
        
        # If band has a description, use that; otherwise try metadata or use a default name
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

def evaluate_regression(truth_path, pred_path, output_path, band_mapping=None, mask_path=None):
    """
    Evaluate regression results by comparing theoretical proportions with predicted proportions
    
    Args:
        truth_path: Path to the theoretical proportion TIFF
        pred_path: Path to the predicted proportion TIFF
        output_path: Path to save the output plots and metrics
        band_mapping: Dictionary mapping prediction band names to truth band names
        mask_path: Path to a mask file (optional). Only pixels where mask is not 0 will be evaluated
    
    Returns:
        results: Dictionary of dictionaries with metrics and arrays for each band
    """
    # Load TIFFs
    print(f"Loading ground truth TIF: {truth_path}")
    truth_bands, truth_band_names, truth_width, truth_height, truth_transform, truth_projection = load_tif_bands(truth_path)
    print(f"Found bands: {truth_band_names}")
    
    print(f"\nLoading prediction TIF: {pred_path}")
    pred_bands, pred_band_names, pred_width, pred_height, pred_transform, pred_projection = load_tif_bands(pred_path)
    print(f"Found bands: {pred_band_names}")
    
    # Check dimensions match
    if truth_width != pred_width or truth_height != pred_height:
        print(f"Warning: Dimension mismatch! Truth: {truth_width}x{truth_height}, Prediction: {pred_width}x{pred_height}")
        print("Will try to use common valid pixels, but results may be unreliable.")
    
    # Load mask if provided
    mask = None
    if mask_path:
        mask = load_mask(mask_path, truth_transform, truth_projection, truth_width, truth_height)
    
    # If no band mapping provided, try to use direct band name matching
    if not band_mapping:
        # Default: Try direct matching first
        band_mapping = {}
        for band_name in truth_band_names:
            if band_name in pred_band_names:
                band_mapping[band_name] = band_name
            elif band_name.startswith('sqrt_') and band_name[5:] in pred_band_names:
                band_mapping[band_name] = band_name[5:]
            elif 'sqrt_' + band_name in pred_band_names:
                band_mapping[band_name] = 'sqrt_' + band_name
    
    # Get bands to compare based on what we have in both TIFs
    selected_bands = list(band_mapping.keys())
    print(f"Will compare these bands: {selected_bands}")
    
    # Initialize results dictionary
    results = {}
    
    # Compare each band
    for truth_band_name in selected_bands:
        # Get the corresponding band name in prediction
        pred_band_name = band_mapping[truth_band_name]
        
        # Check if both bands exist
        if truth_band_name not in truth_bands or pred_band_name not in pred_bands:
            print(f"Warning: Band {truth_band_name} or {pred_band_name} not found in TIFs.")
            continue
        
        # Get band data
        truth_data = truth_bands[truth_band_name]
        pred_data = pred_bands[pred_band_name]
        
        # Apply mask if provided
        if mask is not None:
            # Make sure mask has the same dimensions as truth data
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

def plot_evaluation_results(results, output_path):
    """
    Plot scatter plots of truth vs prediction for all bands in a single figure with subplots
    
    Args:
        results: Dictionary with comparison results
        output_path: Path to save the output plot
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Plot each band
    for i, (band_name, band_data) in enumerate(results.items()):
        ax = axes[i]
        truth = band_data['truth']
        pred = band_data['pred']
        metrics = band_data['metrics']
        
        # Create scatter plot with blue dots
        ax.scatter(truth, pred, color='#1F77B4', alpha=0.6, s=10, edgecolor='none')
        
        # Get max value for plot limits
        max_val = max(np.max(truth), np.max(pred))
        
        # Plot identity line (y=x)
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        
        # Set labels and title
        ax.set_xlabel("Actual value (original scale)", fontsize=10)
        ax.set_ylabel("Predicted value (original scale)", fontsize=10)
        
        # Add metrics to title
        ax.set_title(f"{band_name}\n"
                    f"R² = {metrics['r2']:.3f}, "
                    f"RMSE = {metrics['rmse']:.3f}, "
                    f"r = {metrics['pearson']:.3f}", 
                    fontsize=12)
        
        # Set axis limits
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        
        # Add grid with light lines
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
    
    # Display metrics in a more readable format
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

def run_regression(wap, bands_dir, indices_dir, model_paths, target_classes, 
                  output_path, reference_path=None, mask_path=None, normalise=False):
    """
    Run the regression process to create a prediction map
    
    Args:
        wap: WAP site number
        bands_dir: Directory containing Sentinel band TIFs
        indices_dir: Directory containing Sentinel index TIFs
        model_paths: Dictionary of paths to models for each target class
        target_classes: Dictionary mapping model names to output band names
        output_path: Path to save the output TIFF
        reference_path: Optional path to a reference TIF for dimensions and projection
        mask_path: Path to mask file (if None, uses default path)
        normalise: Whether to normalize predictions
        
    Returns:
        Path to the created regression map
    """
    # Check which models exist
    available_models = {}
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            available_models[model_name] = path
        else:
            print(f"Warning: Model not found: {path}")
    
    if not available_models:
        raise ValueError(f"No regression models found")
    
    # Filter target_classes to only include available models
    available_target_classes = {model: target_classes[model] for model in available_models.keys()}
    
    # Check if reference file exists
    if reference_path and os.path.exists(reference_path):
        print(f"Using reference file: {reference_path}")
    else:
        reference_path = None
        print("No valid reference file specified, will use first band as reference")
    
    # Run regression process
    print(f"\n{'='*80}\nGenerating regression map\n{'='*80}")
    process_regression(
        model_paths=available_models,
        target_classes=available_target_classes,
        bands_dir=bands_dir,
        indices_dir=indices_dir,
        output_path=output_path,
        reference_path=reference_path,
        mask_path=mask_path,
        normalise=normalise
    )
    
    return output_path

def run_evaluation(truth_path, pred_path, output_path, band_mapping=None, mask_path=None):
    """
    Run the evaluation process to compare predicted vs theoretical proportions
    
    Args:
        truth_path: Path to theoretical proportion map
        pred_path: Path to regression prediction map
        output_path: Path for evaluation output
        band_mapping: Dictionary mapping prediction band names to truth band names
        mask_path: Path to mask file (optional)
        
    Returns:
        Path to the evaluation results plot
    """
    if not os.path.exists(truth_path):
        raise ValueError(f"Theoretical proportion map not found at {truth_path}")
    
    if not os.path.exists(pred_path):
        raise ValueError(f"Regression prediction map not found at {pred_path}")
    
    # Run evaluation
    print(f"\n{'='*80}\nEvaluating regression results\n{'='*80}")
    evaluate_regression(
        truth_path=truth_path,
        pred_path=pred_path,
        output_path=output_path,
        band_mapping=band_mapping,
        mask_path=mask_path
    )
    
    return output_path

def main():
    """
    Main function with all path configurations for easy modification
    """
    # ===== CONFIGURATION PARAMETERS =====
    # Basic parameters
    wap = 23  # WAP site number to process
    do_regression = True  # Whether to run regression
    do_evaluation = True  # Whether to run evaluation
    
    # Data configuration
    use_peat = False  # Whether to use peat data
    superresolution = False  # Use 5m (True) or 10m (False) resolution
    moy5m = False  # Whether to use moy5m data
    normalise = False  # Whether to normalize predictions
    output_suffix = ""  # Custom suffix to add to output filenames
    
    # Derived path components
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m_suffix = "_moy5m" if moy5m else ""
    file_prefix = "" if not moy5m else "10m_"
    norm_suffix = "_normalised" if normalise else ""
    
    # ===== PATH DEFINITIONS =====
    # Base directories
    base_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Model source directory (using WAP32 models by default)
    regression_dir = f"data/regressions/regression_merged_model/regression_wap32{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_results"
    
    # Input/output paths
    reference_path = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_bands/{file_prefix}mediane_clipped_STACK_2023_BandB4_WAP{wap}_deflate.tif"
    theoretical_map_path = f"{base_dir}/proportions_WAP{wap}.tif"
    mask_path = f"drone_treated/WAP{wap}_tiles/mask_WAP{wap}_peat.tif"
    regression_map_path = f"{base_dir}/regression_predictions_WAP{wap}{output_suffix}{norm_suffix}.tif"
    
    # Create evaluation directory to prevent empty path issue
    evaluation_dir = os.path.join(base_dir, "regression_evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    evaluation_path = os.path.join(evaluation_dir, f"evaluation{output_suffix}.png")
    
    # Check if files exist
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file not found at {mask_path}, proceeding without mask")
        mask_path = None
    
    # Sentinel data directories
    bands_dir = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_bands/"
    indices_dir = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_indices/"
    
    # Model paths
    model_paths = {
        'sqrt_Pure_Lichen': os.path.join(regression_dir, 'sqrt_Pure_Lichen_rf.joblib'),
        'sqrt_Degraded_Lichen': os.path.join(regression_dir, 'sqrt_Degraded_Lichen_rf.joblib'),
        'sqrt_Green': os.path.join(regression_dir, 'sqrt_Green_rf.joblib'),
        'sqrt_all_lichen': os.path.join(regression_dir, 'sqrt_all_lichen_rf.joblib'),
        'sqrt_through_proportion': os.path.join(regression_dir, 'sqrt_through_proportion_rf.joblib'),
    }
    
    # Target class mapping
    target_classes = {
        'sqrt_Pure_Lichen': 'Pure_Lichen',
        'sqrt_Degraded_Lichen': 'Degraded_Lichen',
        'sqrt_Green': 'Green',
        'sqrt_all_lichen': 'all_lichen',
        'sqrt_through_proportion': 'through_proportion',
    }
    
    # Band mapping for evaluation
    band_mapping = {
        "Pure_Lichen": "Pure_Lichen",
        "Degraded_Lichen": "Degraded_Lichen",
        "Green": "Green",
        "all_lichen": "all_lichen",
        "through_proportion": "through_proportion"
    }
    
    # ===== EXECUTE WORKFLOW =====
    # Run regression if requested
    if do_regression:
        regression_map_path = run_regression(
            wap=wap,
            bands_dir=bands_dir,
            indices_dir=indices_dir,
            model_paths=model_paths,
            target_classes=target_classes,
            output_path=regression_map_path,
            reference_path=theoretical_map_path if os.path.exists(theoretical_map_path) else reference_path,
            mask_path=mask_path,
            normalise=normalise
        )
        print(f"Regression completed. Output saved to: {regression_map_path}")
    
    # Run evaluation if requested
    if do_evaluation:
        try:
            if not os.path.exists(theoretical_map_path):
                print(f"Warning: Theoretical proportion map not found at {theoretical_map_path}")
                print("Evaluation will be skipped.")
            else:
                evaluation_path = run_evaluation(
                    truth_path=theoretical_map_path,
                    pred_path=regression_map_path,
                    output_path=evaluation_path,
                    band_mapping=band_mapping,
                    mask_path=mask_path
                )
                print(f"Evaluation completed successfully. Results saved to {evaluation_path}")
        except ValueError as e:
            print(f"Evaluation could not be performed: {e}")

if __name__ == "__main__":
    main()
