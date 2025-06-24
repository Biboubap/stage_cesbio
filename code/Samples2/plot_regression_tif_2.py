"""
Create a multi-band TIFF file containing the predicted proportions from regression models.
Each band represents a different class proportion, scaled from 0-100 (percentage) 
and optionally normalized to ensure they sum to 100%.
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal
import joblib
from tqdm import tqdm

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

def load_sentinel_features(bands_dir, indices_dir):
    """
    Load all Sentinel bands and indices as features
    
    Args:
        bands_dir: Directory containing band TIF files
        indices_dir: Directory containing indices TIF files
        
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
    
    # Get reference dimensions and geotransform
    ref_ds = gdal.Open(band_files[0])
    if ref_ds is None:
        raise ValueError(f"Could not open reference raster: {band_files[0]}")
    
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
        arr = ds.GetRasterBand(1).ReadAsArray()
        bands.append(arr)
        file_name = os.path.splitext(os.path.basename(tif))[0]
        band_name = file_name.replace('mediane_clipped_STACK_2023_Band', '').replace('_WAP32', '')
        band_names.append(f"band_{band_name}")
    
    # Load index files
    for tif in index_files:
        ds = gdal.Open(tif)
        if ds is None:
            print(f"Warning: Could not open {tif}")
            continue
        arr = ds.GetRasterBand(1).ReadAsArray()
        bands.append(arr)
        file_name = os.path.splitext(os.path.basename(tif))[0]
        index_name = file_name.replace('mediane_clipped_', '').replace('_WAP32', '')
        band_names.append(f"index_{index_name}")
    
    features = np.stack(bands, axis=0)  # (n_features, rows, cols)
    return features, band_names, geo_transform, projection

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

def create_regression_tiff(model_paths, bands_dir, indices_dir, output_path, mask_path=None, normalise=True):
    """
    Create a multi-band TIFF with regression predictions for each target class
    
    Args:
        model_paths: Dictionary of paths to models for each target class
                     (e.g., {'sqrt_Pure_Lichen': 'path/to/lichen_model.joblib', ...})
        bands_dir: Directory containing Sentinel band TIFs
        indices_dir: Directory containing Sentinel index TIFs
        output_path: Path to save the output TIFF
        mask_path: Path to a mask file (optional). Only pixels where mask is 1 will be processed
        normalise: Whether to normalise predictions so that their sum is 100% for each pixel
    """
    # Get target classes from model_paths
    target_classes = list(model_paths.keys())
    
    # Load all models
    models = {}
    all_feature_names = None
    
    for target_class, model_path in model_paths.items():
        model, feature_names = load_regression_model(model_path)
        models[target_class] = model
        if all_feature_names is None:
            all_feature_names = feature_names
    
    print(f"Creating regression TIFF with {len(target_classes)} target classes: {target_classes}")
    
    # Load Sentinel features
    features, band_names, geo_transform, projection = load_sentinel_features(bands_dir, indices_dir)
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
    
    # Initialize prediction arrays
    prediction_arrays = {target_class: np.full((height, width), -1, dtype=np.float32)  # Initialize with NoData (-1)
                        for target_class in target_classes}
    
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
        for target_class in target_classes:
            model = models[target_class]
            
            # Make prediction
            predictions = model.predict(X)
            
            # Apply squared transformation for sqrt_ classes
            if target_class.startswith("sqrt_"):
                predictions = predictions ** 2  # Convert back from sqrt space
            
            # Assign predictions back to the correct columns
            for i, col in enumerate(valid_cols):
                prediction_arrays[target_class][row, col] = predictions[i]
    
    if normalise:
        # Normalize predictions to sum to 1 (100%)
        print("Normalizing predictions...")
        sum_array = np.zeros((height, width), dtype=np.float32)
        for target_class in target_classes:
            # Only normalize valid pixels (non-negative values)
            valid_mask = prediction_arrays[target_class] >= 0
            prediction_arrays[target_class][valid_mask] = np.maximum(0, prediction_arrays[target_class][valid_mask])  # Clip negative values
            sum_array[valid_mask] += prediction_arrays[target_class][valid_mask]
        
        # Avoid division by zero
        valid_sum_mask = sum_array > 0
        
        # Scale to 0-100% range as integers
        for i, target_class in enumerate(target_classes):
            # Only normalize pixels that are in mask and have positive sum
            valid_pixels = (prediction_arrays[target_class] >= 0) & valid_sum_mask
            prediction_arrays[target_class][valid_pixels] = np.divide(
                prediction_arrays[target_class][valid_pixels], 
                sum_array[valid_pixels]
            ) * 100
            
            # Convert to int16
            normalized_int = np.full((height, width), -1, dtype=np.int16)  # Initialize with NoData
            normalized_int[valid_pixels] = np.clip(prediction_arrays[target_class][valid_pixels], 0, 100).astype(np.int16)
            
            # Write to band
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(normalized_int)
            band.SetDescription(target_class)
            band.SetNoDataValue(-1)
            band.FlushCache()
    else:
        # No normalization: just clip to 0-100 and write as integer percentages
        print("Writing predictions without normalization...")
        for i, target_class in enumerate(target_classes):
            # Convert to percentages only for valid pixels (non-negative)
            valid_pixels = prediction_arrays[target_class] >= 0
            
            # Create output array with NoData values
            output_array = np.full((height, width), -1, dtype=np.int16)
            
            # Process only valid pixels
            output_array[valid_pixels] = np.clip(
                prediction_arrays[target_class][valid_pixels] * 100, 
                0, 100
            ).astype(np.int16)
            
            # Write to band
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(output_array)
            band.SetDescription(target_class)
            band.SetNoDataValue(-1)
            band.FlushCache()

    # Add band descriptions
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": target_class for i, target_class in enumerate(target_classes)})
    out_ds = None
    
    print(f"Multi-band regression TIFF created at {output_path}")

def main():
    """
    Main function to create regression TIFFs using models saved by sentinel_proportion_median.py
    """
    # Default paths for models and data
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""
    mediane_dir = "mediane" if not moy5m else "mediane_10m"
    file_prefix = "" if not moy5m else "10m_"
    
    # Base directory where regression results are stored
    
    base_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}"
    regression_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_results"

    
    model_paths = {
        'sqrt_Pure_Lichen': os.path.join(regression_dir, 'sqrt_Pure_Lichen_rf.joblib'),
        'sqrt_Degraded_Lichen': os.path.join(regression_dir, 'sqrt_Degraded_Lichen_rf.joblib'),
        'sqrt_Green' : os.path.join(regression_dir, 'sqrt_Green_rf.joblib'),
        'sqrt_all_lichen' : os.path.join(regression_dir, 'sqrt_all_lichen_rf.joblib'),
        'sqrt_through_proportion' : os.path.join(regression_dir, 'sqrt_through_proportion_rf.joblib'),
    }
   
    # Sentinel data directories
    bands_dir = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_bands_10m/"
    indices_dir = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_indices_10m/"
    
    # Mask path
    mask_path = f"drone_treated/WAP{wap}_tiles/mask_WAP{wap}{peat_suffix}.tif"
    
    # Output path
    output_path = os.path.join(base_dir, f"regression_predictions_WAP{wap}{peat_suffix}{resolution_suffix}.tif")
    
    # Check if all model files exist
    missing_models = [path for path, file_path in model_paths.items() if not os.path.exists(file_path)]
    if missing_models:
        print(f"Warning: The following model files were not found: {missing_models}")
        print(f"Please ensure the models have been created by running the appropriate scripts first.")
        return
    
    # Create regression TIFF
    normalise = False  # Flag pour activer/désactiver la normalisation
    
    # Ajout du suffixe '_normalised' au nom de fichier si normalisation active
    if normalise:
        output_path = output_path.replace('.tif', '_normalised.tif')
    
    print(f"Creating regression TIFF using models from {regression_dir}")
    create_regression_tiff(
        model_paths=model_paths,
        bands_dir=bands_dir,
        indices_dir=indices_dir,
        output_path=output_path,
        mask_path=mask_path,  # Pass the mask path to the function
        normalise=normalise   # Active ou désactive la normalisation des proportions
    )
    
    print(f"Regression TIFF created at {output_path}")


def main():
    """
    Main function to create regression TIFFs using models saved by sentinel_proportion_median.py
    """
    # Default paths for models and data
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""
    mediane_dir = "mediane" if not moy5m else "mediane_10m"
    file_prefix = "" if not moy5m else "10m_"
    
    # Base directory where regression results are stored
    
    base_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}"
    regression_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_results"

    
    model_paths = {
        'sqrt_Pure_Lichen': os.path.join(regression_dir, 'sqrt_Pure_Lichen_rf.joblib'),
        'sqrt_Degraded_Lichen': os.path.join(regression_dir, 'sqrt_Degraded_Lichen_rf.joblib'),
        'sqrt_Green' : os.path.join(regression_dir, 'sqrt_Green_rf.joblib'),
        'sqrt_all_lichen' : os.path.join(regression_dir, 'sqrt_all_lichen_rf.joblib'),
        'sqrt_through_proportion' : os.path.join(regression_dir, 'sqrt_through_proportion_rf.joblib'),
    }
   
    # Sentinel data directories
    bands_dir = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_bands_10m/"
    indices_dir = f"DataCubeS2/WAP{wap}{peat_suffix}{resolution_suffix}/mediane_indices_10m/"
    
    # Mask path
    mask_path = f"drone_treated/WAP{wap}_tiles/mask_WAP{wap}{peat_suffix}.tif"
    
    # Output path
    output_path = os.path.join(base_dir, f"regression_predictions_WAP{wap}{peat_suffix}{resolution_suffix}.tif")
    
    # Check if all model files exist
    missing_models = [path for path, file_path in model_paths.items() if not os.path.exists(file_path)]
    if missing_models:
        print(f"Warning: The following model files were not found: {missing_models}")
        print(f"Please ensure the models have been created by running the appropriate scripts first.")
        return
    
    # Create regression TIFF
    normalise = False  # Flag pour activer/désactiver la normalisation
    
    # Ajout du suffixe '_normalised' au nom de fichier si normalisation active
    if normalise:
        output_path = output_path.replace('.tif', '_normalised.tif')
    
    print(f"Creating regression TIFF using models from {regression_dir}")
    create_regression_tiff(
        model_paths=model_paths,
        bands_dir=bands_dir,
        indices_dir=indices_dir,
        output_path=output_path,
        mask_path=mask_path,  # Pass the mask path to the function
        normalise=normalise   # Active ou désactive la normalisation des proportions
    )
    
    print(f"Regression TIFF created at {output_path}")

if __name__ == "__main__":
    main()
