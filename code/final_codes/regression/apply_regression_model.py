#!/usr/bin/env python3
"""
Apply Regression Model to Sentinel-2 Data

This script applies a trained regression model to Sentinel-2 bands and indices
to create a prediction map. It provides a simple, user-friendly interface for
generating land cover proportion maps from satellite imagery using pre-trained models.

Usage:
    python apply_regression_model.py --bands-dir path/to/bands --indices-dir path/to/indices 
                                    --model path/to/model.joblib --output path/to/output.tif
                                    [--square-transform]
"""

import os
import sys
import argparse
import numpy as np
import joblib
from osgeo import gdal
from tqdm import tqdm
from merge_proportion import normalize_feature_name

def parse_arguments():
    """
    Parse command line arguments for the regression model application.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Apply regression model to Sentinel-2 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--bands-dir', required=True,
                        help='Directory containing Sentinel-2 band GeoTIFFs')
    parser.add_argument('--indices-dir', required=True,
                        help='Directory containing Sentinel-2 indices GeoTIFFs')
    parser.add_argument('--model', required=True,
                        help='Path to trained regression model (.joblib file)')
    parser.add_argument('--output', required=True,
                        help='Path to save the output prediction GeoTIFF')
    
    # Optional arguments
    parser.add_argument('--square-transform', action='store_true',
                        help='Square the predictions (for sqrt-transformed models)')
    
    return parser.parse_args()

def load_regression_model(model_path):
    """
    Load a regression model from a joblib file.
    
    Args:
        model_path: Path to the joblib file containing the model
        
    Returns:
        Tuple of (model, feature_names, target_name)
    """
    print(f"Loading model from {model_path}...")
    
    model_data = joblib.load(model_path)
    
    # Check if it's a dictionary containing the model and metadata
    if isinstance(model_data, dict):
        model = model_data.get("model")
        feature_names = model_data.get("feature_names")
        target_name = model_data.get("target_name", "unknown")
        
        if model is None:
            raise ValueError("Model not found in the joblib file")
        
        print(f"Loaded model for target: {target_name}")
        print(f"Model requires {len(feature_names)} features")
        return model, feature_names, target_name
    else:
        # Direct model without metadata
        print("Loaded model without metadata")
        return model_data, None, "unknown"

def load_sentinel_features(bands_dir, indices_dir):
    """
    Load Sentinel bands and indices as features for regression.
    
    Args:
        bands_dir: Directory containing band GeoTIFFs
        indices_dir: Directory containing indices GeoTIFFs
        
    Returns:
        Tuple of:
        - features_array: 3D array of shape (n_features, height, width)
        - feature_names: List of feature names
        - width: Raster width in pixels
        - height: Raster height in pixels
        - geo_transform: GeoTransform for output
        - projection: Projection for output
    """
    # List all tif files in the directories
    band_files = sorted([os.path.join(bands_dir, f) for f in os.listdir(bands_dir) 
                        if f.lower().endswith('.tif')])
    index_files = sorted([os.path.join(indices_dir, f) for f in os.listdir(indices_dir) 
                         if f.lower().endswith('.tif')])
    
    print(f"Found {len(band_files)} band files and {len(index_files)} index files")
    
    # Use first band file as reference for dimensions
    if band_files:
        ref_ds = gdal.Open(band_files[0])
        print(f"Using dimensions from first band: {band_files[0]}")
    else:
        raise ValueError("No band files found to determine dimensions")
        
    if ref_ds is None:
        raise ValueError("Could not open reference band file")
    
    # Get dimensions and projection from reference
    width = ref_ds.RasterXSize
    height = ref_ds.RasterYSize
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    # Load and process bands and indices
    bands_data = []
    feature_names = []
    
    # Process band files
    print("Loading band files...")
    for tif in tqdm(band_files, desc="Bands"):
        ds = gdal.Open(tif)
        if ds is None:
            print(f"Warning: Could not open {tif}")
            continue
            
        # Get band name
        filename = os.path.basename(tif)
        feature_name = normalize_feature_name(filename)
        
        # Read the band data
        band_array = ds.GetRasterBand(1).ReadAsArray()
        
        # Handle NoData values (replacing with NaN)
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        if nodata is not None:
            band_array = band_array.astype(np.float32)
            band_array[band_array == nodata] = np.nan
            
        bands_data.append(band_array)
        feature_names.append(feature_name)
    
    # Process index files
    print("Loading index files...")
    for tif in tqdm(index_files, desc="Indices"):
        ds = gdal.Open(tif)
        if ds is None:
            print(f"Warning: Could not open {tif}")
            continue
            
        # Get index name
        filename = os.path.basename(tif)
        feature_name = normalize_feature_name(filename)
        
        # Read the index data
        index_array = ds.GetRasterBand(1).ReadAsArray()
        
        # Handle NoData values (replacing with NaN)
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        if nodata is not None:
            index_array = index_array.astype(np.float32)
            index_array[index_array == nodata] = np.nan
            
        bands_data.append(index_array)
        feature_names.append(feature_name)
    
    # Stack all bands into a 3D array (n_features, height, width)
    if not bands_data:
        raise ValueError("No valid bands or indices found")
        
    features_array = np.stack(bands_data, axis=0)
    print(f"Loaded {len(feature_names)} features with dimensions {features_array.shape}")
    
    return features_array, feature_names, width, height, geo_transform, projection

def apply_regression_model(model, required_feature_names, features_array, feature_names, 
                           width, height, geo_transform, projection, output_path, 
                           square_transform=False):
    """
    Apply regression model to features and save the prediction as a GeoTIFF.
    
    Args:
        model: Trained regression model
        required_feature_names: Feature names required by the model
        features_array: 3D array of features (n_features, height, width)
        feature_names: List of feature names corresponding to features_array
        width, height: Dimensions of the output raster
        geo_transform, projection: Georeferencing information
        output_path: Path to save the output GeoTIFF
        square_transform: Whether to square the predictions (for sqrt models)
        
    Returns:
        Path to the created GeoTIFF
    """
    print("Applying regression model to create prediction map...")
    
    # Map available features to required features
    feature_indices = []
    missing_features = []
    
    for req_name in required_feature_names:
        if req_name in feature_names:
            feature_indices.append(feature_names.index(req_name))
        else:
            missing_features.append(req_name)
    
    if missing_features:
        print(f"Warning: {len(missing_features)} required features are missing:")
        for i, name in enumerate(missing_features[:10]):  # Show up to 10 missing features
            print(f"  - {name}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
        
        raise ValueError("Missing required features. Check that your bands and indices match the model requirements.")
    
    # Select only the features required by the model
    model_features = features_array[feature_indices]
    
    # Reshape features for prediction (n_features, height, width) -> (height*width, n_features)
    n_features, height, width = model_features.shape
    features_flat = model_features.reshape(n_features, -1).T  # Transpose to get (height*width, n_features)
    
    # Create mask for valid pixels (non-NaN values)
    valid_mask = ~np.isnan(features_flat).any(axis=1)
    
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid pixels to process (all pixels have NaN values)")
    
    # Apply model to valid pixels
    print(f"Making predictions for {np.sum(valid_mask)} valid pixels...")
    predictions = np.full(valid_mask.shape, np.nan)
    predictions[valid_mask] = model.predict(features_flat[valid_mask])
    
    # Apply square transform if requested (for sqrt models)
    if square_transform:
        print("Applying square transform to predictions (sqrt model detected)")
        predictions[valid_mask] = predictions[valid_mask] ** 2
    
    # Reshape predictions back to 2D
    prediction_map = np.full((height, width), -1, dtype=np.float32)
    prediction_map.flat[:] = predictions
    
    # Scale proportions to 0-100% for integer storage
    prediction_int = np.full_like(prediction_map, -1, dtype=np.int16)
    valid_indices = ~np.isnan(prediction_map) & (prediction_map >= 0)
    prediction_int[valid_indices] = np.clip(prediction_map[valid_indices] * 100, 0, 100).astype(np.int16)
    
    # Create output GeoTIFF
    print(f"Creating output GeoTIFF at {output_path}")
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path, 
        width, 
        height, 
        1,  # One band for the prediction
        gdal.GDT_Int16,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    )
    
    # Set georeference information
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Write data and set nodata value
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(-1)
    out_band.WriteArray(prediction_int)
    
    # Add metadata
    out_ds.SetMetadata({
        "PREDICTION_TYPE": "Land cover proportion",
        "UNITS": "Percentage (0-100%)",
        "SQUARED_TRANSFORM": "True" if square_transform else "False"
    })
    
    # Close the dataset
    out_ds.FlushCache()
    out_ds = None
    
    print(f"Prediction map saved to {output_path}")
    print(f"Prediction statistics:")
    print(f"  Valid pixels: {np.sum(valid_indices)}")
    print(f"  Min value: {np.nanmin(prediction_map[valid_indices]):.4f}")
    print(f"  Max value: {np.nanmax(prediction_map[valid_indices]):.4f}")
    print(f"  Mean value: {np.nanmean(prediction_map[valid_indices]):.4f}")
    
    return output_path

def main():
    """
    Main function to apply a regression model to Sentinel-2 data.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate inputs
    for required_path in [args.bands_dir, args.indices_dir, args.model]:
        if not os.path.exists(required_path):
            print(f"Error: Path does not exist: {required_path}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Load the regression model
        model, feature_names, target_name = load_regression_model(args.model)
        print(feature_names)
        
        # Check if it's likely a sqrt model based on name
        if not args.square_transform and target_name.startswith('sqrt_'):
            print(f"Note: Target '{target_name}' suggests this is a sqrt-transformed model.")
            print(f"Consider using --square-transform if you want to convert predictions back to original scale.")
        
        # Load Sentinel features
        features_array, available_features, width, height, geo_transform, projection = \
            load_sentinel_features(args.bands_dir, args.indices_dir)
        print(available_features)
        
        # Apply the model and save the output
        apply_regression_model(
            model=model,
            required_feature_names=feature_names,
            features_array=features_array,
            feature_names=available_features,
            width=width,
            height=height,
            geo_transform=geo_transform,
            projection=projection,
            output_path=args.output,
            square_transform=args.square_transform
        )
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Example usage:
python apply_regression_model.py \
  --bands-dir path/to/sentinel_bands \
  --indices-dir path/to/sentinel_indices \
  --model path/to/lichen_model.joblib \
  --output path/to/lichen_prediction.tif \
  --square-transform
"""
        
# python C:Loris/CESBIO/stage_cesbio/code/final_codes/regression/apply_regression_model.py `
#   --bands-dir D:/Loris/SentinelBands/Chesnay_10m/mediane_bands `
#   --indices-dir D:/Loris/SentinelBands/Chesnay_10m/mediane_indices `
#   --model C:Loris/CESBIO/stage_cesbio/data/regressions/regression_multisite/results3/lichen/lichen_proportion_model.joblib `
#   --output C:Loris/CESBIO/stage_cesbio/data/regressions/regression_multisite/results3/lichen/Chesnay_Lichen_prediction.tif

