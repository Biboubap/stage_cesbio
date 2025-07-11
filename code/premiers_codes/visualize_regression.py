import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from interaction_sentinel_drone import load_proportion_csv, sentinel_to_drone_bounds, geo_to_pixel
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin

def load_sentinel_band(band_path):
    """Load a Sentinel-2 band as a NumPy array"""
    ds = gdal.Open(band_path)
    arr = ds.GetRasterBand(1).ReadAsArray()
    return arr, ds.GetGeoTransform()

def create_rgb_from_sentinel(red_path, green_path, blue_path, percentiles=(2, 98)):
    """
    Create an RGB image from Sentinel-2 bands with percentile-based normalization
    
    Args:
        red_path: Path to red band (B4)
        green_path: Path to green band (B3)
        blue_path: Path to blue band (B2)
        percentiles: Tuple with (min_percentile, max_percentile) for contrast stretching
        
    Returns:
        RGB array (height, width, 3) with values from 0-255
    """
    # Load bands
    red, _ = load_sentinel_band(red_path)
    green, _ = load_sentinel_band(green_path)
    blue, _ = load_sentinel_band(blue_path)
    
    # Create a function for normalization
    def normalize_band(band, p_low=2, p_high=98):
        # Compute percentiles
        min_val = np.percentile(band[band > 0], p_low)
        max_val = np.percentile(band[band > 0], p_high)
        
        # Clip and normalize to 0-255
        clipped = np.clip(band, min_val, max_val)
        normalized = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return normalized
    
    # Normalize each band
    r_norm = normalize_band(red, percentiles[0], percentiles[1])
    g_norm = normalize_band(green, percentiles[0], percentiles[1])
    b_norm = normalize_band(blue, percentiles[0], percentiles[1])
    
    # Stack into RGB
    rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
    return rgb

def predict_lichen_proportions(model_path, bands_dir, indices_dir, output_tif):
    """
    Use a trained Random Forest model to predict lichen proportions
    
    Args:
        model_path: Path to the trained model joblib file
        bands_dir: Directory with Sentinel-2 bands
        indices_dir: Directory with vegetation indices
        output_tif: Path to save the prediction GeoTIFF
    
    Returns:
        Predicted lichen proportions as a NumPy array
    """
    # Load model data (which is a dictionary containing the model and metadata)
    model_data = joblib.load(model_path)
    print(f"Model data loaded from {model_path}")
    
    # Extract the actual model and band names
    if isinstance(model_data, dict):
        # The model is stored in a dictionary with metadata
        rf_model = model_data["model"]
        band_names_model = model_data.get("band_names", None)
        r2_score = model_data.get("r2_score", None)
        sqrt_transform = model_data.get("sqrt_transform", False)
        print(f"Model extracted from dictionary (R²: {r2_score}, sqrt transform: {sqrt_transform})")
    else:
        # Direct model without metadata
        rf_model = model_data
        band_names_model = None
        print("Model loaded directly (no metadata)")
    
    # Get reference band to determine dimensions - only consider actual .tif files
    tif_files = [f for f in os.listdir(bands_dir) if f.lower().endswith('.tif') and not f.endswith('.aux.xml')]
    if not tif_files:
        raise ValueError(f"No TIF files found in {bands_dir}")
        
    ref_band_path = os.path.join(bands_dir, tif_files[0])
    ref_ds = gdal.Open(ref_band_path)
    if ref_ds is None:
        raise ValueError(f"Failed to open reference band: {ref_band_path}")
        
    ref_arr = ref_ds.GetRasterBand(1).ReadAsArray()
    height, width = ref_arr.shape
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    print(f"Prediction area: {width}x{height} pixels")
    
    # Load all bands/indices for prediction - only consider actual .tif files
    band_files = sorted([os.path.join(bands_dir, f) for f in os.listdir(bands_dir) 
                        if f.lower().endswith('.tif') and not f.endswith('.aux.xml')])
    indices_files = sorted([os.path.join(indices_dir, f) for f in os.listdir(indices_dir) 
                           if f.lower().endswith('.tif') and not f.endswith('.aux.xml')])
    all_files = band_files + indices_files
    
    print(f"Loading {len(all_files)} features...")
    features_stack = []
    for file_path in all_files:
        ds = gdal.Open(file_path)
        if ds is None:
            print(f"Warning: Could not open {file_path}, skipping...")
            continue
        arr = ds.GetRasterBand(1).ReadAsArray()
        features_stack.append(arr)
        ds = None  # Close the dataset

    # Prepare predictions array
    predictions = np.zeros((height, width), dtype=np.float32)
    
    # Make predictions
    print("Predicting lichen proportions...")
    for row in tqdm(range(height)):
        # Extract features for the entire row
        row_features = []
        for i in range(width):
            pixel_features = [band[row, i] for band in features_stack]
            row_features.append(pixel_features)
        
        # Predict entire row at once
        if len(row_features) > 0:
            row_predictions = rf_model.predict(row_features)  # Now using the extracted model
            
            # If the model was trained with sqrt transform, convert back
            if sqrt_transform:
                # Convert predictions back from sqrt space to original proportion space
                row_predictions = (row_predictions / 10) ** 2
            
            predictions[row, :] = row_predictions
    
    # Scale predictions to 0-1000 integer range
    print("Scaling predictions to 0-1000 range...")
    scaled_predictions = np.clip(predictions * 1000, 0, 1000).astype(np.uint16)
    
    # Save as GeoTIFF
    print(f"Saving predictions to {output_tif}...")
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, width, height, 1, gdal.GDT_UInt16)
    out_ds.GetRasterBand(1).WriteArray(scaled_predictions)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_ds.FlushCache()
    out_ds = None
    
    return predictions

def find_band_file(directory, band_pattern):
    """
    Find a Sentinel-2 band file in the specified directory that matches the pattern.
    
    Args:
        directory: Directory to search
        band_pattern: Pattern to match in filename (e.g., "BandB2" for blue band)
        
    Returns:
        Full path to the matching file, or None if not found
    """
    matching_files = [f for f in os.listdir(directory) 
                     if band_pattern in f and f.lower().endswith('.tif') and not f.endswith('.aux.xml')]
    
    if not matching_files:
        print(f"Warning: No files matching '{band_pattern}' found in {directory}")
        return None
    
    # Return first match (should be only one)
    return os.path.join(directory, matching_files[0])

def create_lichen_colormap():
    """Create a custom colormap from dark brown/orange (0%) to bright white (100%) for lichen proportions"""
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Define colors: orange-brown (0%) to bright white (100%)
    colors = [
        (0.6, 0.3, 0.0),  # Brown/orange (0%)
        (0.7, 0.4, 0.1),  # Lighter brown/orange (20%)
        (0.8, 0.6, 0.3),  # Light brown (40%)
        (0.9, 0.8, 0.7),  # Very light brown/beige (60%)
        (0.95, 0.9, 0.85),  # Almost white with slight warmth (80%)
        (1.0, 1.0, 1.0)  # Pure bright white (100%)
    ]
    
    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("lichen_cmap", colors, N=256)
    
    # Add masking for zero values (transparent/black)
    cmap.set_bad(color='black')
    
    return cmap

def create_visualization(drone_rgb_path, sentinel_dirs, prediction_array, output_png, 
                         region=None, sentinel_2_bands=("BandB4", "BandB3", "BandB2")):
    """
    Create a three-panel visualization: drone RGB, Sentinel RGB, and lichen prediction
    
    Args:
        drone_rgb_path: Path to drone RGB GeoTIFF
        sentinel_dirs: Dictionary with 'bands' and 'indices' keys for Sentinel data
        prediction_array: Numpy array with lichen predictions (0-1 range)
        output_png: Path to save the visualization
        region: Optional tuple (xmin, ymin, xmax, ymax) to limit visualization to a region
        sentinel_2_bands: Tuple of band patterns to use for RGB visualization
    """
    # Load drone RGB
    drone_ds = gdal.Open(drone_rgb_path)
    drone_transform = drone_ds.GetGeoTransform()
    
    # Get Sentinel band paths using pattern matching
    red_path = find_band_file(sentinel_dirs['bands'], sentinel_2_bands[0])
    green_path = find_band_file(sentinel_dirs['bands'], sentinel_2_bands[1])
    blue_path = find_band_file(sentinel_dirs['bands'], sentinel_2_bands[2])
    
    if not all([red_path, green_path, blue_path]):
        raise ValueError("Could not find all required Sentinel-2 bands")
    
    # Create Sentinel RGB
    sentinel_rgb = create_rgb_from_sentinel(red_path, green_path, blue_path)
    
    # Determine region to display - using the entire Twin Lake area instead of just a portion
    if region is not None:
        xmin, ymin, xmax, ymax = region
    else:
        # Use the entire Sentinel image for Twin Lake
        s2_ds = gdal.Open(red_path)
        s2_width = s2_ds.RasterXSize
        s2_height = s2_ds.RasterYSize
        
        xmin = 0
        ymin = 0
        xmax = s2_width
        ymax = s2_height
    
    # Extract region from Sentinel RGB and prediction
    s2_rgb_region = sentinel_rgb[ymin:ymax, xmin:xmax]
    pred_region = prediction_array[ymin:ymax, xmin:xmax]
    
    # Find corresponding region in drone image
    drone_region = extract_drone_region(xmin, ymin, xmax, ymax, red_path, drone_rgb_path)
    
    # Create colormap for lichen predictions
    cmap = create_lichen_colormap()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot drone RGB
    axes[0].imshow(drone_region)
    axes[0].set_title("Drone RGB")
    axes[0].axis("off")
    
    # Plot Sentinel RGB
    axes[1].imshow(s2_rgb_region)
    axes[1].set_title("Sentinel-2 RGB (Bands 4-3-2)")
    axes[1].axis("off")
    
    # Plot prediction - mask out zero/nan values to be shown as black
    masked_pred = np.ma.masked_where((pred_region == 0) | np.isnan(pred_region), pred_region)
    im = axes[2].imshow(masked_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Predicted Lichen Proportion")
    axes[2].axis("off")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Lichen Proportion (0-100%)")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_png}")

def extract_drone_region(xmin, ymin, xmax, ymax, sentinel_path, drone_path):
    """
    Extract the region from drone image corresponding to the given Sentinel region
    """
    # Get Sentinel and drone transforms
    s2_ds = gdal.Open(sentinel_path)
    s2_transform = s2_ds.GetGeoTransform()
    
    drone_ds = gdal.Open(drone_path)
    drone_transform = drone_ds.GetGeoTransform()
    
    # Get geo coordinates of the Sentinel region
    x_geo_min, y_geo_max = gdal.ApplyGeoTransform(s2_transform, xmin, ymin)
    x_geo_max, y_geo_min = gdal.ApplyGeoTransform(s2_transform, xmax, ymax)
    
    # Convert to drone pixel coordinates
    drone_xmin, drone_ymin = geo_to_pixel(drone_transform, x_geo_min, y_geo_min)
    drone_xmax, drone_ymax = geo_to_pixel(drone_transform, x_geo_max, y_geo_max)
    
    # Ensure coordinates are within bounds
    drone_width = drone_ds.RasterXSize
    drone_height = drone_ds.RasterYSize
    
    drone_xmin = max(0, min(drone_xmin, drone_width-1))
    drone_xmax = max(0, min(drone_xmax, drone_width-1))
    drone_ymin = max(0, min(drone_ymin, drone_height-1))
    drone_ymax = max(0, min(drone_ymax, drone_height-1))
    
    # Ensure xmin < xmax and ymin < ymax
    drone_xmin, drone_xmax = min(drone_xmin, drone_xmax), max(drone_xmin, drone_xmax)
    drone_ymin, drone_ymax = min(drone_ymin, drone_ymax), max(drone_ymin, drone_ymax)
    
    # Load the drone RGB data
    drone_rgb = np.zeros((drone_ymax - drone_ymin, drone_xmax - drone_xmin, 3), dtype=np.uint8)
    
    for i in range(3):
        band = drone_ds.GetRasterBand(i+1)
        data = band.ReadAsArray(drone_xmin, drone_ymin, drone_xmax - drone_xmin, drone_ymax - drone_ymin)
        drone_rgb[:, :, i] = data
    
    return drone_rgb

def main():
    # Paths to input data
    model_path = "data/samples/selection8/regression/rf_lichen_regression_model.joblib"
    bands_dir = "DataCubeS2/Bandes/mediane2"
    indices_dir = "DataCubeS2/TwinLakeCubeIndex/mediane"
    drone_rgb_path = "data/rgb_reshaped.tif"
    
    # Output paths
    output_dir = "data/samples/selection8/regression_results"
    os.makedirs(output_dir, exist_ok=True)
    output_tif = os.path.join(output_dir, "lichen_proportion_prediction.tif")
    output_png = os.path.join(output_dir, "lichen_visualization_full.png")
    
    # Create sentinel directories dictionary
    sentinel_dirs = {
        'bands': bands_dir,
        'indices': indices_dir
    }
    
    # Predict lichen proportions
    predictions = predict_lichen_proportions(model_path, bands_dir, indices_dir, output_tif)
    
    # Create visualization using pattern-based band selection, showing the entire area
    create_visualization(drone_rgb_path, sentinel_dirs, predictions, output_png, 
                        sentinel_2_bands=("BandB4", "BandB3", "BandB2"))
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
