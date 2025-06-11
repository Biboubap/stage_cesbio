"""
Create a multi-band TIFF file containing the predicted proportions from regression models.
Each band represents a different class proportion (lichen, chicoutai_green, through_proportion),
scaled from 0-100 (percentage) and normalized to ensure they sum to 100%.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import joblib
import argparse
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
        elif "models" in model_data:
            # Models dictionary for individual target predictors
            model = model_data["models"]
            feature_names = model_data.get("feature_names", None)
            return model, feature_names
        elif "class_names" in model_data:
            # Grouped models from sentinel_regression4.py
            models = model_data.get("models", {})
            feature_names = model_data.get("feature_names", None)
            class_names = model_data.get("class_names", [])
            
            # Map the grouped class names to standard names
            grouped_to_standard = {}
            for class_name in class_names:
                if 'lichen' in class_name:
                    grouped_to_standard['lichen'] = models[class_name]
                elif 'chicoutai_green' in class_name or 'group2' in class_name:
                    grouped_to_standard['chicoutai_green'] = models[class_name]
                elif 'through' in class_name or 'group3' in class_name:
                    grouped_to_standard['through_proportion'] = models[class_name]
            
            return grouped_to_standard, feature_names
    
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

def create_regression_tiff(model_paths, bands_dir, indices_dir, output_path, sqrt_transform=False, normalise=True):
    """
    Create a multi-band TIFF with regression predictions for each target class
    
    Args:
        model_paths: Dictionary of paths to models for each target class
                     (e.g., {'lichen': 'path/to/lichen_model.joblib', ...})
                     or {'grouped_model': 'path/to/grouped_model.joblib'} for grouped classes
        bands_dir: Directory containing Sentinel band TIFs
        indices_dir: Directory containing Sentinel index TIFs
        output_path: Path to save the output TIFF
        sqrt_transform: Whether to apply inverse sqrt transform to through_proportion predictions
        normalise: Whether to normalise predictions so that their sum is 100% for each pixel
    """
    # Check if we're using grouped models
    is_grouped_model = len(model_paths) == 1 and 'grouped_model' in model_paths
    
    # For grouped models, load the single model file and extract individual models
    if is_grouped_model:
        models, feature_names = load_regression_model(model_paths['grouped_model'])
        target_classes = list(models.keys())
    else:
        # Get target classes from model_paths for individual models
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
    prediction_arrays = {target_class: np.zeros((height, width), dtype=np.float32) 
                        for target_class in target_classes}
    
    # Make predictions
    print("Making predictions...")
    for row in tqdm(range(height)):
        # Extract features for the entire row
        row_features = []
        for col in range(width):
            # Extract features for this pixel
            pixel_features = [band[row, col] for band in features]
            row_features.append(pixel_features)
        
        # Convert to numpy array
        X = np.array(row_features)
        
        # Predict for each target class
        for target_class in target_classes:
            model = models[target_class]
            
            if target_class == 'through_proportion' and sqrt_transform:
                # Apply inverse sqrt transform for through_proportion predictions
                raw_predictions = model.predict(X)
                predictions = raw_predictions ** 2  # Convert back from sqrt space
            else:
                predictions = model.predict(X)
            
            prediction_arrays[target_class][row, :] = predictions
    
    if normalise:
        # Normalize predictions to sum to 1 (100%)
        print("Normalizing predictions...")
        sum_array = np.zeros((height, width), dtype=np.float32)
        for target_class in target_classes:
            prediction_arrays[target_class] = np.maximum(0, prediction_arrays[target_class])  # Clip negative values
            sum_array += prediction_arrays[target_class]
        
        # Avoid division by zero
        sum_array = np.where(sum_array > 0, sum_array, 1)
        
        # Scale to 0-100% range as integers
        for i, target_class in enumerate(target_classes):
            normalized = np.divide(prediction_arrays[target_class], sum_array) * 100
            normalized_int = np.clip(normalized, 0, 100).astype(np.int16)
            
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
            # Assurons-nous que les valeurs sont positives
            prediction_arrays[target_class] = np.maximum(0, prediction_arrays[target_class])
            
            # Convertir les proportions (0-1) en pourcentages (0-100)
            scaled = prediction_arrays[target_class] * 100
            clipped = np.clip(scaled, 0, 100).astype(np.int16)
            
            # Write to band
            band = out_ds.GetRasterBand(i + 1)
            band.WriteArray(clipped)
            band.SetDescription(target_class)
            band.SetNoDataValue(-1)
            band.FlushCache()

    # Add band descriptions
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": target_class for i, target_class in enumerate(target_classes)})
    out_ds = None
    
    print(f"Multi-band regression TIFF created at {output_path}")
    
    # Create a color table for visualization
    create_color_interpretation_file(output_path, target_classes)
    
    # Also create a visualization of the predictions
    create_visualization(prediction_arrays, output_path.replace('.tif', '_visualization.png'))

def create_color_interpretation_file(tiff_path, classes):
    """
    Create a color interpretation file for QGIS visualization
    
    Args:
        tiff_path: Path to the multi-band TIFF file
        classes: List of class names in band order
    """
    # Define color scheme for visualization
    colors = {
        "lichen": "#C8C8C8",           # Light gray
        "chicoutai_green": "#228B22",   # Forest green
        "through_proportion": "#A0522D", # Sienna
        "sqrt_through_proportion": "#A0522D", # Same as through_proportion
        "chicoutai": "#006400",        # Dark green
        "green_depression": "#32CD32"   # Lime green
    }
    
    # Create .vrt file
    vrt_path = tiff_path.replace('.tif', '.vrt')
    vrt_options = gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(vrt_path, [tiff_path], options=vrt_options)
    
    # Create .qml file for QGIS
    qml_path = tiff_path.replace('.tif', '.qml')
    with open(qml_path, 'w') as f:
        f.write("""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.4-Białowieża">
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" name="name" value=""/>
      <Option name="properties"/>
      <Option type="QString" name="type" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer opacity="1" type="singlebandpseudocolor" band="1" classificationMin="0" classificationMax="100">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader maximumValue="100" classificationMode="1" colorRampType="INTERPOLATED" clip="0" labelPrecision="0" minimumValue="0">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option type="QString" name="color1" value="#ffffff"/>
              <Option type="QString" name="color2" value="{}"/>
              <Option type="QString" name="discrete" value="0"/>
              <Option type="QString" name="rampType" value="gradient"/>
            </Option>
          </colorramp>
          <item label="0%" alpha="0" color="#ffffff" value="0"/>
          <item label="50%" alpha="128" color="{}" value="50"/>
          <item label="100%" alpha="255" color="{}" value="100"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" invertColors="0" colorizeBlue="128" grayscaleMode="0" colorizeOn="0" saturation="0" colorizeRed="255" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
""".format(colors.get(classes[0], "#ff0000"), 
            colors.get(classes[0], "#ff0000"), 
            colors.get(classes[0], "#ff0000")))
    
    print(f"Created color interpretation files: {vrt_path} and {qml_path}")
    print("NOTE: In QGIS, use the 'Select Band' option in layer properties to view each class proportion")

def create_visualization(prediction_arrays, output_path):
    """
    Create a visualization of the predictions as a multi-panel image
    
    Args:
        prediction_arrays: Dictionary of prediction arrays for each target class
        output_path: Path to save the visualization
    """
    # Define colors for visualization
    colors = {
        "lichen": plt.cm.Greys_r,           # Grayscale
        "chicoutai_green": plt.cm.Greens,   # Green colormap
        "through_proportion": plt.cm.YlOrBr, # Yellow-Orange-Brown colormap
        "sqrt_through_proportion": plt.cm.YlOrBr, # Same as through_proportion
    }
    
    # Create figure
    n_classes = len(prediction_arrays)
    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes*6, 6))
    
    # Ensure axes is always a list
    if n_classes == 1:
        axes = [axes]
    
    # Plot each prediction
    for i, (target_class, pred_array) in enumerate(prediction_arrays.items()):
        # Normalize to 0-1 for visualization
        norm_array = np.clip(pred_array / 100.0, 0, 1)
        
        # Choose colormap
        cmap = colors.get(target_class, plt.cm.viridis)
        
        # Plot
        im = axes[i].imshow(norm_array, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(f"{target_class}")
        axes[i].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label(f"{target_class} Proportion (0-100%)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def main():
    """
    Main function to create regression TIFFs using models saved by sentinel_proportion_median.py
    """
    # Default paths for models and data
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    
    # Set path modifiers based on superresolution flag
    mediane_dir = "mediane" if superresolution else "mediane_10m"
    file_prefix = "" if superresolution else "10m_"
    resolution = 5 if superresolution else 10
    
    # Base directory where regression results are stored
    resolution_suffix = "" if superresolution else "_10m"
    base_dir = f"data/samples/selection14/regression_wap{wap}_no_chicoutai{peat_suffix}{resolution_suffix}"
    regression_dir = f"{base_dir}/regression_results"
    results_dir = f"{base_dir}/results"  # Directory for sentinel_regression4.py results
    
    # Choose model type - options:
    # - 'individual': Individual models for each class (from sentinel_proportion_median.py)
    # - 'grouped': Grouped class models (from sentinel_regression4.py)
    model_type = 'individual' 
    
    # Input model paths based on model type
    if model_type == 'individual':
        model_paths = {
            'lichen': os.path.join(regression_dir, 'individual_lichen_rf.joblib'),
            'chicoutai_green': os.path.join(regression_dir, 'individual_chicoutai_green_rf.joblib'),
            'through_proportion': os.path.join(regression_dir, 'individual_sqrt_through_rf.joblib')
        }
    elif model_type == 'grouped':  # 'grouped'
        model_paths = {
            'grouped_model': os.path.join(results_dir, 'grouped_classes_rf_models_sqrt.joblib')
        }
    
    # Sentinel data directories
    bands_dir = f"DataCubeS2/BandsS22023_WAP{wap}{peat_suffix}/{mediane_dir}"
    indices_dir = f"DataCubeS2/IndicesS22023_WAP{wap}{peat_suffix}/{mediane_dir}"
    
    # Output path
    output_path = os.path.join(base_dir, f"regression_predictions_{model_type}_WAP{wap}{peat_suffix}{resolution_suffix}.tif")
    
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
    
    print(f"Creating regression TIFF using models from {regression_dir if model_type == 'individual' else results_dir}")
    create_regression_tiff(
        model_paths=model_paths,
        bands_dir=bands_dir,
        indices_dir=indices_dir,
        output_path=output_path,
        sqrt_transform=True,  # Apply inverse sqrt transform for through_proportion
        normalise=normalise   # Active ou désactive la normalisation des proportions
    )
    
    print(f"Regression TIFF created at {output_path}")

if __name__ == "__main__":
    main()
