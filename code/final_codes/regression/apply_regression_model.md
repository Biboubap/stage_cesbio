# Applying Regression Models to Sentinel-2 Data

This guide explains how to use the `apply_regression_model.py` script to generate land cover proportion maps from Sentinel-2 satellite imagery using pre-trained regression models.

## Overview

The `apply_regression_model.py` script takes a trained regression model (created by `train_regression_model.py`) and applies it to Sentinel-2 bands and indices to produce a GeoTIFF map showing predicted land cover proportions. This is the final step in the regression pipeline that allows you to generate proportion maps for new areas where only Sentinel-2 data is available.

## Requirements

### Python Dependencies

- **Core Python**: Python 3.7 or newer
- **Data Handling**:
  - numpy (1.19.0+)
  - pandas (1.0.0+)
- **Geospatial Libraries**:
  - gdal/osgeo (3.0.0+)
- **Machine Learning**:
  - joblib (1.0.0+)
  - scikit-learn (0.24.0+)
- **Progress Visualization**:
  - tqdm (4.45.0+)

### Input Requirements

1. **Sentinel-2 bands directory**: Directory containing GeoTIFF files of Sentinel-2 spectral bands
   - Recommended: Use median composites (created with `sentinel_median.sh`)
   - Each band should be in a separate file
   - All bands must have the same spatial resolution and projection

2. **Sentinel-2 indices directory**: Directory containing GeoTIFF files of spectral indices
   - Common indices: NDVI, NDWI, GNDVI, etc. Check what indices are required by the model.
   - Must be computed from the same Sentinel-2 images as the bands

3. **Trained model**: A `.joblib` file containing a trained regression model
   - Must include the model itself and required feature names
   - Created using `train_regression_model.py`

## Usage

### Basic Command

```bash
python apply_regression_model.py --bands-dir path/to/bands --indices-dir path/to/indices \
                               --model path/to/model.joblib --output path/to/output.tif
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--bands-dir` | Directory containing Sentinel-2 band GeoTIFFs | Yes |
| `--indices-dir` | Directory containing Sentinel-2 indices GeoTIFFs | Yes |
| `--model` | Path to the trained regression model (.joblib file) | Yes |
| `--output` | Path to save the output prediction GeoTIFF | Yes |
| `--square-transform` | Square the predictions (for sqrt-transformed models) | No |

### Example Commands

#### Windows PowerShell:

```powershell
python C:/Loris/CESBIO/stage_cesbio/code/final_codes/regression/apply_regression_model.py `
  --bands-dir D:/Loris/SentinelBands/Chesnay_10m/mediane_bands `
  --indices-dir D:/Loris/SentinelBands/Chesnay_10m/mediane_indices `
  --model C:/Loris/CESBIO/stage_cesbio/data/regressions/regression_multisite/results/lichen/lichen_proportion_model.joblib `
  --output C:/Loris/CESBIO/stage_cesbio/data/regressions/Chesnay_Lichen_prediction.tif `
  --square-transform
```

#### Linux:

```bash
python /home/user/stage_cesbio/code/final_codes/regression/apply_regression_model.py \
  --bands-dir /home/user/data/sentinel/bands \
  --indices-dir /home/user/data/sentinel/indices \
  --model /home/user/models/lichen_proportion_model.joblib \
  --output /home/user/results/lichen_prediction.tif \
  --square-transform
```

## Output

The script produces a single-band GeoTIFF file with the following characteristics:

1. **Data Type**: Int16 (16-bit signed integer)
2. **Values**: 0-100 (representing percentages)
3. **NoData Value**: -1
4. **Georeferencing**: Preserves the georeferencing from the input Sentinel data
5. **Metadata**: Includes information about the prediction type and whether square transform was applied

The output GeoTIFF is optimized with:
- LZW compression
- Tiled structure (256x256 pixel blocks)
- Appropriate metadata

## Workflow

The script follows this workflow:

1. **Load the regression model**:
   - Reads the `.joblib` file containing the model and required feature names
   - Detects if the model was trained on sqrt-transformed data

2. **Load Sentinel-2 data**:
   - Reads all band and index files from the specified directories
   - Normalizes feature names for consistency with the model requirements
   - Handles NoData values by converting them to NaN

3. **Map features to model requirements**:
   - Identifies which loaded bands/indices correspond to the features required by the model
   - Reports any missing required features

4. **Make predictions**:
   - Applies the regression model to valid pixels
   - Optionally squares predictions (for sqrt-transformed models)

5. **Save results**:
   - Converts proportions to percentage (0-100)
   - Creates a GeoTIFF with proper georeferencing
   - Reports statistics about the prediction

## Implementation Details

### Feature Name Normalization

The script uses the `normalize_feature_name()` function from `merge_proportion.py` to ensure consistent feature naming between training and inference. This function extracts the core identifier (e.g., "B12", "NDVI") from filenames that might have site-specific prefixes or suffixes.

### NoData Handling

- Input NoData values are converted to NaN
- Only pixels that have valid data in all required bands/indices are processed
- Output pixels that couldn't be predicted are assigned the NoData value (-1)

### Square Transform

When using a model trained on sqrt-transformed data (e.g., `sqrt_lichen_proportion`), use the `--square-transform` flag to convert predictions back to the original scale. The script automatically suggests this if the model's target name starts with "sqrt_".

## Common Issues and Solutions

### Missing Features

**Issue**: Error message about missing required features.

**Solution**: Check that your bands and indices directories contain files with names that correctly normalize to the features required by the model. You can see the required features in the script output.

### Memory Issues

**Issue**: Out of memory errors with large Sentinel images.

**Solution**: The script processes the entire image at once. For extremely large images, consider splitting them into smaller tiles and processing each separately.

### Value Range

**Issue**: Predicted proportions are outside the expected 0-1 range.

**Solution**: 
1. For values slightly outside range: The script automatically clips values to 0-100% range
2. For severely out-of-range values: Your model might be applied to data that's very different from what it was trained on

## Code Structure

The script is organized into these main functions:

1. `parse_arguments()`: Parses command line arguments
2. `load_regression_model()`: Loads the model and extracts feature requirements
3. `load_sentinel_features()`: Loads and preprocesses Sentinel data
4. `apply_regression_model()`: Makes predictions and saves the output
5. `main()`: Orchestrates the entire process

## Tips for Best Results

1. **Use median composites** for Sentinel data to reduce noise and artifacts
2. **Apply the square transform** when using models trained on sqrt-transformed data
3. **Check model feature requirements** before running to ensure you have all necessary bands/indices
4. **Inspect the output statistics** to verify that predictions are in a reasonable range
5. **Compare with validation data** if available to assess prediction quality
