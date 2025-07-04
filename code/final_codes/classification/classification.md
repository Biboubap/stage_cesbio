# Classification System Documentation

## Overview

This classification system is designed to perform efficient land cover classification on very large drone-acquired imagery. It implements a parallel processing approach that divides large images into manageable blocks and processes them concurrently.

The system uses two pre-trained machine learning models in combination to classify different peatland cover types. It's specifically optimized for analyzing high-resolution drone RGB imagery along with Digital Surface Models (DSM) to identify features like lichen, vegetation, and depressions.

## System Structure

The classification system consists of the following components:

1. **Main Processing Script**: `process_classification.py`
   - Orchestrates the entire classification process
   - Handles command line arguments and resource management
   - Divides large images into processing blocks
   - Uses parallel processing with Dask

2. **Utility Classes**:
   - `BlockRastersManager`: Manages raster data loading and extraction
   - `BlockSample`: Represents patches within the raster and calculates their features
   - `block_processor`: Contains functions for feature extraction and prediction

3. **Machine Learning Models**:
   - Two separate models are used in combination:
     - `model_wap32_no_chicoutai.joblib`: General classification model
     - `model_16_7.joblib`: Detailed classification model for specific classes

## Processing Flow

The classification process follows these steps:

1. **Initialization**:
   - Parse command line arguments
   - Set up parallel processing environment with Dask
   - Load classification models
   - Create output raster file

2. **Block Processing**:
   - Divide input raster into overlapping blocks to avoid edge effects
   - For each block:
     - Load block data from RGB and DSM rasters
     - Divide block into patches
     - Calculate statistics for each patch (RGB means, variances, etc.)
     - Calculate neighborhood statistics
     - Extract features for machine learning models
     - Apply both classification models
     - Merge predictions from both models
     - Filter isolated pixels to reduce noise
     - Upsample to full resolution

3. **Merging Results**:
   - Combine valid portions of each processed block
   - Write final classification to output raster

## Requirements

To use the classification system, you need:

1. **Input Data**:
   - RGB image (GeoTIFF format)
   - Digital Surface Model (DSM) (GeoTIFF format)
   - Both rasters must have the same resolution and dimensions

2. **Machine Learning Models**:
   - `model_wap32_no_chicoutai.joblib`
   - `model_16_7.joblib`

3. **Python Dependencies**:
   - Core: numpy, gdal (osgeo)
   - Parallel processing: dask, distributed
   - Machine learning: joblib
   - System: os, psutil, logging

## Usage

### Basic Command

```bash
python process_classification.py --rgb path/to/rgb.tif --dsm path/to/dsm.tif --out path/to/output.tif
```

### Advanced Options

```bash
python process_classification.py \
  --rgb path/to/rgb.tif \
  --dsm path/to/dsm.tif \
  --out path/to/output.tif \
  --patch-size 16 \
  --block-size 1024 \
  --overlap 48 \
  --workers 4 \
  --memory-limit 8
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--rgb` | Path to RGB image (GeoTIFF) |
| `--dsm` | Path to Digital Surface Model (GeoTIFF) |
| `--out` | Path to save classification output |
| `--patch-size` | Size of analysis patches in pixels (default: auto-calculated) |
| `--block-size` | Size of processing blocks in pixels (default: 1024) |
| `--overlap` | Overlap between blocks in pixels (default: 48) |
| `--workers` | Number of parallel worker processes (default: auto) |
| `--memory-limit` | Memory limit per worker in GB (default: auto) |
| `--threads-per-worker` | Threads per worker (default: 1) |

## Output

The system produces a GeoTIFF raster where each pixel value represents a specific land cover class:

| Value | Class | Description |
|-------|-------|-------------|
| 0 | No Data | Areas with no classification (background) |
| 1 | Pure Lichen | Areas covered predominantly by lichen |
| 2 | Degraded Lichen | Areas with partially degraded lichen cover |
| 3 | Green | Areas with green vegetation |
| 4 | Sphagnum | Areas covered by sphagnum moss |
| 5 | Depression | Dry or non-vegetated depression areas |
| 6 | Water | Areas covered by water |

The output raster has the same georeferencing, projection, and dimensions as the input RGB raster, ensuring it aligns perfectly with the source data.

## Adapting the System

### Using Different Models

To use different machine learning models:

1. Replace the model paths in `process_classification.py`:
   ```python
   MODEL1_PATH = "/path/to/your/first_model.joblib"
   MODEL2_PATH = "/path/to/your/second_model.joblib"
   ```

2. If your models use different features, ensure the feature extraction in `extract_features()` and `filter_features()` functions matches your model requirements.

3. Update the class mapping in the `merge_predictions()` function to match your models' output classes.

### Adding Thermal Data

The system supports thermal data but doesn't require it. To include thermal data:

1. Extend the `BlockRastersManager` initialization in `process_block_with_overlap()`:
   ```python
   block_rasters = BlockRastersManager(rgb_path=args.rgb, dsm_path=args.dsm, thermal_path=args.thermal)
   ```

2. Add a command line argument for the thermal data path:
   ```python
   parser.add_argument('--thermal', help='Path to thermal raster')
   ```

The current model is not using thermal data. However, adding it can be useful if you develop a new model.

### Using Different Raster Types

The system can be adapted to use different types of input data:

1. Modify the `BlockRastersManager` class to handle your specific raster types.
2. Update the feature extraction to use the appropriate statistics from your raster types.
3. Retrain the models on features extracted from your data types.

## Performance Considerations

- **Memory Usage**: Adjust the `--block-size` parameter based on available memory. Smaller blocks use less memory but increase processing overhead.
- **Parallelism**: The `--workers` parameter controls how many blocks are processed simultaneously. More workers increase speed but require more memory.
- **Patch Size**: The `--patch-size` parameter should be set according to the scale of features you're trying to classify. The system can calculate this automatically based on raster resolution.

## Troubleshooting

- **Memory Errors**: Reduce block size or increase memory limit with `--memory-limit`.
- **Missing Features**: Check that your input rasters match what the models expect. The models require RGB and DSM data at minimum.
- **Edge Artifacts**: Increase overlap with `--overlap` if you see visible artifacts at block boundaries.
- **Classification Errors**: The models are trained on specific peatland environments. For different environments, retraining may be necessary.

## File Descriptions

### `process_classification.py`

Main script that orchestrates the classification process. It handles command-line arguments, sets up parallel processing, and coordinates the classification workflow.

### `utils/block_rasters_manager.py`

Class for managing raster data in blocks. Handles loading of RGB, DSM, and optionally thermal data, and provides methods to extract patches and calculate statistics.

### `utils/block_sample.py`

Class representing a patch within a block. Calculates various statistics for the patch, including neighborhood information, which are used as features for classification.

### `utils/block_processor.py`

Contains functions for feature extraction, prediction, and post-processing. Includes the core processing function `process_block_with_overlap()` that handles individual blocks.
