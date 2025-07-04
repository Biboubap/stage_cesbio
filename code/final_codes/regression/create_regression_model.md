# Creating Regression Models for Land Cover Proportion Prediction

This guide explains the complete workflow for creating regression models that predict land cover proportions from Sentinel-2 satellite imagery. These models can estimate the percentage of different land cover types (such as lichen, vegetation, and wetland) within each Sentinel-2 pixel using machine learning.

## Overview of the Process

The regression model creation involves four main stages:

1. **Computing Proportions** - Calculate the proportions of each land cover class within Sentinel-2 pixels using high-resolution drone classification maps
2. **Merging Data** - Combine proportion data from multiple sites to create a diverse training dataset
3. **Balancing Data** - Ensure even representation across the range of proportion values
4. **Training Models** - Train and evaluate Random Forest regression models using cross-validation

## Prerequisites

Before starting, you'll need:

1. **High-resolution classification maps** - GeoTIFF files from drone imagery with pixel-level classification
2. **Sentinel-2 imagery** - Preferably median composites of multiple acquisitions
3. **Python environment** with the required dependencies:
   - numpy, pandas, matplotlib, scipy
   - gdal/osgeo
   - scikit-learn
   - joblib
   - tqdm

## Detailed Workflow

### 1. Computing Proportions (compute_proportion.py)

This script calculates what percentage of each Sentinel-2 pixel is covered by each land cover class in the high-resolution drone classification map.

#### Key Inputs:
- `--classification`: Path to the classified drone imagery (GeoTIFF)
- `--sentinel-band`: Path to a reference Sentinel-2 band (for pixel grid)
- `--bands-dir`: Directory containing Sentinel-2 spectral bands
- `--indices-dir`: Directory containing Sentinel-2 spectral indices
- `--output-dir`: Directory to save outputs
- `--site-name`: Name of the site (for labeling)

#### Key Outputs:
- `proportions_[site]_proportions.json`: JSON file containing pixel data with features and proportions
- `proportions_[site].tif`: GeoTIFF showing spatial distribution of proportions

#### Example Command:
```bash
python compute_proportion.py \
  --classification path/to/classification.tif \
  --sentinel-band path/to/sentinel_B4_band.tif \
  --bands-dir path/to/sentinel_bands \
  --indices-dir path/to/sentinel_indices \
  --output-dir output/regression_site1 \
  --site-name Site1
```

#### How It Works:
1. For each Sentinel-2 pixel, the script identifies the corresponding area in the drone classification
2. It counts the number of drone pixels of each class within that area
3. It calculates proportions for three main categories:
   - **Lichen** (Pure_Lichen + Degraded_Lichen)
   - **Green** vegetation
   - **Trough** (Sphagnum + Depression + Water)
4. It extracts Sentinel-2 spectral values (bands and indices) for each pixel
5. It combines the proportion data with spectral data to create a JSON file

### 2. Merging Data (merge_proportion.py)

This script combines proportion data from multiple sites to create a larger, more diverse dataset.

#### Key Inputs:
- List of JSON files from different sites (from compute_proportion.py)
- `--output`: Path for the merged JSON file
- `--plots-dir`: Directory to save visualization plots

#### Key Outputs:
- `merged_pixels.json`: Combined dataset with standardized feature names
- Histogram plots showing class distributions

#### Example Command:
```bash
python merge_proportion.py \
  data/regression_site1/proportions_Site1_proportions.json \
  data/regression_site2/proportions_Site2_proportions.json \
  data/regression_site3/proportions_Site3_proportions.json \
  --output data/merged/merged_pixels.json \
  --plots-dir data/merged/plots
```

#### How It Works:
1. Loads proportion data from multiple JSON files
2. Normalizes feature names to ensure consistency across sites
3. Combines all pixel data into a single dataset
4. Creates visualizations showing the distribution of proportions

### 3. Balancing Data (balance_proportion.py)

This script balances the merged dataset to ensure even representation across the range of proportion values.

#### Key Inputs:
- Merged JSON file (from merge_proportion.py)
- `--output`: Directory for balanced datasets
- `--bins`: Number of bins to divide the data into (default: 25)
- `--quantile`: Quantile for determining samples per bin (default: 0.5)

#### Key Outputs:
- Separate balanced JSON files for each proportion type:
  - `balanced_lichen_proportion.json`
  - `balanced_green_proportion.json`
  - `balanced_trough_proportion.json`
- Histogram plots showing the balanced distributions

#### Example Command:
```bash
python balance_proportion.py \
  data/merged/merged_pixels.json \
  --output data/balanced \
  --quantile 0.6 \
  --bins 25
```

#### How It Works:
1. Divides the range of proportion values (0-1) into equal-width bins
2. Counts the number of samples in each bin
3. Determines a target sample count based on the specified quantile
4. Randomly samples from each bin to create a more balanced distribution
5. Creates separate balanced datasets for each proportion type

### 4. Training Models (train_regression_model.py)

This script trains and evaluates Random Forest regression models using the balanced datasets.

#### Key Inputs:
- Balanced JSON file (from balance_proportion.py)
- Output directory for model and evaluation results
- `--grid-search`: Optional flag to perform parameter optimization

#### Key Outputs:
- Trained model file (`.joblib`)
- Performance evaluation plots:
  - Predicted vs. actual scatter plots
  - Feature importance bar chart
  - Site-specific performance plots
- Statistics report (`.txt`)

#### Example Command:
```bash
python train_regression_model.py \
  data/balanced/balanced_lichen_proportion.json \
  data/models/lichen \
  --grid-search
```

#### How It Works:
1. Automatically detects the target variable from the input data
2. Extracts features and target values
3. Optionally performs grid search to find optimal parameters
4. Performs cross-validation to evaluate model performance
5. Calculates site-specific metrics to assess geographic robustness
6. Creates visualizations and a comprehensive statistics report
7. Saves the final trained model for later use

## Optimizing Your Models

### Grid Search Parameters

The grid search explores different combinations of these parameters:
- `n_estimators`: Number of trees (100, 200, 300)
- `max_depth`: Maximum tree depth (None, 15, 30, 50)
- `min_samples_split`: Minimum samples to split a node (2, 5, 10)
- `min_samples_leaf`: Minimum samples at leaf nodes (1, 2, 4)
- `max_features`: Features considered for splits ('sqrt', 'log2', 0.8)

You can modify these in the `PARAM_GRID` dictionary in `train_regression_model.py`.

### Balancing Strategy

The balancing process significantly impacts model performance:
- **More bins** (e.g., 30-50) provide finer control but may result in fewer samples per bin
- **Higher quantile** (e.g., 0.7-0.8) keeps more samples but may result in less perfect balance
- **Lower quantile** (e.g., 0.3-0.4) creates more perfect balance but may discard more data

Adjust the `--bins` and `--quantile` parameters based on your dataset characteristics.

## Using the Trained Models

Once trained, models can be used to predict land cover proportions from Sentinel-2 imagery. The output models (`*_model.joblib`) contain:
- The trained RandomForest model
- Feature names required by the model
- Target variable name
- Model parameters

These can be loaded and applied to new Sentinel-2 data using the `joblib.load()` function.

## Preprocessing Sentinel-2 Data

### Creating Temporal Median Composites

Temporal median composites from Sentinel-2 L2A images help provide stable, representative spectral data by:

1. **Minimizing remaining artifacts**: Even in L2A products, residual cloud masks or orbit differences can occur
2. **Reducing temporal variability**: Creating more consistent spectral signatures across seasons
3. **Improving model robustness**: Providing more reliable features for regression

The `sentinel_median/sentinel_median.sh` script automates this process, generating median values across multiple acquisitions for each band and index while preserving georeferencing information.

## Tips for Best Results

1. **Use well-classified drone imagery** - The quality of your regression model depends directly on the quality of your classification map
2. **Include diverse sites** - Models trained on data from multiple sites tend to be more robust and generalizable
3. **Balance your data** - The distribution of proportion values greatly affects model performance
4. **Perform grid search** - Finding optimal parameters can significantly improve accuracy
5. **Check site-specific performance** - Look for systematic biases at specific sites
6. **Consider feature importance** - Focus on the most predictive bands and indices
   - Computes the pixel-wise median value across all acquisition dates
   - Stores the results in organized directories (separate for bands and indices)
   - Preserves the georeferencing information for each output

The resulting median images provide more reliable spectral information than any single acquisition and serve as ideal input features for the regression models.

## Tips for Best Results

1. **Use well-classified drone imagery** - The quality of your regression model depends directly on the quality of your classification map
2. **Include diverse sites** - Models trained on data from multiple sites tend to be more robust and generalizable
3. **Balance your data** - The distribution of proportion values greatly affects model performance
4. **Perform grid search** - Finding optimal parameters can significantly improve accuracy
5. **Check site-specific performance** - Look for systematic biases at specific sites
6. **Consider feature importance** - Focus on the most predictive bands and indices
