# Peatland Monitoring Tools: Project Overview

This document provides an overview of the comprehensive peatland monitoring toolkit, explaining the different components and their relationships. The project includes tools for classification, regression modeling, peatland extraction, and resolution analysis.

## Project Components

The toolkit consists of six main components:

1. **Classification Model Creation** - Tools for creating land cover classification models from drone imagery
2. **Classification Model Application** - System for applying trained models to large drone imagery areas
3. **Regression Model Creation** - Pipeline for building models to predict land cover proportions from Sentinel-2 data
4. **Regression Model Application** - Tools to apply trained models to new Sentinel-2 imagery
5. **Resolution Analysis** - Tools to study the impact of spatial resolution on prediction quality
6. **Peatland Extraction** - Specialized tool for identifying potential peatland areas in large-scale imagery

## Workflow Relationships

The components form a cohesive workflow for peatland analysis:

1. **Classification Model Creation** produces trained land cover models from drone imagery samples
2. **Classification Model Application** applies these models to classify entire drone imagery datasets
3. The resulting classification maps serve as input to **Regression Model Creation** which learns to predict land cover proportions from satellite data
4. **Regression Model Application** applies these models to new areas without drone coverage
5. **Resolution Analysis** evaluates how different spatial resolutions affect model performance
6. **Peatland Extraction** provides a quick method to identify potential peatland areas for further analysis

## Component Details

### 1. Classification Model Creation

Location: `/code/final_codes/classification/create_classification_model/`

This component provides tools for creating land cover classification models from drone imagery through a three-step process:

1. **Interactive Sample Selection** - GUI tool to select and categorize training samples
2. **Sample Population Manipulation** - Utilities to merge, filter, and balance sample collections
3. **Model Training** - Random Forest classifier training with evaluation metrics

[Detailed Documentation](./classification/create_classification_model/create_classification_model.md)

### 2. Classification Model Application

Location: `/code/final_codes/classification/apply_classification_model/`

This component implements a parallel processing system to apply classification models to very large drone imagery:

1. **Block Processing** - Divides large images into manageable, overlapping blocks
2. **Parallel Execution** - Processes blocks concurrently using Dask
3. **Feature Extraction** - Calculates statistics for patches and their neighborhoods
4. **Model Application** - Applies two models in combination to classify land cover types

**⚠️ Important Warning**: The classification application system is specifically designed for a pipeline of two pre-existing models working in combination. If you want to use models created with the Classification Model Creation tools, you will need to modify the code or create an alternative implementation adapted to your specific models.

[Detailed Documentation](./classification/apply_classification_model/apply_classification_model.md)

### 3. Regression Model Creation

Location: `/code/final_codes/regression/create_regression_model/`

This component builds regression models that predict land cover proportions from Sentinel-2 satellite imagery:

1. **Computing Proportions** - Calculate class proportions within Sentinel-2 pixels using classified drone imagery
2. **Merging Data** - Combine proportion data from multiple sites
3. **Balancing Data** - Ensure even representation across proportion values
4. **Training Models** - Train and evaluate Random Forest regression models

[Detailed Documentation](./regression/create_regression_model/create_regression_model.md)

### 4. Regression Model Application

Location: `/code/final_codes/regression/apply_regression_model/`

This component applies trained regression models to new Sentinel-2 imagery:

1. **Feature Extraction** - Extracts required bands and indices from Sentinel-2 imagery
2. **Model Application** - Applies regression models to predict land cover proportions
3. **Output Generation** - Creates GeoTIFF maps of predicted proportions

[Detailed Documentation](./regression/apply_regression_model/apply_regression_model.md)

### 5. Resolution Analysis

Location: `/code/final_codes/regression/study_5m_dispersion/`

This component studies the benefits of higher resolution (5m) Sentinel-2 data versus standard 10m resolution:

1. **Band Histogram Analysis** - Compares spectral characteristics at different resolutions
2. **Resolution Fusion** - Downsamples 5m predictions to 10m for comparison
3. **Performance Evaluation** - Quantifies prediction quality improvements

[Detailed Documentation](./regression/study_5m_dispersion/study_5m_dispersion.md)

### 6. Peatland Extraction

Location: `/code/final_codes/VVE_peat_extraction/`

This specialized tool creates binary masks of potential peatland areas from Sentinel-2 bands:

1. **Block Processing** - Efficiently handles very large rasters (~9GB per band)
2. **Spectral Thresholding** - Applies spectral criteria to identify peatland areas
3. **Post-processing** - Refines results with morphological operations

[Detailed Documentation](./VVE_peat_extraction/VVE_peat_extraction.md)

## Python Requirements

The complete toolkit requires the following Python packages:

### Core Dependencies
- **Python**: 3.7 or newer
- **NumPy**: 1.19.0 or newer - Numerical processing foundation
- **Pandas**: 1.0.0 or newer - Data handling and manipulation
- **Matplotlib**: 3.3.0 or newer - Visualization and plotting
- **SciPy**: 1.5.0 or newer - Scientific computing and image processing

### Geospatial Processing
- **GDAL/osgeo**: 3.0.0 or newer - Raster and vector data manipulation
- **Rasterio**: 1.1.0 or newer - Raster data access (used in some modules)

### Machine Learning
- **scikit-learn**: 0.24.0 or newer - Machine learning algorithms and metrics
- **joblib**: 1.0.0 or newer - Model serialization and parallel processing

### Utility & Visualization
- **tqdm**: 4.45.0 or newer - Progress bars for long-running operations
- **Seaborn**: 0.11.0 or newer - Advanced statistical visualizations

### Parallel Processing
- **Dask**: 2021.3.0 or newer - Parallel and distributed computing
- **Distributed**: 2021.3.0 or newer - Dask distributed scheduler

### Environment Setup

You can create a suitable environment using conda:

```bash
conda create -n peatland-tools python=3.8
conda activate peatland-tools
conda install numpy pandas matplotlib scipy scikit-learn joblib tqdm seaborn
conda install -c conda-forge gdal rasterio dask distributed
```

Or using pip (with GDAL installed separately):

```bash
pip install numpy pandas matplotlib scipy scikit-learn joblib tqdm seaborn dask distributed
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/peatland-monitoring-tools.git
   cd peatland-monitoring-tools
   ```

2. Set up the Python environment as described above

3. Follow the workflow documentation for your specific use case:
   - For classification model creation, start with the [Classification Model Creation](./classification/create_classification_model/create_classification_model.md) guide
   - For applying classification models to large areas, see the [Classification Model Application](./classification/apply_classification_model/apply_classification_model.md) guide
   - For regression modeling, follow the [Regression Model Creation](./regression/create_regression_model/create_regression_model.md) guide
   - For applying regression models to new areas, see the [Regression Model Application](./regression/apply_regression_model/apply_regression_model.md) documentation
   - For resolution analysis, check the [Resolution Analysis](./regression/study_5m_dispersion/study_5m_dispersion.md) guide
   - For peatland identification, refer to the [Peatland Extraction](./VVE_peat_extraction/VVE_peat_extraction.md) documentation
