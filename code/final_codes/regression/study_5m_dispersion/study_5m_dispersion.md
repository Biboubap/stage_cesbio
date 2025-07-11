# 5m Resolution Dispersion Study Tools

This guide explains the tools and workflow for studying the benefits of 5m resolution Sentinel-2 data compared to standard 10m resolution, specifically focusing on land cover proportion prediction accuracy.

## Overview

The study examines whether higher resolution (5m) predictions, downsampled to 10m, provide better results than native 10m predictions. The workflow consists of three main components:

1. **Band Histogram Analysis** - Compare spectral characteristics at different resolutions
2. **Resolution Fusion** - Downsample 5m predictions to 10m 
3. **Performance Evaluation** - Compare results against ground truth

These tools allow quantitative assessment of whether the finer spatial detail in 5m data improves prediction accuracy when aggregated to coarser resolutions.

## Requirements

### Python Dependencies

- **Core Python**: Python 3.7 or newer
- **Data Handling**:
  - numpy (1.19.0+)
  - pandas (1.0.0+)
  - json
  - argparse
- **Geospatial Libraries**:
  - gdal/osgeo (3.0.0+)
- **Statistics & Machine Learning**:
  - scipy
  - sklearn
- **Visualization**:
  - matplotlib
  - tqdm

### External Dependencies
- GDAL command-line tools (for the fusion shell script)

## Tools Description

### 1. Band Histograms Analysis (`band_histograms.py`)

This script analyzes how spectral band values are distributed within different proportion ranges. It creates histograms showing the distribution of band values (e.g., B2) within specific bins of land cover proportion (e.g., lichen 0-5%, 5-10%, etc.).

#### Key Features:
- Side-by-side comparison of 5m and 10m resolutions
- Per-bin statistical summaries (mean, standard deviation)
- Normalized density histograms for fair comparison
- Summary plots showing how band values change with proportion

#### Usage:
```bash
python band_histograms.py \
    --balanced-5m path/to/balanced_5m_file.json \
    --balanced-10m path/to/balanced_10m_file.json \
    --output-dir path/to/output_directory \
    --proportion-type lichen_proportion \
    --band B2
```

#### Parameters:
- `--balanced-5m`: JSON file with 5m balanced samples
- `--balanced-10m`: JSON file with 10m balanced samples  
- `--output-dir`: Directory to save the output plots
- `--proportion-type`: Type of proportion to analyze (default: lichen_proportion)
- `--band`: Band to analyze (default: B2)
- `--n-bins`: Number of histogram bins (default: 30)

#### Workflow:
1. Loads balanced sample data for both resolutions
2. Divides the proportion range (0-1) into 5% bins
3. For each bin, creates histogram pairs showing the distribution of band values
4. Creates summary plots showing mean values with standard deviation across all bins
5. Saves results in organized directories

### 2. 5m to 10m Fusion (`5m_fusion.sh`)

This shell script takes regression predictions at 5m resolution and aggregates them to 10m resolution using averaging. This creates a "fusion" product that preserves the information content of the higher resolution while making it directly comparable to 10m predictions.

#### Key Features:
- Batch processing of multiple sites and categories
- Proper spatial alignment with 10m reference grids
- Preservation of georeferencing information
- Optimization of output GeoTIFF (compression, tiling)

#### Usage:
```bash
chmod +x 5m_fusion.sh
./5m_fusion.sh
```

#### Customization:
- Modify the site names array: `sites=("WAP12" "WAP23" "WAP32" "Belcher" "Chesnay" "Lamprey")`
- Modify the categories array: `categories=("lichen" "trough")`
- **Adjust file paths as needed for your directory structure**

#### Workflow:
1. Sets up output directory for fusion results
2. For each site and category:
   - Identifies a 10m reference band to establish target grid
   - Extracts spatial extent information from reference
   - Uses GDAL's `gdalwarp` with average resampling to fuse 4 pixels (5m) into 1 (10m)
   - Saves the result with proper georeferencing

### 3. Fusion Performance Evaluation (`evaluate_5m_fusion.py`)

This script is a versatile evaluation tool that compares prediction rasters against ground truth data. While originally designed for evaluating 5m fusion predictions, it can be used to evaluate any regression outputs (5m, 10m, or any resolution) as long as the prediction and truth rasters correspond spatially. The script calculates accuracy metrics and creates visualizations to quantify prediction quality.

**Important**: Despite its name focusing on "5m_fusion", this tool is general-purpose - you can use it to evaluate any prediction raster against corresponding ground truth data. The "fusion" parameter simply refers to the prediction being evaluated, and can be any compatible GeoTIFF.

#### Key Features:
- Spatial alignment of different resolution products
- Comprehensive statistical comparisons (R², RMSE, Pearson r)
- Percentage improvement calculations
- Scatter plots and hexbin density plots
- Masked comparison raster outputs for visual inspection

#### Usage:
```bash
python evaluate_5m_fusion.py \
    --truth path/to/proportions.tif \
    --fusion path/to/5m_fusion.tif \
    --native path/to/10m_prediction.tif \
    --category lichen \
    --site WAP23 \
    --output path/to/output_dir
```

#### Parameters:
- `--truth`: Ground truth proportion TIF from compute_proportion.py
- `--fusion`: 5m fusion prediction TIF (resampled to 10m)
- `--category`: Proportion category to evaluate (lichen, trough, or green)
- `--site`: Site name for labeling plots
- `--output`: Directory to save output plots and metrics
- `--native`: (Optional) Native 10m prediction TIF for comparison
- `--no-hexbin`: (Optional) Disable hexbin plot creation

#### Workflow:
1. Loads ground truth proportions, 5m fusion, and optionally 10m native predictions
2. Aligns all datasets spatially to ensure pixel-to-pixel comparison
3. Identifies valid overlapping pixels for fair comparison
4. Calculates accuracy metrics for 5m fusion vs. ground truth
5. If native 10m predictions are provided:
   - Calculates metrics for 10m predictions vs. ground truth
   - Computes percentage improvement from 10m to 5m fusion
6. Creates scatter plots and hexbin density plots
7. Saves masked comparison rasters for visual inspection
8. Outputs metric summary to CSV file

## Complete Workflow Example

To conduct a full 5m dispersion study:

1. **First, analyze band characteristics at different resolutions:**
   ```bash
   python band_histograms.py \
       --balanced-5m data/balanced_5m/balanced_lichen_proportion.json \
       --balanced-10m data/balanced_10m/balanced_lichen_proportion.json \
       --output-dir results/band_histograms \
       --proportion-type lichen_proportion \
       --band B2
   ```

2. **Next, fuse 5m predictions to 10m resolution:**
   ```bash
   ./5m_fusion.sh
   ```

3. **Finally, evaluate the performance compared to ground truth:**
   ```bash
   python evaluate_5m_fusion.py \
       --truth data/proportions/site_proportions.tif \
       --fusion results/fusion/site_5m_fusion_lichen_prediction.tif \
       --native results/native/site_10m_lichen_prediction.tif \
       --category lichen \
       --site SiteName \
       --output results/evaluation
   ```

## Understanding Results

### Band Histograms

The histograms reveal whether 5m resolution captures more detailed spectral information than 10m resolution:
- Narrower distributions suggest greater spectral purity
- Wider distributions suggest more mixed pixels
- The means and standard deviations help quantify the differences

### Fusion Performance

The evaluation metrics help quantify the benefits of using 5m data:
- **R² improvement**: Higher values indicate better explanation of proportion variance
- **RMSE improvement**: Lower values indicate more accurate predictions
- **Visual improvements**: Scatter plots reveal whether predictions cluster closer to the 1:1 line

## Tips and Best Practices

1. **Use matched datasets** - Ensure the 5m and 10m datasets cover exactly the same areas
2. **Check alignment** - Visual inspection of the comparison rasters helps identify any misregistration
3. **Consider multiple bands** - Different bands may show different levels of improvement at higher resolution
4. **Statistical significance** - Consider whether improvements are statistically significant, not just numerically different
5. **Multiple sites** - Test on several sites with different landscape characteristics to ensure robust conclusions
