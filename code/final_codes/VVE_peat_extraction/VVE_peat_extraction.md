# Peatland Extraction Tool for Sentinel-2 Imagery

This document explains the `extract_peatland_VVE.py` script, a specialized tool for creating binary masks of potential peatland areas from Sentinel-2 satellite imagery.

## Overview

The script applies spectral thresholds to Sentinel-2 bands to identify peatland areas. It processes very large rasters (~9GB per band) efficiently through block-based processing, making it suitable for large-scale regional analysis.

## Key Functions

### `process_rasters`
The core function that orchestrates the entire process. It:
1. Verifies raster consistency
2. Processes data in overlapping blocks
3. Applies spectral thresholds
4. Performs post-processing
5. Creates the output mask

### `expand_mask_boundaries`
Expands the mask boundaries by dilating the binary mask, used to include adjacent pixels when specified.

### `fill_small_holes`
Identifies and fills small holes (areas of 0s surrounded by 1s) in the mask up to a specified maximum size.

### `verify_rasters_consistency`
Ensures all input rasters have matching dimensions and compatible projections.

## Processing Workflow

1. **Preparation**: Validates input files and creates the output structure
2. **Block Processing**: 
   - Divides the large rasters into manageable blocks with overlap
   - For each block:
     - Reads data from input bands
     - Applies spectral thresholds to create binary mask
     - Performs optional boundary expansion
     - Fills small holes based on size threshold
   - Writes the processed block to the output file
3. **Finalization**: Computes statistics and finalizes the output GeoTIFF

## Key Parameters

### Spectral Thresholds
- **Infrared (B8)**: `IR_MIN = 2000, IR_MAX = 3000`
- **Red (B4)**: `RED_MIN = 750, RED_MAX = 1500`
- **Green (B3)**: `GREEN_MIN = 600, GREEN_MAX = 1500`

### Post-processing Parameters
- `--distance-inclusion`: Number of pixels to expand the mask boundaries (default: 0)
- `--groupe-inclusion`: Maximum size of holes to fill (default: 10)

### Processing Parameters
- `--block-size`: Size of processing blocks in pixels (default: 2048)

## Requirements

### Python Dependencies
- **Core Python**: Python 3.7 or newer
- **Numerical Processing**:
  - numpy
  - scipy (for morphological operations)
- **Geospatial Libraries**:
  - osgeo/gdal (for raster processing)
- **Progress Monitoring**:
  - tqdm (for progress bars)

## Usage Example

```bash
python extract_peatland_VVE.py \
    --ir path/to/infrared.tif \
    --red path/to/red.tif \
    --green path/to/green.tif \
    --output path/to/output_mask.tif \
    --distance-inclusion 0 \
    --groupe-inclusion 15
```

## Performance Considerations

- The script uses block-based processing to handle very large rasters efficiently
- Block size can be adjusted based on available memory
- Processing includes adequate overlap between blocks to ensure correct boundary processing
- For very large areas (~9GB per band), processing may take several hours
- The output is optimized with DEFLATE compression and tiling for efficient access
