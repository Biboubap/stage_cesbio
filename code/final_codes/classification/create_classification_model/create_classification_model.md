# Classification Model Creation Workflow

This directory contains tools for creating classification models through a three-step process: sample selection, sample manipulation, and model training. These tools help you build accurate land cover classification models from drone imagery.

## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - matplotlib
  - gdal/osgeo
  - scikit-learn
  - joblib
  - seaborn

## 1. Interactive Sample Selection

The `interactive_sample_selection.py` script provides a GUI tool for selecting and categorizing sample patches from drone imagery.

### Inputs

- **RGB Image**: GeoTIFF containing RGB drone imagery
- **DSM** (optional): Digital Surface Model GeoTIFF
- **Thermal Image** (optional): Thermal imagery GeoTIFF
- **Sample Parameters**:
  - `--sample-size`: Size of each sample patch in pixels (default: 32)
  - `--distance-start-horizontal/vertical`: Starting position in meters from top-left
  - `--blocks-x/y`: Number of 12×12 sample grids to display horizontally/vertically
  - `--classes`: Comma-separated list of class names

### Selection Tools

- **Space**: Change active category
- **Number Pad (1-9)**: Select/deselect 3×3 grid regions
- **Arrow Keys**: Toggle selections in half-window sections
  - Left: Toggle top half
  - Right: Toggle bottom half
  - Up: Toggle left half
  - Down: Toggle right half
- **0 Key**: Select/deselect all samples in current window
- **Q Key**: Stop selection and save
- **Mouse Click**: Toggle individual sample selection

### Example Command

```bash
python interactive_sample_selection.py \
    --rgb drone_image.tif \
    --dsm dsm.tif \
    --classes "Lichen,Green,Trough" \
    --output samples_dir \
    --sample-size 64 \
    --distance-start-horizontal 214 --distance-start-vertical 418 \
    --blocks-x 2 --blocks-y 2
```

## 2. Sample Population Manipulation

The `pop_interaction_utils.py` script provides utilities to manipulate sample populations (JSON files) created during selection.

### Key Functions

1. **Merge Samples**
   - Combine samples from multiple JSON files
   ```bash
   python pop_interaction_utils.py merge-samples \
       --input file1.json file2.json \
       --output merged.json
   ```

2. **Merge Samples from Directory**
   - Merge all JSON files in a directory
   ```bash
   python pop_interaction_utils.py merge-samples-from-dir \
       --input-dir samples_dir/ \
       --output merged.json
   ```

3. **Remove Category**
   - Filter out samples of a specific category
   ```bash
   python pop_interaction_utils.py remove-category \
       --input samples.json \
       --category "Lichen" \
       --output filtered.json
   ```

4. **Remove Specific Samples**
   - Remove samples at specific indices
   ```bash
   python pop_interaction_utils.py remove-samples \
       --input samples.json \
       --indices 0 5 10 \
       --output filtered.json
   ```

5. **Balance Categories**
   - Randomly sample to ensure each category has at most a specified number of samples
   ```bash
   python pop_interaction_utils.py balance-categories \
       --input samples.json \
       --max-samples 50 \
       --output balanced.json
   ```

## 3. Training Classification Model

The `train_classification_model.py` script trains a Random Forest classifier on the prepared sample data.

### Features

- Automatically detects available features in samples (RGB, DSM, etc.)
- Performs train/test split for evaluation (80/20)
- Generates detailed performance reports and visualizations
- Optional grid search for hyperparameter optimization

### Parameters

- **Input JSON**: Path to the prepared sample JSON file
- **Output Directory**: Where to save the model and reports
- **Grid Search**: Optional flag to enable hyperparameter optimization

### Example Commands

Basic training:
```bash
python train_classification_model.py \
    data/samples/merged_samples.json \
    output_directory
```

With grid search:
```bash
python train_classification_model.py \
    data/samples/merged_samples.json \
    output_directory \
    --grid-search
```

### Output

The training process generates:
- Trained model file (`.joblib`)
- Text report with model statistics and performance metrics
- Confusion matrix visualization
- Feature importance plot

## Complete Workflow Example

```bash
# 1. Select samples
python interactive_sample_selection.py \
    --rgb drone.tif \
    --dsm dsm.tif \
    --classes "Class1,Class2,Class3" \
    --output samples_dir \
    --sample-size 64 \
    --blocks-x 3 --blocks-y 3

# 2. Merge and balance samples
python pop_interaction_utils.py merge-samples-from-dir \
    --input-dir samples_dir/ \
    --output merged_samples.json

python pop_interaction_utils.py balance-categories \
    --input merged_samples.json \
    --max-samples 100 \
    --output balanced_samples.json

# 3. Train model with grid search
python train_classification_model.py \
    balanced_samples.json \
    model_output \
    --grid-search
```