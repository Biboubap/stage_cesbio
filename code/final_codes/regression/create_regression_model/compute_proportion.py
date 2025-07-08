"""
Compute Proportion

This script processes classification data to calculate proportions of merged classes
within Sentinel-2 pixels. It's the first step in the regression model workflow.

The script:
1. Takes a high-resolution drone classification map and a Sentinel-2 pixel grid
2. Computes the proportion of each class within each Sentinel-2 pixel
3. Merges classes into three main categories (Lichen, Green, Trough)
4. Extracts Sentinel-2 spectral bands and indices for each pixel
5. Generates a JSON file with pixel features and proportions for model training
6. Creates a TIF map of the calculated proportions

Usage:
  python compute_proportion.py --classification path/to/classification.tif 
                              --sentinel-band path/to/sentinel_band.tif 
                              --bands-dir path/to/bands_directory
                              --indices-dir path/to/indices_directory
                              --output-dir path/to/output_directory
                              [--keep-csv] [--site-name SITE_NAME]
"""
import os
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
from osgeo import gdal
import matplotlib.pyplot as plt

def pixel_to_geo(transform, px, py):
    """
    Convert pixel coordinates to geographic coordinates.
    
    Args:
        transform: GDAL geotransform array
        px, py: Pixel coordinates
        
    Returns:
        (x, y): Geographic coordinates in the projection of the raster
    """
    x = transform[0] + px * transform[1] + py * transform[2]
    y = transform[3] + px * transform[4] + py * transform[5]
    return x, y

def geo_to_pixel(transform, x, y):
    """
    Convert geographic coordinates to pixel coordinates.
    
    Args:
        transform: GDAL geotransform array
        x, y: Geographic coordinates
        
    Returns:
        (px, py): Pixel coordinates (rounded to nearest integer)
    """
    inv_det = 1 / (transform[1] * transform[5] - transform[2] * transform[4])
    px = inv_det * (transform[5] * (x - transform[0]) - transform[2] * (y - transform[3]))
    py = inv_det * (-transform[4] * (x - transform[0]) + transform[1] * (y - transform[3]))
    return int(round(px)), int(round(py))

def sentinel_to_drone_bounds(col_s, row_s, sentinel_path, drone_path):
    """
    Calculate the drone pixel boundaries that correspond to a Sentinel-2 pixel.
    
    This is a critical function that maps between the coarse Sentinel-2 grid
    and the high-resolution drone imagery by converting between coordinate systems.
    
    Args:
        col_s, row_s: Column and row indices of the Sentinel-2 pixel
        sentinel_path: Path to Sentinel-2 raster
        drone_path: Path to drone classification raster
        
    Returns:
        (xmin, xmax, ymin, ymax): Bounds of drone pixels within the Sentinel-2 pixel
    """
    ds_sentinel = gdal.Open(sentinel_path)
    ds_drone = gdal.Open(drone_path)
    gt_sentinel = ds_sentinel.GetGeoTransform()
    gt_drone = ds_drone.GetGeoTransform()
    
    # Get geo coordinates of the top-left corner of the Sentinel pixel
    x_min, y_max = pixel_to_geo(gt_sentinel, col_s, row_s)
    # Get geo coordinates of the bottom-right corner of the Sentinel pixel
    x_max, y_min = pixel_to_geo(gt_sentinel, col_s + 1, row_s + 1)
    
    # Convert to pixel coordinates in the drone image
    col_min, row_min = geo_to_pixel(gt_drone, x_min, y_min)  # bottom-left
    col_max, row_max = geo_to_pixel(gt_drone, x_max, y_max)  # top-right
    
    # Ensure xmin < xmax and ymin < ymax
    xmin = min(col_min, col_max)
    xmax = max(col_min, col_max)
    ymin = min(row_min, row_max)
    ymax = max(row_min, row_max)
    
    return xmin, xmax, ymin, ymax

def compute_class_proportions(classification_path, sentinel_path, output_csv, class_names, 
                              lichen_class_labels, trough_class_labels, green_class_labels):
    """
    Compute the proportion of merged classes within each Sentinel-2 pixel.
    
    This is the core function that calculates what percentage of each Sentinel-2 pixel
    is covered by each land cover class in the high-resolution classification map.
    
    The proportions are calculated for three merged class groups:
    - Lichen: Pure_Lichen + Degraded_Lichen
    - Green: Green vegetation
    - Trough: Sphagnum + Depression + Water
    
    Args:
        classification_path: Path to the high-resolution classification raster
        sentinel_path: Path to the Sentinel-2 raster (for pixel grid reference)
        output_csv: Path to save the CSV results
        class_names: Dictionary mapping class values to names
        lichen_class_labels: List of class names to merge into "Lichen"
        trough_class_labels: List of class names to merge into "Trough"
        green_class_labels: List of class names to merge into "Green"
        
    Returns:
        DataFrame with pixel class proportions for each Sentinel-2 pixel
    """
    # Load rasters
    ds_class = gdal.Open(classification_path)
    ds_sent = gdal.Open(sentinel_path)
    
    # Get dimensions
    cols_sent = ds_sent.RasterXSize
    rows_sent = ds_sent.RasterYSize
    
    # Load classification data
    class_data = ds_class.GetRasterBand(1).ReadAsArray()
    
    # Prepare results dictionary
    results = []
    
    # Process each Sentinel pixel
    print(f"Computing class proportions for {cols_sent}x{rows_sent} Sentinel pixels...")
    for row_s in tqdm(range(rows_sent)):
        for col_s in range(cols_sent):
            # Get drone pixel bounds for this Sentinel pixel
            xmin, xmax, ymin, ymax = sentinel_to_drone_bounds(col_s, row_s, sentinel_path, classification_path)
            
            # Skip if out of bounds
            if (xmin < 0 or ymin < 0 or 
                xmax >= class_data.shape[1] or 
                ymax >= class_data.shape[0]):
                continue
            
            # Extract classification data for this region
            subclass = class_data[ymin:ymax, xmin:xmax]
            
            # Skip if empty
            if subclass.size == 0:
                continue
            
            # Count total valid pixels (non-nodata) and nodata pixels
            valid_pixels = np.sum(subclass != 0)
            total_pixels = subclass.size
            no_data_pixels = total_pixels - valid_pixels
            
            if valid_pixels == 0:
                continue
                
            # Calculate proportions
            valid_proportion = valid_pixels / total_pixels if total_pixels > 0 else 0
            no_data_proportion = no_data_pixels / total_pixels if total_pixels > 0 else 1
            
            # Initialize row data
            result = {
                "col_s": col_s,
                "row_s": row_s,
                "valid_pixels": valid_pixels,
                "total_pixels": total_pixels,
                "valid_proportion": valid_proportion,
                "no_data_proportion": no_data_proportion
            }
            
            # Compute raw class proportions (but don't save them to result)
            raw_proportions = {}
            for class_val, class_name in class_names.items():
                count = np.sum(subclass == class_val)
                
                # For class 0 (No Data), calculate proportion of all pixels
                if class_val == 0:
                    proportion = count / total_pixels if total_pixels > 0 else 1
                else:
                    # For other classes, calculate proportion of valid pixels
                    proportion = count / valid_pixels if valid_pixels > 0 else 0
                    
                raw_proportions[class_name] = proportion
                # Don't add individual classes to result dictionary
            
            # Compute merged class proportions
            # 1. Lichen proportion (sum of Pure_Lichen and Degraded_Lichen)
            lichen_proportion = sum(raw_proportions.get(name, 0) for name in lichen_class_labels)
            result["lichen_proportion"] = lichen_proportion
            result["sqrt_lichen_proportion"] = np.sqrt(lichen_proportion) if lichen_proportion > 0 else 0
            
            # 2. Trough proportion (sum of Sphagnum, Depression, and Water)
            trough_proportion = sum(raw_proportions.get(name, 0) for name in trough_class_labels)
            result["trough_proportion"] = trough_proportion
            result["sqrt_trough_proportion"] = np.sqrt(trough_proportion) if trough_proportion > 0 else 0
            
            # 3. Green proportion
            green_proportion = sum(raw_proportions.get(name, 0) for name in green_class_labels)
            result["green_proportion"] = green_proportion
            result["sqrt_green_proportion"] = np.sqrt(green_proportion) if green_proportion > 0 else 0
            
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved class proportions to {output_csv} ({len(df)} pixels)")
    
    return df

def extract_sentinel_values(sentinel_bands_dir, sentinel_indices_dir, proportions_csv, output_csv):
    """
    Extract spectral values from Sentinel-2 bands and indices for each pixel.
    
    This function:
    1. Loads all Sentinel-2 band and index rasters from the specified directories
    2. For each pixel in the proportions CSV, extracts the corresponding values
    3. Creates a comprehensive dataset that combines class proportions with spectral data
    
    Args:
        sentinel_bands_dir: Directory containing Sentinel-2 band rasters
        sentinel_indices_dir: Directory containing Sentinel-2 indices rasters
        proportions_csv: Path to the CSV with class proportions or DataFrame
        output_csv: Path to save the merged CSV with Sentinel values
        
    Returns:
        DataFrame with extracted Sentinel values and class proportions
    """
    # Check if proportions_csv is a DataFrame or path
    if isinstance(proportions_csv, pd.DataFrame):
        df = proportions_csv
    else:
        df = pd.read_csv(proportions_csv)
    
    # Find all band and index files
    band_files = [f for f in os.listdir(sentinel_bands_dir) if f.endswith('.tif')]
    index_files = [f for f in os.listdir(sentinel_indices_dir) if f.endswith('.tif')]
    
    print(f"Found {len(band_files)} band files and {len(index_files)} index files")
    
    # Process each band file
    for band_file in band_files:
        band_path = os.path.join(sentinel_bands_dir, band_file)
        band_name = os.path.splitext(band_file)[0].replace('mediane_clipped_STACK_2023_Band', '').replace('_WAP32', '').replace('_WAP23', '')
        
        # Open the raster
        ds = gdal.Open(band_path)
        if ds is None:
            print(f"Warning: Could not open {band_path}")
            continue
            
        # Read the raster data
        band_data = ds.GetRasterBand(1).ReadAsArray()
        
        # Extract values for each pixel in the dataframe
        values = []
        for _, row in df.iterrows():
            col_s = int(row["col_s"])
            row_s = int(row["row_s"])
            # Get value at this pixel
            if 0 <= row_s < band_data.shape[0] and 0 <= col_s < band_data.shape[1]:
                val = band_data[row_s, col_s]
            else:
                val = None
            values.append(val)
        
        # Add to dataframe
        df[f"band_{band_name}"] = values
        
        # Close the dataset
        ds = None
    
    # Process each index file
    for index_file in index_files:
        index_path = os.path.join(sentinel_indices_dir, index_file)
        index_name = os.path.splitext(index_file)[0].replace('mediane_clipped_', '').replace('_WAP32', '').replace('_WAP23', '')
        
        # Open the raster
        ds = gdal.Open(index_path)
        if ds is None:
            print(f"Warning: Could not open {index_path}")
            continue
            
        # Read the raster data
        index_data = ds.GetRasterBand(1).ReadAsArray()
        
        # Extract values for each pixel in the dataframe
        values = []
        for _, row in df.iterrows():
            col_s = int(row["col_s"])
            row_s = int(row["row_s"])
            # Get value at this pixel
            if 0 <= row_s < index_data.shape[0] and 0 <= col_s < index_data.shape[1]:
                val = index_data[row_s, col_s]
            else:
                val = None
            values.append(val)
        
        # Add to dataframe
        df[f"index_{index_name}"] = values
        
        # Close the dataset
        ds = None
    
    # Save merged dataframe
    df.to_csv(output_csv, index=False)
    print(f"Saved merged data to {output_csv}")
    
    return df

def filter_by_valid_proportion(input_csv, output_csv, min_valid_proportion=0.95):
    """
    Filter the dataset to keep only pixels with sufficient valid (non-NoData) coverage.
    
    This removes pixels that are partially outside the classified area or contain
    too many NoData values, which could bias the regression model.
    
    Args:
        input_csv: Path to the input CSV or DataFrame
        output_csv: Path to save the filtered CSV
        min_valid_proportion: Minimum proportion of valid pixels required (default: 0.95)
        
    Returns:
        DataFrame with filtered pixels that meet the validity threshold
    """
    # Check if input_csv is a DataFrame or path
    if isinstance(input_csv, pd.DataFrame):
        df = input_csv
    else:
        df = pd.read_csv(input_csv)
        
    total_rows = len(df)
    
    # Filter by valid proportion directly
    df_filtered = df[df['no_data_proportion'] <= (1 - min_valid_proportion)]
    kept_rows = len(df_filtered)
    
    print(f"Filtered from {total_rows} to {kept_rows} pixels " +
          f"({kept_rows/total_rows*100:.1f}%) having at least {min_valid_proportion*100:.0f}% valid data")
    
    # Save filtered data
    df_filtered.to_csv(output_csv, index=False)
    
    return df_filtered

def plot_class_proportions(csv_path, output_dir, purcent_exclusion=0.05):
    """
    Create histograms showing the distribution of class proportions.
    
    These visualizations help understand the distribution of different land cover types
    in the dataset and identify potential imbalances.
    
    Args:
        csv_path: Path to the CSV file with class proportions or DataFrame
        output_dir: Directory to save the plots
        purcent_exclusion: Threshold below which samples are excluded from visualization
    """
    # Check if csv_path is a DataFrame or path
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path
    else:
        df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot merged proportions
    merged_props = ['lichen_proportion', 'green_proportion', 'trough_proportion']
    sqrt_merged_props = ['sqrt_lichen_proportion', 'sqrt_green_proportion', 'sqrt_trough_proportion']
    
    # Plot merged class proportions
    plt.figure(figsize=(15, 5))
    n_cols = len(merged_props)
    
    for i, col in enumerate(merged_props):
        plt.subplot(1, n_cols, i+1)
        low_count = (df[col] <= purcent_exclusion).sum()
        filtered_data = df[df[col] > purcent_exclusion][col]
        filtered_data.hist(bins=20)
        plt.title(f"{col}\n({low_count} samples ≤ {purcent_exclusion*100:.0f}% excluded)")
        plt.xlabel('Proportion')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merged_classes_histograms.png'))
    plt.close()
    
    # Plot sqrt merged class proportions
    plt.figure(figsize=(15, 5))
    
    for i, col in enumerate(sqrt_merged_props):
        plt.subplot(1, n_cols, i+1)
        low_count = (df[col] <= purcent_exclusion).sum()
        filtered_data = df[df[col] > purcent_exclusion][col]
        filtered_data.hist(bins=20)
        plt.title(f"{col}\n({low_count} samples ≤ {purcent_exclusion*100:.0f}% excluded)")
        plt.xlabel('Sqrt Proportion')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sqrt_merged_classes_histograms.png'))
    plt.close()
    
    print(f"Class proportion histograms saved in {output_dir}")

def create_proportion_tif(filtered_csv, output_tif, sentinel_path):
    """
    Create a multi-band GeoTIFF file showing the spatial distribution of class proportions.
    
    This creates a visualization of the calculated proportions that can be displayed
    in GIS software, with each band representing a different class proportion.
    
    Args:
        filtered_csv: Path to the filtered CSV with proportions or DataFrame
        output_tif: Path to save the output multi-band TIFF
        sentinel_path: Path to the Sentinel reference image for georeferencing
        
    Returns:
        Path to the created TIF file
    """
    print(f"Creating proportion TIF from filtered CSV data...")
    
    # Check if filtered_csv is a DataFrame or path
    if isinstance(filtered_csv, pd.DataFrame):
        df = filtered_csv
    else:
        df = pd.read_csv(filtered_csv)
    
    # Open sentinel reference for dimensions and georeference
    sentinel_ds = gdal.Open(sentinel_path)
    if sentinel_ds is None:
        raise ValueError(f"Could not open sentinel reference: {sentinel_path}")
    
    # Get dimensions and georeference
    width = sentinel_ds.RasterXSize
    height = sentinel_ds.RasterYSize
    geo_transform = sentinel_ds.GetGeoTransform()
    projection = sentinel_ds.GetProjection()
    
    # Define proportion columns to include in the output TIFF
    proportion_columns = ["lichen_proportion", "green_proportion", "trough_proportion"]
    
    # Create the output raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, width, height, len(proportion_columns), gdal.GDT_Int16,
                          options=['COMPRESS=DEFLATE', 'TILED=YES'])
    
    # Set projection and geotransform
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Create empty arrays initialized with NoData (-1)
    proportion_arrays = {}
    for col in proportion_columns:
        proportion_arrays[col] = np.full((height, width), -1, dtype=np.int16)
    
    # Fill arrays from the CSV data
    for _, row in df.iterrows():
        x = int(row['col_s'])
        y = int(row['row_s'])
        
        # Make sure coordinates are valid
        if 0 <= y < height and 0 <= x < width:
            for col in proportion_columns:
                proportion = row.get(col, 0)
                proportion_arrays[col][y, x] = int(proportion * 100)  # Scale to percentage (0-100)
    
    # Write arrays to bands
    for i, col in enumerate(proportion_columns):
        band = out_ds.GetRasterBand(i + 1)
        band.WriteArray(proportion_arrays[col])
        band.SetDescription(col)
        band.SetNoDataValue(-1)
        band.FlushCache()
    
    # Add band descriptions as metadata
    out_ds.SetMetadata({f"BAND_{i+1}_NAME": name for i, name in enumerate(proportion_columns, start=1)})
    
    # Close the dataset
    out_ds = None
    print(f"Multi-band proportion TIFF created at {output_tif}")
    
    return output_tif

def merge_csv_files(features_csv, proportions_csv, output_csv, output_json):
    """
    Merge features CSV and proportions CSV into a single CSV file and a JSON file.
    
    The resulting JSON file is the primary output of the compute_proportion script
    and will be used for training regression models.
    
    Args:
        features_csv: Path to CSV with Sentinel features or DataFrame
        proportions_csv: Path to CSV with class proportions or DataFrame
        output_csv: Path to save the merged CSV
        output_json: Path to save the JSON file with sentinel features and proportions
        
    Returns:
        tuple: (path to merged CSV, path to JSON file)
    """
    print(f"Merging CSV files...")
    
    # Check if inputs are DataFrames or paths
    if isinstance(features_csv, pd.DataFrame):
        features_df = features_csv
    else:
        features_df = pd.read_csv(features_csv)
        
    if isinstance(proportions_csv, pd.DataFrame):
        proportions_df = proportions_csv
    else:
        proportions_df = pd.read_csv(proportions_csv)
    
    # Merge on col_s and row_s (pixel position)
    merged_df = pd.merge(
        features_df, proportions_df,
        on=['col_s', 'row_s'],
        how='inner',
        suffixes=('', '_dup')
    )
    
    # Remove duplicate columns (those ending with _dup)
    cols_to_drop = [col for col in merged_df.columns if col.endswith('_dup')]
    merged_df = merged_df.drop(columns=cols_to_drop)
    
    # Save the merged dataframe to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged data saved to {output_csv} ({len(merged_df)} rows)")
    
    # Create JSON with only sentinel features and merged class proportions
    json_data = []
    
    # Define the columns to keep in the JSON
    feature_cols = [col for col in merged_df.columns if col.startswith('band_') or col.startswith('index_')]
    proportion_cols = ['lichen_proportion', 'green_proportion', 'trough_proportion',
                      'sqrt_lichen_proportion', 'sqrt_green_proportion', 'sqrt_trough_proportion']
    
    # For each row in the merged DataFrame
    for _, row in merged_df.iterrows():
        # Create a pixel entry with features and proportions
        pixel_data = {}
        
        # Add Sentinel features
        features = {col: float(row[col]) if not pd.isna(row[col]) else None for col in feature_cols}
        pixel_data["features"] = features
        
        # Add merged class proportions
        proportions = {col: float(row[col]) if not pd.isna(row[col]) else 0.0 for col in proportion_cols}
        pixel_data["proportions"] = proportions
        
        json_data.append(pixel_data)
    
    # Save the JSON file
    with open(output_json, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Sentinel features and proportions saved to {output_json} ({len(json_data)} pixels)")
    
    return output_csv, output_json

def main():
    """
    Main function to process classification data and create proportion outputs.
    
    Workflow:
    1. Compute class proportions for each Sentinel pixel
    2. Filter out pixels with insufficient valid data
    3. Extract Sentinel spectral values for each pixel
    4. Create visualization plots
    5. Generate proportion TIF map
    6. Merge features and proportions into CSV and JSON files
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compute class proportions for Sentinel-2 pixels.")
    parser.add_argument('--classification', required=True, help="Path to the classification raster")
    parser.add_argument('--sentinel-band', required=True, help="Path to the Sentinel-2 band for projection reference")
    parser.add_argument('--bands-dir', required=True, help="Directory containing Sentinel-2 band rasters")
    parser.add_argument('--indices-dir', required=True, help="Directory containing Sentinel-2 indices rasters")
    parser.add_argument('--output-dir', required=True, help="Directory to save output files")
    parser.add_argument('--keep-csv', action='store_true', help="Keep intermediate CSV files (default: delete them)")
    parser.add_argument('--site-name', default="Site", help="Name of the site (used in output filenames)")
    
    args = parser.parse_args()
    
    # Set paths from arguments
    classification_path = args.classification
    sentinel_path = args.sentinel_band
    bands_dir = args.bands_dir
    indices_dir = args.indices_dir
    output_dir = args.output_dir
    site_name = args.site_name
    keep_csv = args.keep_csv
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_base = f"{output_dir}/proportions_{site_name}"

    # Class names and groupings
    class_labels = {
        1: "Pure_Lichen",
        2: "Degraded_Lichen", 
        3: "Green",
        4: "Sphagnum",
        5: "Depression", 
        6: "Water",
        0: "No_Data"
    }
    
    # Define the specific class groupings
    trough_class_labels = ["Sphagnum", "Depression", "Water"]
    lichen_class_labels = ["Pure_Lichen", "Degraded_Lichen"]
    green_class_labels = ["Green"]
    
    # Step 1: Compute class proportions
    print("Step 1: Computing class proportions...")
    proportions_csv = f"{output_base}_all.csv"
    proportions_df = compute_class_proportions(
        classification_path=classification_path,
        sentinel_path=sentinel_path,
        output_csv=proportions_csv,
        class_names=class_labels,
        lichen_class_labels=lichen_class_labels,
        trough_class_labels=trough_class_labels,
        green_class_labels=green_class_labels
    )
    
    # Step 2: Filter by valid proportion
    print("Step 2: Filtering by valid proportion...")
    filtered_csv = f"{output_base}_filtered.csv"
    filtered_df = filter_by_valid_proportion(
        input_csv=proportions_csv,
        output_csv=filtered_csv,
        min_valid_proportion=0.95  # Keep pixels with at least 95% valid data
    )
    
    # Step 3: Extract Sentinel band and index values
    print("Step 3: Extracting Sentinel values...")
    features_csv = f"{output_base}_features.csv"
    features_df = extract_sentinel_values(
        sentinel_bands_dir=bands_dir,
        sentinel_indices_dir=indices_dir,
        proportions_csv=filtered_csv,
        output_csv=features_csv
    )
    
    # Step 4: Create plots
    print("Step 4: Creating plots...")
    plots_dir = os.path.join(output_dir, "plots")
    plot_class_proportions(
        csv_path=filtered_csv,
        output_dir=plots_dir,
        purcent_exclusion=0.00
    )
    
    # Step 5: Create proportion TIF from filtered data
    print("Step 5: Creating proportion TIF...")
    output_tif = f"{output_base}.tif"
    create_proportion_tif(
        filtered_csv=filtered_csv, 
        output_tif=output_tif,
        sentinel_path=sentinel_path
    )
    
    # Step 6: Merge features CSV with proportions CSV and create JSON
    print("Step 6: Merging feature and proportion data...")
    merged_csv = f"{output_base}_merged.csv"
    output_json = f"{output_base}.json"
    merged_csv, output_json = merge_csv_files(
        features_csv=features_csv,
        proportions_csv=filtered_csv,
        output_csv=merged_csv,
        output_json=output_json
    )
    
    # Step 7: Delete CSV files if not keeping them
    if not keep_csv:
        print("Cleaning up CSV files...")
        csv_files = [proportions_csv, filtered_csv, features_csv, merged_csv]
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                os.remove(csv_file)
                print(f"  Removed {csv_file}")
    
    print(f"All processing completed!")
    print(f"Results saved in {output_dir}")
    print(f"- Proportion TIF: {output_tif}")
    print(f"- JSON data: {output_json}")
    if keep_csv:
        print(f"- Filtered proportions: {filtered_csv}")
        print(f"- Merged data: {merged_csv}")

if __name__ == "__main__":
    main()
"""
# Command line examples:

Linux example:

 python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/compute_proportion.py \
    --classification /media/lcousin/FASTBOYSLIM/Loris/final_data/drone_classif_peatcut/Belcher_classif_cut.tif \
    --sentinel-band /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Belcher_10m/mediane_bands/mediane_STACK_2023_BandB4_Belcher_deflate.tif \
    --bands-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Belcher_10m/mediane_bands \
    --indices-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Belcher_10m/mediane_indices \
    --output-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/Belcher \
    --site-name Belcher     

 python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/compute_proportion.py \
    --classification /media/lcousin/FASTBOYSLIM/Loris/final_data/drone_classif_peatcut/Chesnay_classif_cut.tif \
    --sentinel-band /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Chesnay_10m/mediane_bands/mediane_STACK_2023_BandB4_Chesnay_deflate.tif \
    --bands-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Chesnay_10m/mediane_bands \
    --indices-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Chesnay_10m/mediane_indices \
    --output-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/Chesnay \
    --site-name Chesnay       

 python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/compute_proportion.py \
    --classification /media/lcousin/FASTBOYSLIM/Loris/final_data/drone_classif_peatcut/wap23_classif_cut.tif \
    --sentinel-band /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/WAP23_10m/mediane_bands/mediane_STACK_2023_BandB4_WAP23_deflate.tif \
    --bands-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/WAP23_10m/mediane_bands \
    --indices-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/WAP23_10m/mediane_indices \
    --output-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/WAP23_10m \
    --site-name WAP23

 python /home/lcousin/stage_cesbio/code/final_codes/regression/create_regression_model/compute_proportion.py \
    --classification /media/lcousin/FASTBOYSLIM/Loris/final_data/drone_classif_peatcut/Lamprey_classif_cut.tif \
    --sentinel-band /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Lamprey_10m/mediane_bands/mediane_STACK_2023_BandB4_Lamprey_deflate.tif \
    --bands-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Lamprey_10m/mediane_bands \
    --indices-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/sentinel_2/Lamprey_10m/mediane_indices \
    --output-dir /media/lcousin/FASTBOYSLIM/Loris/final_data/regression_population/site_proportion/Lamprey \
    --site-name Lamprey
"""

