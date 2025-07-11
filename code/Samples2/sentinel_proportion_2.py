""""à partir des codes dans le dossier Sample2, et du code interaction_sentinel_drone, crée-moi sentinel_proportion.py dans le dossier Sample 2 qui reprend les fonctionnalités actuelles de interaction_sentinel_drone mais avec la classe sample2 et le raster manager.

Je souhaite donc pouvoir : 
te fournir un raster sentinel2 supperposé à une classification RF que tu chargeras et feras les transformées nécessaires pour compute les proportion de chaque classe et me sauvegarder un csv contenant pixel par pixel les données.
Je te transmetterai des tuiles sentinel """

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from osgeo import gdal
from rasters_manager import RastersManager
from sample2 import Sample2
from samples_set2 import SamplesSet2

def pixel_to_geo(transform, px, py):
    """
    Convert pixel coordinates (col, row) to geo coordinates (x, y) of the top-left corner
    """
    x = transform[0] + px * transform[1] + py * transform[2]
    y = transform[3] + px * transform[4] + py * transform[5]
    return x, y

def geo_to_pixel(transform, x, y):
    """
    Convert geo coordinates (x, y) to pixel coordinates (col, row)
    """
    inv_det = 1 / (transform[1] * transform[5] - transform[2] * transform[4])
    px = inv_det * (transform[5] * (x - transform[0]) - transform[2] * (y - transform[3]))
    py = inv_det * (-transform[4] * (x - transform[0]) + transform[1] * (y - transform[3]))
    return int(round(px)), int(round(py))

def sentinel_to_drone_bounds(col_s, row_s, sentinel_path, drone_path):
    """
    For a Sentinel pixel (col_s, row_s), return the bounds xmin, xmax, ymin, ymax
    of drone pixels covered by this Sentinel pixel.
    
    Args:
        col_s, row_s: Column and row indices of the Sentinel pixel
        sentinel_path: Path to the Sentinel raster
        drone_path: Path to the drone raster
        
    Returns:
        xmin, xmax, ymin, ymax: Bounds of drone pixels
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

def compute_class_proportions(classification_path, sentinel_path, output_csv, class_names=None, through_class_names=None):
    """
    Compute the proportion of each class within each Sentinel-2 pixel.
    
    Args:
        classification_path: Path to the classification raster (from RF model)
        sentinel_path: Path to the Sentinel-2 raster
        output_csv: Path to save the CSV results
        class_names: Dictionary mapping class values to names (e.g., {1: "chicoutai", 2: "dry_depression"})
                    If None, class values will be used as names
    """
    # Default class mapping if not provided
    if class_names is None:
        class_names = {
            1: "chicoutai",
            2: "dry_depression", 
            3: "green_depression",
            4: "lichen",
            5: "sphaignes",
            6: "watered_depression",
            7: "black_depression",
            0: "none",
        }
    if through_class_names is None:
        through_class_names = ["sphaignes", "dry_depression", "black_depression", "watered_depression"]
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
            
            # Count total valid pixels (non-nodata)
            valid_pixels = np.sum((subclass != 0))
            if valid_pixels == 0:
                continue
            
            # Compute proportions for each class
            class_proportions = {}
            for class_val, class_name in class_names.items():
                if class_val == 0:  # Skip nodata
                    continue
                count = np.sum(subclass == class_val)
                prop = count / valid_pixels if valid_pixels > 0 else 0
                class_proportions[class_name] = prop
                class_proportions[f"sqrt_{class_name}"] = np.sqrt(prop) if prop > 0 else 0

            # through_proportion and sqrt_through_proportion (sum of through_class_names)
            through_proportion = sum(class_proportions.get(name, 0) for name in through_class_names)
            class_proportions["through_proportion"] = through_proportion
            class_proportions["sqrt_through_proportion"] = np.sqrt(through_proportion) if through_proportion > 0 else 0

            # all_lichen and sqrt_all_lichen (sum of Pure_Lichen and Degraded_Lichen)
            all_lichen = class_proportions.get("Pure_Lichen", 0) + class_proportions.get("Degraded_Lichen", 0)
            class_proportions["all_lichen"] = all_lichen
            class_proportions["sqrt_all_lichen"] = np.sqrt(all_lichen) if all_lichen > 0 else 0

            result = {
                "col_s": col_s,
                "row_s": row_s,
                "valid_pixels": valid_pixels,
                "total_pixels": subclass.size,
                "valid_proportion": valid_pixels / subclass.size
            }
            result.update(class_proportions)
            results.append(result)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved class proportions to {output_csv} ({len(df)} pixels)")
    
    return df

def extract_sentinel_values(sentinel_bands_dir, sentinel_indices_dir, proportions_csv, output_csv):
    """
    Extract values from multiple Sentinel-2 bands and indices for each pixel in the proportions CSV.
    
    Args:
        sentinel_bands_dir: Directory containing Sentinel-2 band rasters
        sentinel_indices_dir: Directory containing Sentinel-2 indices rasters
        proportions_csv: Path to the CSV with class proportions
        output_csv: Path to save the merged CSV with Sentinel values
    """
    # Load proportions data
    df = pd.read_csv(proportions_csv)
    
    # Find all band and index files
    band_files = [f for f in os.listdir(sentinel_bands_dir) if f.endswith('.tif')]
    index_files = [f for f in os.listdir(sentinel_indices_dir) if f.endswith('.tif')]
    
    print(f"Found {len(band_files)} band files and {len(index_files)} index files")
    
    # Process each band file
    for band_file in band_files:
        band_path = os.path.join(sentinel_bands_dir, band_file)
        band_name = os.path.splitext(band_file)[0].replace('mediane_clipped_STACK_2023_Band', '').replace('_WAP', '')
        
        # Open the raster directly with GDAL (single-band rasters)
        ds = gdal.Open(band_path)
        if ds is None:
            print(f"Warning: Could not open {band_path}")
            continue
            
        # Read the raster data as a NumPy array (only band 1)
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
        index_name = os.path.splitext(index_file)[0].replace('mediane_clipped_', '').replace('_WAP', '')
        
        # Open the raster directly with GDAL (single-band rasters)
        ds = gdal.Open(index_path)
        if ds is None:
            print(f"Warning: Could not open {index_path}")
            continue
            
        # Read the raster data as a NumPy array (only band 1)
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

def filter_by_valid_proportion(input_csv, output_csv, min_valid_proportion=0.5):
    """
    Filter the CSV to keep only pixels with at least the specified proportion of valid data.
    A pixel is considered valid if its 'none' proportion is below (1 - min_valid_proportion).
    
    Args:
        input_csv: Path to the input CSV
        output_csv: Path to save the filtered CSV
        min_valid_proportion: Minimum proportion of valid (non-"none") pixels required
    """
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    
    # Check if 'none' is in the columns
    if 'none' in df.columns:
        # Filter by the inverse of 'none' proportion (i.e., keep pixels where 'none' is small)
        # A pixel should have at most (1 - min_valid_proportion) classified as 'none'
        max_none_proportion = 1 - min_valid_proportion
        df_filtered = df[df['none'] <= max_none_proportion]
        kept_rows = len(df_filtered)
        
        print(f"Filtered from {total_rows} to {kept_rows} pixels " +
              f"({kept_rows/total_rows*100:.1f}%) having at most {max_none_proportion*100:.0f}% 'none' class")
    else:
        # If 'none' column doesn't exist, fall back to the valid_proportion column
        df_filtered = df[df['valid_proportion'] >= min_valid_proportion]
        kept_rows = len(df_filtered)
        
        print(f"Filtered from {total_rows} to {kept_rows} pixels " +
              f"({kept_rows/total_rows*100:.1f}%) having at least {min_valid_proportion*100:.0f}% valid pixels")
    
    # Save filtered data
    df_filtered.to_csv(output_csv, index=False)
    
    return df_filtered

def plot_class_proportions(csv_path, output_dir, purcent_exclusion=0.05):
    """
    Create histograms of class proportions from the CSV file.
    Excludes samples with proportions less than or equal to purcent_exclusion% from the plot
    but shows their count in the subtitle.
    
    Args:
        csv_path: Path to the CSV file with class proportions
        output_dir: Directory to save the plots
        purcent_exclusion: Threshold below which samples are excluded (default: 0.05 = 5%)
    """
    import matplotlib.pyplot as plt
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find class columns (excluding metadata columns)
    metadata_cols = ['col_s', 'row_s', 'valid_pixels', 'total_pixels', 'valid_proportion']
    class_columns = [col for col in df.columns if col not in metadata_cols]

    # Ajout des throughs dans les plots récapitulatifs
    original_class_columns = [col for col in class_columns if not col.startswith('sqrt_')]
    if "through_proportion" not in original_class_columns and "through_proportion" in class_columns:
        original_class_columns.append("through_proportion")
    sqrt_class_columns = [col for col in class_columns if col.startswith('sqrt_')]
    if "sqrt_through_proportion" not in sqrt_class_columns and "sqrt_through_proportion" in class_columns:
        sqrt_class_columns.append("sqrt_through_proportion")

    # Plot all classes (original only, + through_proportion)
    plt.figure(figsize=(15, 10))
    n_classes = len(original_class_columns)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    for i, col in enumerate(original_class_columns):
        plt.subplot(n_rows, n_cols, i+1)
        low_count = (df[col] <= purcent_exclusion).sum()
        filtered_data = df[df[col] > purcent_exclusion][col]
        filtered_data.hist(bins=20)
        plt.title(f"{col}\n({low_count} samples ≤ {purcent_exclusion*100:.0f}% excluded)")
        plt.xlabel('Proportion')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_classes_histograms.png'))
    plt.close()

    # Plot sqrt classes (+ sqrt_through_proportion)
    if sqrt_class_columns:
        plt.figure(figsize=(15, 6))
        n_sqrt_classes = len(sqrt_class_columns)
        n_cols_sqrt = min(3, n_sqrt_classes)
        n_rows_sqrt = (n_sqrt_classes + n_cols_sqrt - 1) // n_cols_sqrt
        for i, col in enumerate(sqrt_class_columns):
            plt.subplot(n_rows_sqrt, n_cols_sqrt, i+1)
            low_count = (df[col] <= purcent_exclusion).sum()
            filtered_data = df[df[col] > purcent_exclusion][col]
            filtered_data.hist(bins=20)
            plt.title(f"{col}\n({low_count} samples ≤ {purcent_exclusion*100:.0f}% excluded)")
            plt.xlabel('Sqrt Proportion')
            plt.ylabel('Count')
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sqrt_all_classes_histograms.png'))
        plt.close()
    print(f"Class proportion histograms saved in {output_dir}")

def process_wap_data(wap_number, classification_path, output_dir=None, use_peat=True, superresolution=True, moy5m=True, class_names=None, through_class_names=None):
    """
    Process data for a specific WAP site.
    
    Args:
        wap_number: WAP site number (e.g., 23 or 32)
        classification_path: Path to the classification raster
        output_dir: Directory to save outputs (default is data/sentinel_proportions/WAP{wap_number})
        use_peat: Whether to use the peat dataset paths (with "_peat" suffix)
        superresolution: Whether to use 5m (True) or 10m (False) resolution data
    """
    # Set up paths
    if output_dir is None:
        output_dir = f"data/sentinel_proportions/WAP{wap_number}"
    
    # Paths to data, with conditional _peat suffix
    peat_suffix = "_peat" if use_peat else ""
    
    # Set resolution and path modifiers based on superresolution flag
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    mediane_dir = "mediane" if not moy5m else "mediane_10m"
    file_prefix = "" if not moy5m else "10m_"
    
    # Add resolution suffix to output directory if not using superresolution
    output_dir = f"{output_dir}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    sentinel_path = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_bands/{file_prefix}mediane_clipped_STACK_2023_BandB2_WAP{wap_number}_deflate.tif"
    sentinel_bands_dir = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_bands/"
    sentinel_indices_dir = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_indices/"


    # Output paths
    proportions_csv = os.path.join(output_dir, f"class_proportions_WAP{wap_number}.csv")
    filtered_csv = os.path.join(output_dir, f"class_proportions_WAP{wap_number}_filtered.csv")
    merged_csv = os.path.join(output_dir, f"sentinel_features_WAP{wap_number}.csv")
    plots_dir = os.path.join(output_dir, "plots")
    
    # Step 1: Compute class proportions for each Sentinel pixel
    print(f"Computing class proportions for WAP{wap_number}...")
    compute_class_proportions(
        classification_path=classification_path,
        sentinel_path=sentinel_path,
        output_csv=proportions_csv,
        class_names=class_names, 
        through_class_names=through_class_names
    )
    
    # Step 2: Filter by valid proportion
    print(f"Filtering by valid proportion...")
    filter_by_valid_proportion(
        input_csv=proportions_csv,
        output_csv=filtered_csv,
        min_valid_proportion=0.95  # Keep pixels with at least 95% valid data
    )
    
    # # Step 3: Extract Sentinel band and index values
    # print(f"Extracting Sentinel band and index values...")
    # extract_sentinel_values(
    #     sentinel_bands_dir=sentinel_bands_dir,
    #     sentinel_indices_dir=sentinel_indices_dir,
    #     proportions_csv=filtered_csv,
    #     output_csv=merged_csv
    # )
    
    # Step 4: Create plots
    print(f"Creating plots...")
    plot_class_proportions(
        csv_path=filtered_csv,
        output_dir=plots_dir,
        purcent_exclusion=0.00
    )

    print(f"All processing completed for WAP{wap_number}")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    wap = 23
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution = 5 if superresolution else 10
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m = False
    moy5m_suffix = "_moy5m" if moy5m else ""
    
    # Base output directory without resolution suffix
    base_output_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}"

    classification_path = f"drone_treated/WAP{wap}_tiles/WAP{wap}_classif_merged.tif"

    class_labels = {
        1: "Pure_Lichen",
        2: "Degraded_Lichen",
        3: "Green",
        4: "Sphagnum",
        5: "Depression",
        6: "Water",
        0: "No Data",
    }

    through_class_labels = ["Sphagnum", "Depression", "Water"]
       
    print(f"Processing WAP{wap} data with {'peat' if use_peat else 'standard'} dataset at {resolution}m resolution...")
    process_wap_data(wap, classification_path=classification_path, output_dir=base_output_dir, 
                   use_peat=use_peat, superresolution=superresolution, moy5m=moy5m, class_names=class_labels, through_class_names=through_class_labels)