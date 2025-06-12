"""
Implementation of Canny-Otsu filtering for DSM rasters.
This module provides functions to load DSM images, apply Canny edge detection 
followed by Otsu thresholding, and save the result.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, morphology
from osgeo import gdal


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Samples2.rasters_manager import RastersManager

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def load_dsm_as_array(dsm_path):
    """
    Load a DSM raster as a NumPy array using RastersManager.
    
    Args:
        dsm_path: Path to the DSM .tif file
    
    Returns:
        dsm_array: NumPy array containing the DSM data
    """
    print(f"Loading DSM from {dsm_path}")
    
    # Initialize the rasters manager
    rm = RastersManager()
    rm.set_paths(dz_path=dsm_path)
    
    # Load the raster
    rm.load_rasters()
    
    # Get the DSM array
    dsm_array = rm.rast_z
    
    print(f"Loaded DSM with shape: {dsm_array.shape}")
    
    return dsm_array

def apply_normalization_filter(dsm_array):
    """
    Apply only the normalization step to visualize the DSM after normalization.
    
    Args:
        dsm_array: NumPy array containing the DSM data
    
    Returns:
        dsm_norm: Normalized DSM array (0-255 range)
    """
    print("Applying normalization filter...")
    
    # Normalize DSM to 0-255
    dsm_norm = dsm_array.copy()
    
    # Handle NaN values if present
    if np.isnan(dsm_norm).any():
        print("Warning: NaN values detected in DSM. Replacing with minimum value.")
        min_val = np.nanmin(dsm_norm)
        dsm_norm = np.nan_to_num(dsm_norm, nan=min_val)
    
    # Normalize to 0-255
    dsm_min = np.min(dsm_norm)
    dsm_max = np.max(dsm_norm)
    
    dsm_norm = ((dsm_norm - dsm_min) / (dsm_max - dsm_min) * 255).astype(np.uint8)
    
    return dsm_norm


def enhance_edges_for_detection(dsm_norm):
    """
    Enhance edges in the DSM for better detection.
    
    Args:
        dsm_norm: Normalized DSM array (0-255 range)
        
    Returns:
        enhanced_dsm: Edge-enhanced DSM
    """
    # Apply unsharp masking to enhance edges
    blurred = filters.gaussian(dsm_norm, sigma=2)
    enhanced = dsm_norm + (dsm_norm - blurred) * 1.5
    
    # Clip to 0-255 range and convert to uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

def apply_identity_filter(dsm_array):
    """
    Apply identity filter (no change) to the DSM array.
    Included as a simple example filter.
    
    Args:
        dsm_array: NumPy array containing the DSM data
    
    Returns:
        dsm_array: The same array (unchanged)
    """
    print("Applying identity filter (no change)...")
    return dsm_array

def save_raster_as_tif(data_array, reference_tif_path, output_tif_path):
    """
    Save a NumPy array as a GeoTIFF using the georeference from a reference file.
    
    Args:
        data_array: NumPy array to save
        reference_tif_path: Path to reference GeoTIFF for georeference information
        output_tif_path: Path where output GeoTIFF will be saved
    """
    print(f"Saving raster as {output_tif_path}...")
    
    # Open the reference file to get georeference information
    ref_ds = gdal.Open(reference_tif_path)
    if ref_ds is None:
        raise ValueError(f"Could not open reference file: {reference_tif_path}")
    
    # Get georeference information
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    # Create output file
    driver = gdal.GetDriverByName('GTiff')
    
    # Handle binary vs. continuous data
    if data_array.dtype == bool:
        data_array = data_array.astype(np.uint8) * 255
        out_type = gdal.GDT_Byte
    elif np.issubdtype(data_array.dtype, np.integer):
        out_type = gdal.GDT_Int16
    else:
        out_type = gdal.GDT_Float32
    
    # Create output raster
    out_ds = driver.Create(
        output_tif_path,
        data_array.shape[1],  # Width
        data_array.shape[0],  # Height
        1,                    # Number of bands
        out_type              # Data type
    )
    
    # Set georeference information
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # Write data
    out_ds.GetRasterBand(1).WriteArray(data_array)
    
    # Close files
    out_ds = None
    ref_ds = None
    
    print(f"Raster saved successfully to {output_tif_path}")

def create_comparison_plot(original_array, filtered_array, title1="Original DSM", 
                          title2="Filtered DSM", output_path=None):
    """
    Create a subplot comparison of original and filtered DSM and save as PNG.
    
    Args:
        original_array: NumPy array of the original DSM
        filtered_array: NumPy array of the filtered DSM
        title1: Title for the original DSM subplot
        title2: Title for the filtered DSM subplot
        output_path: Path where output PNG will be saved
    """
    print("Creating comparison plot...")
    
    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original DSM
    im1 = axs[0].imshow(original_array, cmap='terrain')
    axs[0].set_title(title1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Plot filtered DSM
    if filtered_array.dtype == bool or (filtered_array.dtype == np.uint8 and np.max(filtered_array) == 1):
        # Binary image (edges)
        im2 = axs[1].imshow(filtered_array, cmap='gray')
        axs[1].set_title(title2)
    else:
        # Regular DSM
        im2 = axs[1].imshow(filtered_array, cmap='terrain')
        axs[1].set_title(title2)
        fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    plt.tight_layout()
    
    # Save if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    
    plt.close()
def filter_by_curvature_severity(edges, max_curvature_threshold=0.5, max_segment_length=500):
    """
    Filter edges to separate relatively straight segments from highly curved ones.
    
    Args:
        edges: Binary edge image
        max_curvature_threshold: Maximum allowed curvature ratio (lower = straighter)
        max_segment_length: Maximum length of segments to consider
        
    Returns:
        straight_segments: Binary image with relatively straight segments
    """
    from skimage import measure
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    
    # Create output image
    straight_segments = np.zeros_like(edges, dtype=bool)
    
    # Extract contours
    contours = measure.find_contours(edges.astype(float), 0.5)
    print(f"Found {len(contours)} contours to analyze")
    
    straight_count = 0
    curved_count = 0
    
    # Process each contour
    for contour in contours:
        if len(contour) >= max_segment_length or len(contour) < 10:
            continue
            
        # Pour les contours fermés, vérifier si le premier et dernier point sont proches
        is_closed = np.allclose(contour[0], contour[-1], rtol=0, atol=2.0)
        
        if is_closed:
            # Pour les contours fermés, utiliser la distance maximale entre tous les points
            # Calculer toutes les paires de distances
            distances = squareform(pdist(contour))
            # Trouver la distance maximale
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            direct_dist = distances[i, j]
            
            # Redéfinir les points de début et de fin
            start_point = contour[i]
            end_point = contour[j]
        else:
            # Pour les contours ouverts, utiliser les extrémités
            start_point = contour[0]
            end_point = contour[-1]
            direct_dist = np.sqrt(np.sum((end_point - start_point)**2))
        
        # Calculate path length along the contour
        path_length = 0
        for i in range(1, len(contour)):
            path_length += np.sqrt(np.sum((contour[i] - contour[i-1])**2))
        
        # Calculer le ratio de rectitude (amélioré)
        if path_length > 0:
            straightness_ratio = direct_dist / path_length
        else:
            straightness_ratio = 0
            
        # Determine if contour is straight enough
        if straightness_ratio > max_curvature_threshold:
            # This is a relatively straight segment
            for i in range(len(contour)):
                r, c = int(contour[i, 0]), int(contour[i, 1])
                if 0 <= r < edges.shape[0] and 0 <= c < edges.shape[1]:
                    straight_segments[r, c] = True
            straight_count += 1
        else:
            curved_count += 1
    
    print(f"Classified {straight_count} segments as relatively straight, {curved_count} as curved")
    return straight_segments

def apply_canny_otsu(dsm_array, sigma_prefilter = 2, sigma=1.0, low_threshold=None, high_threshold=None, debug=False):
    """
    Apply Canny edge detection followed by Otsu thresholding.
    
    Args:
        dsm_array: NumPy array containing the DSM data
        sigma: Standard deviation for Gaussian filter in Canny
        low_threshold: Low threshold for Canny (if None, calculated automatically)
        high_threshold: High threshold for Canny (if None, calculated automatically)
        debug: If True, print additional debug information
    
    Returns:
        edges: Binary array representing detected edges
    """
    print("Applying Canny edge detection...")
    
    # Normalize DSM to 0-255 for better edge detection
    dsm_norm = apply_normalization_filter(dsm_array)
    
    if debug:
        dsm_min = np.min(dsm_array)
        dsm_max = np.max(dsm_array)
        print(f"DSM range before normalization: {dsm_min} to {dsm_max}")

    # Pre-smooth the DSM to remove small-scale noise
    dsm_norm = filters.gaussian(dsm_norm, sigma=sigma_prefilter)
    
    # Calculate the gradient magnitude using Sobel filters
    gx = filters.sobel_h(dsm_norm)
    gy = filters.sobel_v(dsm_norm)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Compute mean and std of gradient magnitude for better thresholding
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)
    
        # Apply Canny edge detection
    if low_threshold is None or high_threshold is None:
        # Use relative thresholds based on percentages of gradient magnitude
        # (avoids issues with very small gradient values)
        grad_max = np.max(grad_mag)
        low_threshold = 0.05 * grad_max   # 5% of maximum gradient
        high_threshold = 0.15 * grad_max  # 15% of maximum gradient
        
        if debug:
            print(f"Gradient mean: {grad_mean}, std: {grad_std}, max: {grad_max}")
            print(f"Low threshold: {low_threshold}, High threshold: {high_threshold}")

    edges = feature.canny(
        dsm_norm,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    # # Remove small edge segments and strengthen major edges
   
    # edges = morphology.binary_dilation(edges, morphology.disk(30))  # Thicken remaining edges
    # edges = morphology.binary_erosion(edges, morphology.disk(25))  # Clean up
    
    edges = morphology.binary_dilation(edges, morphology.disk(5))  # Thicken remaining edges
    edges = morphology.binary_erosion(edges, morphology.disk(3))  # Clean up
    # 4. Remove very small objects
    edges = morphology.remove_small_objects(edges, min_size=500)
    # edges = morphology.binary_dilation(edges, morphology.disk(2))

    # Dans votre fonction apply_canny_otsu, après la détection des contours:

    # # Filter to keep only relatively straight segments
    # straight_edges = filter_by_curvature_severity(
    #     edges,
    #     max_curvature_threshold=0.5,  # Ajustez cette valeur: 
    #                                    # 0.9=très droit, 0.5=modérément droit, 0.3=légèrement courbé
    #     max_segment_length=500         # Longueur maximale des segments à considérer
    # )
    
    # # # Remplacer les arêtes initiales par les segments relativement droits
    # edges = straight_edges


    if debug:
        # Count number of edge pixels
        edge_count = np.sum(edges)
        total_pixels = edges.size
        edge_percentage = (edge_count / total_pixels) * 100
        print(f"Edge detection found {edge_count} edge pixels ({edge_percentage:.2f}% of image)")
    
    print("Edge detection complete.")
    return edges



def process_dsm_with_filter(dsm_path, output_dir, sigma_prefilter=2, filter_func=None, filter_name=None,
                           filter_kwargs=None, save_tif=True, save_png=True):
    """
    Process a DSM raster with a specified filter function and save results.
    
    Args:
        dsm_path: Path to the input DSM .tif file
        output_dir: Directory where outputs will be saved
        filter_func: Function to apply to the DSM array (default: apply_identity_filter)
        filter_name: Name of the filter for filenames (default: "identity")
        filter_kwargs: Dictionary of keyword arguments for the filter function
        save_tif: Whether to save the result as a GeoTIFF (default: True)
        save_png: Whether to save comparison plot as PNG (default: True)
    """
    # Set defaults
    if filter_func is None:
        filter_func = apply_identity_filter
        
    if filter_name is None:
        filter_name = "identity"
        
    if filter_kwargs is None:
        filter_kwargs = {}
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Extract base filename
    base_filename = os.path.splitext(os.path.basename(dsm_path))[0]

    # Load DSM
    dsm_array = load_dsm_as_array(dsm_path)
    
    # Apply filter
    filtered_array = filter_func(dsm_array, **filter_kwargs)
    
    output_tif_path = None
    output_plot_path = None
    
    # Save filtered raster as TIF if requested
    if save_tif:
        output_tif_path = os.path.join(output_dir, f"{base_filename}_{filter_name}.tif")
        save_raster_as_tif(filtered_array, dsm_path, output_tif_path)
    
    # Create and save comparison plot if requested
    if save_png:
        output_plot_path = os.path.join(output_dir, f"{base_filename}_{filter_name}_comparison.png")
        create_comparison_plot(
            dsm_array, 
            filtered_array, 
            title1=f"Original DSM ({base_filename})",
            title2=f"Filtered DSM ({filter_name})", 
            output_path=output_plot_path
        )
    
    print(f"Processing complete for {base_filename}")
    return output_tif_path, output_plot_path

def main():
    """Main function to demonstrate the Canny-Otsu filtering."""
    # Define input and output paths
    dsm_path = "drone_treated/WAP32_tiles/dsm/WAP32_full_dsm_08_04.tif"
    output_dir = "data/canyotsu"
    
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Try different sigma values with improved thresholds
    print("\n--- Testing different sigma values ---")
    for sigma in [45]:
        canny_otsu_kwargs = {'sigma': sigma, 'debug': True}
        process_dsm_with_filter(
            dsm_path,
            output_dir,
            sigma_prefilter = 2,
            filter_func=apply_canny_otsu,
            filter_name=f"canny_improved_sigma{sigma}",
            filter_kwargs=canny_otsu_kwargs,
            save_tif=False,
            save_png=True
        )
        
if __name__ == "__main__":
    main()
