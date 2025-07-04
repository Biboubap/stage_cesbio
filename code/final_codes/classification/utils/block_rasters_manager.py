import numpy as np
from osgeo import gdal
import gc

class BlockRastersManager:
    """
    A class to manage raster blocks for processing large rasters efficiently.
    
    This class handles the loading and management of large geospatial raster data by 
    reading only specific blocks (windows) of the data as needed. This approach allows
    processing very large images that wouldn't fit in memory all at once.
    
    The class supports RGB, Digital Surface Models (DSM), and thermal data.
    It provides methods to extract data patches and calculate statistics for each patch,
    which are used as features in the classification process.
    """
    def __init__(self, rgb_path=None, dsm_path=None, thermal_path=None):
        """
        Initialize with paths to raster files.
        
        Args:
            rgb_path: Path to the RGB image (GeoTIFF)
            dsm_path: Path to the Digital Surface Model (GeoTIFF)
            thermal_path: Path to the thermal image (GeoTIFF), optional
        """
        self.rgb_path = rgb_path        # Path to RGB image
        self.dsm_path = dsm_path        # Path to Digital Surface Model
        self.thermal_path = thermal_path # Path to thermal data (optional)
        
        # Initialize data storage (will be filled with actual data when load_block is called)
        self.rast_r = None  # Red channel data for current block
        self.rast_g = None  # Green channel data for current block
        self.rast_b = None  # Blue channel data for current block
        self.rast_z = None  # DSM/elevation data for current block
        self.rast_t = None  # Thermal data for current block (if available)
        
        # Block information (will be set when a specific block is loaded)
        self.block = None   # Dictionary with block coordinates
        self.x_start = None # Starting X coordinate of the block
        self.y_start = None # Starting Y coordinate of the block
        self.x_size = None  # Width of the block in pixels
        self.y_size = None  # Height of the block in pixels
    
    def load_block(self, block):
        """
        Load a specific block of data from each raster.
        
        This method loads a rectangular portion (block) of each input raster into memory.
        It uses GDAL's ReadAsArray with the block coordinates to read only the necessary data.
        
        Args:
            block: Dictionary with x_start, y_start, x_end, y_end keys defining the block boundaries
        """
        self.clear_rasters()  # Clear any previously loaded data to free memory
        
        # Store block information for reference
        self.block = block
        self.x_start = block['x_start']
        self.y_start = block['y_start']
        self.x_size = block['x_end'] - block['x_start']
        self.y_size = block['y_end'] - block['y_start']
        
        # Load RGB data if path is provided
        if self.rgb_path:
            ds = gdal.Open(self.rgb_path)
            # Note: GDAL uses (xoff, yoff, xsize, ysize) order for ReadAsArray
            self.rast_r = ds.GetRasterBand(1).ReadAsArray(
                self.x_start, self.y_start, self.x_size, self.y_size)
            self.rast_g = ds.GetRasterBand(2).ReadAsArray(
                self.x_start, self.y_start, self.x_size, self.y_size)
            self.rast_b = ds.GetRasterBand(3).ReadAsArray(
                self.x_start, self.y_start, self.x_size, self.y_size)
            ds = None  # Close the dataset to free resources
    
        # Load DSM data if path is provided
        if self.dsm_path:
            dz = gdal.Open(self.dsm_path)
            # Multiply by 1000 to convert to millimeters (preserves precision)
            self.rast_z = dz.GetRasterBand(1).ReadAsArray(
                self.x_start, self.y_start, self.x_size, self.y_size) * 1000
            dz = None  # Close the dataset
        
        # Load thermal data if path is provided
        if self.thermal_path:
            dt = gdal.Open(self.thermal_path)
            self.rast_t = dt.GetRasterBand(1).ReadAsArray(
                self.x_start, self.y_start, self.x_size, self.y_size)
            dt = None  # Close the dataset

    def clear_rasters(self):
        """
        Release rasters from memory to free up resources.
        
        This method is called before loading a new block and can also be called
        manually when the data is no longer needed.
        """
        self.rast_r = None
        self.rast_g = None
        self.rast_b = None
        self.rast_z = None
        self.rast_t = None
        gc.collect()  # Force garbage collection to free memory
    
    def get_patch(self, x, y, size_patch):
        """
        Extract patches from all available rasters at the given coordinates.
        
        This method returns actual data arrays for the specified patch location.
        Coordinates are relative to the current block, not the entire raster.
        
        Args:
            x, y: Coordinates relative to the current block
            size_patch: Size of the patch to extract
            
        Returns:
            tuple of (r, g, b, z, t) patches, any of which may be None if the data isn't available
        """
        # Make sure we don't go out of bounds of the current block
        if (x < 0 or y < 0 or 
            x + size_patch > self.x_size or 
            y + size_patch > self.y_size):
            raise ValueError(f"Patch coordinates ({x},{y}) with size {size_patch} "
                           f"exceed block bounds ({self.x_size},{self.y_size})")
        
        # Extract the patch from each available raster band
        r = self.rast_r[y:y + size_patch, x:x + size_patch] if self.rast_r is not None else None
        g = self.rast_g[y:y + size_patch, x:x + size_patch] if self.rast_g is not None else None
        b = self.rast_b[y:y + size_patch, x:x + size_patch] if self.rast_b is not None else None
        z = self.rast_z[y:y + size_patch, x:x + size_patch] if self.rast_z is not None else None
        t = self.rast_t[y:y + size_patch, x:x + size_patch] if self.rast_t is not None else None
        
        return r, g, b, z, t
    
    def calculate_patch_stats(self, x, y, size_patch):
        """
        Calculate statistics for patches directly without storing the entire patch.
        
        This method computes mean and variance values for each data type (RGB, DSM, thermal)
        for a patch at the given location. It handles potential NaN values in the data.
        Coordinates are relative to the current block, not the entire raster.
        
        Args:
            x, y: Coordinates relative to the current block
            size_patch: Size of the patch to calculate stats for
            
        Returns:
            Dictionary with mean and variance values for r, g, b, z, t
        """
        # Check if the patch coordinates are within the block bounds
        if (x < 0 or y < 0 or 
            x + size_patch > self.x_size or 
            y + size_patch > self.y_size):
            # For out of bounds requests, return None values
            stats = {k: None for k in [
                'r_mean', 'g_mean', 'b_mean', 'z_mean', 't_mean',
                'r_var', 'g_var', 'b_var', 'z_var', 't_var'
            ]}
            return stats
        
        stats = {}
        
        # Calculate RGB statistics if RGB data is available
        if self.rast_r is not None and self.rast_g is not None and self.rast_b is not None:
            r_patch = self.rast_r[y:y + size_patch, x:x + size_patch]
            g_patch = self.rast_g[y:y + size_patch, x:x + size_patch]
            b_patch = self.rast_b[y:y + size_patch, x:x + size_patch]
            
            # Handle potential NaN values in the data
            if np.isnan(r_patch).any() or np.isnan(g_patch).any() or np.isnan(b_patch).any():
                # Use nanmean/nanvar to ignore NaN values in calculations
                stats['r_mean'] = float(np.nanmean(r_patch)) if not np.isnan(r_patch).all() else 0
                stats['g_mean'] = float(np.nanmean(g_patch)) if not np.isnan(g_patch).all() else 0
                stats['b_mean'] = float(np.nanmean(b_patch)) if not np.isnan(b_patch).all() else 0
                stats['r_var'] = float(np.nanvar(r_patch)) if not np.isnan(r_patch).all() else 0
                stats['g_var'] = float(np.nanvar(g_patch)) if not np.isnan(g_patch).all() else 0
                stats['b_var'] = float(np.nanvar(b_patch)) if not np.isnan(b_patch).all() else 0
            else:
                # No NaN values, use standard mean/var for better performance
                stats['r_mean'] = float(np.mean(r_patch))
                stats['g_mean'] = float(np.mean(g_patch))
                stats['b_mean'] = float(np.mean(b_patch))
                stats['r_var'] = float(np.var(r_patch))
                stats['g_var'] = float(np.var(g_patch))
                stats['b_var'] = float(np.var(b_patch))
        else:
            # No RGB data available, set stats to None
            stats['r_mean'] = stats['g_mean'] = stats['b_mean'] = None
            stats['r_var'] = stats['g_var'] = stats['b_var'] = None
            
        # Calculate DSM/elevation statistics if available
        if self.rast_z is not None:
            z_patch = self.rast_z[y:y + size_patch, x:x + size_patch]
            if np.isnan(z_patch).any():
                stats['z_mean'] = float(np.nanmean(z_patch)) if not np.isnan(z_patch).all() else 0
                stats['z_var'] = float(np.nanvar(z_patch)) if not np.isnan(z_patch).all() else 0
            else:
                stats['z_mean'] = float(np.mean(z_patch))
                stats['z_var'] = float(np.var(z_patch))
        else:
            stats['z_mean'] = stats['z_var'] = None
            
        # Calculate thermal statistics if available
        if self.rast_t is not None:
            t_patch = self.rast_t[y:y + size_patch, x:x + size_patch]
            if np.isnan(t_patch).any():
                stats['t_mean'] = float(np.nanmean(t_patch)) if not np.isnan(t_patch).all() else 0
                stats['t_var'] = float(np.nanvar(t_patch)) if not np.isnan(t_patch).all() else 0
            else:
                stats['t_mean'] = float(np.mean(t_patch))
                stats['t_var'] = float(np.var(t_patch))
        else:
            stats['t_mean'] = stats['t_var'] = None
            
        return stats
    
    def get_block_shape(self):
        """
        Return the shape (height, width) of the current block.
        
        Returns:
            Tuple of (height, width) or (0, 0) if no block is loaded
        """
        if self.rast_r is not None:
            return self.rast_r.shape
        return (0, 0)
    
    def get_absolute_coords(self, rel_x, rel_y):
        """
        Convert block-relative coordinates to absolute raster coordinates.
        
        Args:
            rel_x, rel_y: Coordinates relative to the current block
            
        Returns:
            Tuple of (abs_x, abs_y) absolute coordinates in the full raster
        """
        return self.x_start + rel_x, self.y_start + rel_y
