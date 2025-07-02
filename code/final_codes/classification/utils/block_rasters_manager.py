import numpy as np
from osgeo import gdal
import gc

class BlockRastersManager:
    """
    A class to manage raster blocks for processing large rasters efficiently.
    Reads only specific windows of raster data as needed.
    """
    def __init__(self, rgb_path=None, dsm_path=None, thermal_path=None):
        """Initialize with paths to raster files."""
        self.rgb_path = rgb_path
        self.dsm_path = dsm_path
        self.thermal_path = thermal_path
        
        # Initialize data storage
        self.rast_r = None
        self.rast_g = None
        self.rast_b = None
        self.rast_z = None
        self.rast_t = None
        
        # Block information
        self.block = None
        self.x_start = None
        self.y_start = None
        self.x_size = None
        self.y_size = None
    
    def load_block(self, block):
        """
        Load a specific block of data from each raster.
        
        Args:
            block: Dictionary with x_start, y_start, x_end, y_end keys
        """
        self.clear_rasters()  # Clear any previously loaded data
        
        # Store block information
        self.block = block
        self.x_start = block['x_start']
        self.y_start = block['y_start']
        self.x_size = block['x_end'] - block['x_start']
        self.y_size = block['y_end'] - block['y_start']
        
        # Load RGB data
        if self.rgb_path:
            ds = gdal.Open(self.rgb_path)
            # Note: GDAL uses (xoff, yoff, xsize, ysize) order for ReadAsArray
            self.rast_r = ds.GetRasterBand(1).ReadAsArray(
                self.y_start, self.x_start, self.y_size, self.x_size)
            self.rast_g = ds.GetRasterBand(2).ReadAsArray(
                self.y_start, self.x_start, self.y_size, self.x_size)
            self.rast_b = ds.GetRasterBand(3).ReadAsArray(
                self.y_start, self.x_start, self.y_size, self.x_size)
            ds = None
        
        # Load DSM data
        if self.dsm_path:
            dz = gdal.Open(self.dsm_path)
            self.rast_z = dz.GetRasterBand(1).ReadAsArray(
                self.y_start, self.x_start, self.y_size, self.x_size) * 1000
            dz = None
        
        # Load thermal data
        if self.thermal_path:
            dt = gdal.Open(self.thermal_path)
            self.rast_t = dt.GetRasterBand(1).ReadAsArray(
                self.y_start, self.x_start, self.y_size, self.x_size)
            dt = None
    
    def clear_rasters(self):
        """Release rasters from memory."""
        self.rast_r = None
        self.rast_g = None
        self.rast_b = None
        self.rast_z = None
        self.rast_t = None
        gc.collect()  # Force garbage collection
    
    def get_patch(self, x, y, size_patch):
        """
        Extract patches from all available rasters at the given coordinates.
        Coordinates are relative to the current block, not the entire raster.
        
        Args:
            x, y: Coordinates relative to the current block
            size_patch: Size of the patch to extract
            
        Returns:
            tuple of (r, g, b, z, t) patches
        """
        # Make sure we don't go out of bounds
        if (x < 0 or y < 0 or 
            x + size_patch > self.x_size or 
            y + size_patch > self.y_size):
            raise ValueError(f"Patch coordinates ({x},{y}) with size {size_patch} "
                           f"exceed block bounds ({self.x_size},{self.y_size})")
        
        r = self.rast_r[x:x + size_patch, y:y + size_patch] if self.rast_r is not None else None
        g = self.rast_g[x:x + size_patch, y:y + size_patch] if self.rast_g is not None else None
        b = self.rast_b[x:x + size_patch, y:y + size_patch] if self.rast_b is not None else None
        z = self.rast_z[x:x + size_patch, y:y + size_patch] if self.rast_z is not None else None
        t = self.rast_t[x:x + size_patch, y:y + size_patch] if self.rast_t is not None else None
        
        return r, g, b, z, t
    
    def calculate_patch_stats(self, x, y, size_patch):
        """
        Calculate statistics for patches directly without storing the entire patch.
        Coordinates are relative to the current block, not the entire raster.
        
        Args:
            x, y: Coordinates relative to the current block
            size_patch: Size of the patch to calculate stats for
            
        Returns:
            Dictionary with mean and variance values for r, g, b, z, t
        """
        # Make sure we don't go out of bounds
        if (x < 0 or y < 0 or 
            x + size_patch > self.x_size or 
            y + size_patch > self.y_size):
            # For out of bounds, return None values
            stats = {k: None for k in [
                'r_mean', 'g_mean', 'b_mean', 'z_mean', 't_mean',
                'r_var', 'g_var', 'b_var', 'z_var', 't_var'
            ]}
            return stats
        
        stats = {}
        
        # RGB statistics
        if self.rast_r is not None and self.rast_g is not None and self.rast_b is not None:
            r_patch = self.rast_r[x:x + size_patch, y:y + size_patch]
            g_patch = self.rast_g[x:x + size_patch, y:y + size_patch]
            b_patch = self.rast_b[x:x + size_patch, y:y + size_patch]
            
            # Handle potential NaN values
            if np.isnan(r_patch).any() or np.isnan(g_patch).any() or np.isnan(b_patch).any():
                stats['r_mean'] = float(np.nanmean(r_patch)) if not np.isnan(r_patch).all() else 0
                stats['g_mean'] = float(np.nanmean(g_patch)) if not np.isnan(g_patch).all() else 0
                stats['b_mean'] = float(np.nanmean(b_patch)) if not np.isnan(b_patch).all() else 0
                stats['r_var'] = float(np.nanvar(r_patch)) if not np.isnan(r_patch).all() else 0
                stats['g_var'] = float(np.nanvar(g_patch)) if not np.isnan(g_patch).all() else 0
                stats['b_var'] = float(np.nanvar(b_patch)) if not np.isnan(b_patch).all() else 0
            else:
                stats['r_mean'] = float(np.mean(r_patch))
                stats['g_mean'] = float(np.mean(g_patch))
                stats['b_mean'] = float(np.mean(b_patch))
                stats['r_var'] = float(np.var(r_patch))
                stats['g_var'] = float(np.var(g_patch))
                stats['b_var'] = float(np.var(b_patch))
        else:
            stats['r_mean'] = stats['g_mean'] = stats['b_mean'] = None
            stats['r_var'] = stats['g_var'] = stats['b_var'] = None
            
        # Altitude (z)
        if self.rast_z is not None:
            z_patch = self.rast_z[x:x + size_patch, y:y + size_patch]
            if np.isnan(z_patch).any():
                stats['z_mean'] = float(np.nanmean(z_patch)) if not np.isnan(z_patch).all() else 0
                stats['z_var'] = float(np.nanvar(z_patch)) if not np.isnan(z_patch).all() else 0
            else:
                stats['z_mean'] = float(np.mean(z_patch))
                stats['z_var'] = float(np.var(z_patch))
        else:
            stats['z_mean'] = stats['z_var'] = None
            
        # Temperature (t)
        if self.rast_t is not None:
            t_patch = self.rast_t[x:x + size_patch, y:y + size_patch]
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
        """Return the shape of the current block."""
        if self.rast_r is not None:
            return self.rast_r.shape
        return (0, 0)
    
    def get_absolute_coords(self, rel_x, rel_y):
        """Convert block-relative coordinates to absolute raster coordinates."""
        return self.x_start + rel_x, self.y_start + rel_y
