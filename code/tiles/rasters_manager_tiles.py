import numpy as np
import rasterio
from rasterio.windows import Window
import os
from osgeo import gdal
import gc  # Import garbage collector module

class RastersManager:
    """
    Singleton class to manage raster data across the application.
    Provides centralized loading/unloading of rasters to optimize memory usage.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RastersManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize raster attributes."""
        self.rast_r = None
        self.rast_g = None
        self.rast_b = None
        self.rast_z = None
        self.rast_t = None
        self.ds_path = None
        self.dz_path = None
        self.dt_path = None
        self.is_loaded = False
        self.offset_x = 0
        self.offset_y = 0
        self.ds_handle = None
        self.dz_handle = None
        self.dt_handle = None
        self.full_width = 0
        self.full_height = 0
        # Cache for memoization - limit cache size to prevent memory issues
        self._window_cache = {}
        self._max_cache_size = 3  # Maximum number of windows to keep in cache
        self._z_scale = 1000  # Store the Z scale factor instead of applying it to the whole array
        # Cache for raster info to avoid reopening files
        self._raster_info_cache = {}
    
    def set_paths(self, ds_path=None, dz_path=None, dt_path=None):
        """Set paths to raster files."""
        self.ds_path = ds_path
        self.dz_path = dz_path
        self.dt_path = dt_path
        
        # Clear any existing data before setting new paths
        self.clear_rasters()
        self.clear_cache()
        
        # Store the full dimensions of the raster using the optimized method
        if ds_path:
            width, height, _, _ = self.get_raster_info(ds_path)
            if width is not None and height is not None:
                self.full_width = width
                self.full_height = height
    
    def load_rasters(self):
        """Load rasters into memory."""
        self.load_raster_window(0, 0, None, None)
        
    def load_raster_window(self, x_offset, y_offset, win_width=None, win_height=None, optimized=False):
        """
        Load a specific window of the rasters into memory.
        
        Args:
            x_offset, y_offset: Offset coordinates for the window
            win_width, win_height: Size of the window (None for the rest of the raster)
            optimized: If True, uses more memory efficient methods for large rasters
        """
        # Check cache first
        cache_key = (x_offset, y_offset, win_width, win_height)
        if cache_key in self._window_cache:
            cache_data = self._window_cache[cache_key]
            self.rast_r = cache_data.get('r')
            self.rast_g = cache_data.get('g')
            self.rast_b = cache_data.get('b')
            self.rast_z = cache_data.get('z')
            self.rast_t = cache_data.get('t')
            self.offset_x = x_offset
            self.offset_y = y_offset
            self.is_loaded = True
            return
            
        # Clear current rasters to free memory
        self.clear_rasters()
        self.offset_x = x_offset
        self.offset_y = y_offset
        
        # Close previous handles if they exist
        self.close_handles()
        
        # Get raster dimensions without loading the whole file
        if self.ds_path:
            width, height, _, _ = self.get_raster_info(self.ds_path)
            
            # Set default window size to full raster if not specified
            if win_width is None:
                win_width = width - x_offset
            if win_height is None:
                win_height = height - y_offset
            
            # Ensure window doesn't exceed raster bounds
            win_width = min(win_width, width - x_offset)
            win_height = min(win_height, height - y_offset)
            
            # Load RGB data
            if optimized:
                # Use GDAL for optimized window reading for RGB
                try:
                    ds = gdal.Open(self.ds_path)
                    if ds:
                        # Read each band separately to reduce memory usage
                        red_band = ds.GetRasterBand(1)
                        self.rast_r = red_band.ReadAsArray(x_offset, y_offset, win_width, win_height)
                        red_band = None
                        
                        green_band = ds.GetRasterBand(2)
                        self.rast_g = green_band.ReadAsArray(x_offset, y_offset, win_width, win_height)
                        green_band = None
                        
                        blue_band = ds.GetRasterBand(3)
                        self.rast_b = blue_band.ReadAsArray(x_offset, y_offset, win_width, win_height)
                        blue_band = None
                        
                        ds = None  # Close dataset
                    else:
                        print(f"Could not open raster {self.ds_path} with GDAL")
                        self.rast_r = self.rast_g = self.rast_b = None
                except Exception as e:
                    print(f"Error loading RGB data with GDAL: {e}")
                    self.rast_r = self.rast_g = self.rast_b = None
            else:
                # Use rasterio for general case
                try:
                    with rasterio.open(self.ds_path) as src:
                        # Create a rasterio Window
                        window = Window(x_offset, y_offset, win_width, win_height)
                        
                        # Load RGB bands
                        data = src.read([1, 2, 3], window=window)
                        self.rast_r = data[0]
                        self.rast_g = data[1]
                        self.rast_b = data[2]
                        
                        # Free data array to save memory
                        data = None
                except Exception as e:
                    print(f"Error loading RGB data with rasterio: {e}")
                    self.rast_r = self.rast_g = self.rast_b = None
        
        # Load Z band
        if self.dz_path:
            if optimized:
                # Use GDAL for Z-data
                try:
                    dz = gdal.Open(self.dz_path)
                    if dz:
                        z_band = dz.GetRasterBand(1)
                        self.rast_z = z_band.ReadAsArray(x_offset, y_offset, win_width, win_height)
                        z_band = None
                        dz = None  # Close dataset
                    else:
                        print(f"Could not open raster {self.dz_path} with GDAL")
                        self.rast_z = None
                except Exception as e:
                    print(f"Error loading Z data with GDAL: {e}")
                    self.rast_z = None
            else:
                # Use rasterio for Z
                try:
                    with rasterio.open(self.dz_path) as src:
                        window = Window(x_offset, y_offset, win_width, win_height)
                        # Don't multiply by 1000 here - we'll apply the scaling when needed
                        self.rast_z = src.read(1, window=window)
                except Exception as e:
                    print(f"Error loading Z data with rasterio: {e}")
                    self.rast_z = None
            
        # Load T band
        if self.dt_path:
            if optimized:
                # Use GDAL for T-data
                try:
                    dt = gdal.Open(self.dt_path)
                    if dt:
                        t_band = dt.GetRasterBand(1)
                        self.rast_t = t_band.ReadAsArray(x_offset, y_offset, win_width, win_height)
                        t_band = None
                        dt = None  # Close dataset
                    else:
                        print(f"Could not open raster {self.dt_path} with GDAL")
                        self.rast_t = None
                except Exception as e:
                    print(f"Error loading T data with GDAL: {e}")
                    self.rast_t = None
            else:
                # Use rasterio for T
                try:
                    with rasterio.open(self.dt_path) as src:
                        window = Window(x_offset, y_offset, win_width, win_height)
                        self.rast_t = src.read(1, window=window)
                except Exception as e:
                    print(f"Error loading T data with rasterio: {e}")
                    self.rast_t = None
            
        self.is_loaded = True
        
        # Cache the loaded data, but manage cache size
        if len(self._window_cache) >= self._max_cache_size:
            # Remove the oldest entry
            oldest_key = next(iter(self._window_cache))
            del self._window_cache[oldest_key]
    
        # Store in cache
        self._window_cache[cache_key] = {
            'r': self.rast_r,
            'g': self.rast_g,
            'b': self.rast_b,
            'z': self.rast_z,
            't': self.rast_t
        }
        
        # Force garbage collection after loading new raster data
        gc.collect()
    
    def close_handles(self):
        """Close all open dataset handles."""
        if self.ds_handle:
            self.ds_handle = None
        if self.dz_handle:
            self.dz_handle = None
        if self.dt_handle:
            self.dt_handle = None
        gc.collect()  # Force garbage collection
    
    def clear_rasters(self):
        """Release rasters from memory."""
        self.rast_r = None
        self.rast_g = None
        self.rast_b = None
        self.rast_z = None
        self.rast_t = None
        self.is_loaded = False
        gc.collect()  # Force garbage collection
    
    def clear_cache(self):
        """Clear the window cache to free memory."""
        self._window_cache.clear()
        gc.collect()  # Force garbage collection
        
    def get_patch(self, x, y, size_patch):
        """
        Extract patches from all available rasters at the given coordinates.
        Returns tuple of (r, g, b, z, t) patches.
        """
        if not self.is_loaded:
            self.load_rasters()
        
        # Adjust coordinates to account for the window offset
        local_x = x - self.offset_x
        local_y = y - self.offset_y
            
        r = self.rast_r[local_x:local_x + size_patch, local_y:local_y + size_patch] if self.rast_r is not None else None
        g = self.rast_g[local_x:local_x + size_patch, local_y:local_y + size_patch] if self.rast_g is not None else None
        b = self.rast_b[local_x:local_x + size_patch, local_y:local_y + size_patch] if self.rast_b is not None else None
        
        # Apply scaling on-the-fly only for the small patch
        if self.rast_z is not None:
            z_patch = self.rast_z[local_x:local_x + size_patch, local_y:local_y + size_patch]
            z = z_patch * self._z_scale
        else:
            z = None
            
        t = self.rast_t[local_x:local_x + size_patch, local_y:local_y + size_patch] if self.rast_t is not None else None
        
        return r, g, b, z, t
    
    def calculate_patch_stats(self, x, y, size_patch):
        """
        Calculate statistics for patches directly without storing the entire patch.
        Returns a dictionary with mean and variance values for r, g, b, z, t.
        """
        if not self.is_loaded:
            self.load_rasters()

        # Adjust coordinates to account for the window offset
        local_x = x - self.offset_x
        local_y = y - self.offset_y

        stats = {}
        
        # RGB statistics
        if self.rast_r is not None and self.rast_g is not None and self.rast_b is not None:
            r_patch = self.rast_r[local_x:local_x + size_patch, local_y:local_y + size_patch]
            g_patch = self.rast_g[local_x:local_x + size_patch, local_y:local_y + size_patch]
            b_patch = self.rast_b[local_x:local_x + size_patch, local_y:local_y + size_patch]
            
            stats['r_mean'] = float(np.mean(r_patch))
            stats['g_mean'] = float(np.mean(g_patch))
            stats['b_mean'] = float(np.mean(b_patch))
            stats['r_var'] = float(np.var(r_patch))
            stats['g_var'] = float(np.var(g_patch))
            stats['b_var'] = float(np.var(b_patch))
        else:
            stats['r_mean'] = stats['g_mean'] = stats['b_mean'] = None
            stats['r_var'] = stats['g_var'] = stats['b_var'] = None
            
        # Altitude (z) - apply scaling factor only on the calculated statistics
        if self.rast_z is not None:
            z_patch = self.rast_z[local_x:local_x + size_patch, local_y:local_y + size_patch]
            stats['z_mean'] = float(np.mean(z_patch)) * self._z_scale
            stats['z_var'] = float(np.var(z_patch)) * self._z_scale * self._z_scale  # Variance scales with square of the scale factor
        else:
            stats['z_mean'] = stats['z_var'] = None
            
        # Température (t)
        if self.rast_t is not None:
            t_patch = self.rast_t[local_x:local_x + size_patch, local_y:local_y + size_patch]
            stats['t_mean'] = float(np.mean(t_patch))
            stats['t_var'] = float(np.var(t_patch))
        else:
            stats['t_mean'] = stats['t_var'] = None
            
        return stats

    def get_raster_info(self, filepath):
        """
        Get raster information like size and geotransform using GDAL without loading the entire raster.
        Uses a cache to avoid re-opening files for metadata.
        """
        # Check cache first
        if filepath in self._raster_info_cache:
            return self._raster_info_cache[filepath]
            
        # Use GDAL directly for faster metadata access
        try:
            ds = gdal.Open(filepath)
            if ds is None:
                print(f"Failed to open {filepath} with GDAL")
                return None, None, None, None
                
            width = ds.RasterXSize
            height = ds.RasterYSize
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            # Cache the results
            self._raster_info_cache[filepath] = (width, height, geotransform, projection)
            
            # Close the dataset
            ds = None
            
            return width, height, geotransform, projection
            
        except Exception as e:
            print(f"Error getting raster info with GDAL: {e}")
            
            # Try with rasterio as fallback
            try:
                with rasterio.open(filepath) as src:
                    width = src.width
                    height = src.height
                    geotransform = src.transform.to_gdal()
                    projection = src.crs.to_wkt() if src.crs else None
                    
                # Cache the results
                self._raster_info_cache[filepath] = (width, height, geotransform, projection)
                return width, height, geotransform, projection
            except Exception as e2:
                print(f"Error getting raster info with rasterio: {e2}")
                return None, None, None, None