import numpy as np
from osgeo import gdal
import gc 

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
    
    def set_paths(self, ds_path=None, dz_path=None, dt_path=None):
        """Set paths to raster files."""
        self.ds_path = ds_path
        self.dz_path = dz_path
        self.dt_path = dt_path
        
    def load_rasters(self):
        """Load rasters into memory."""
        if self.is_loaded:
            return
            
        if self.ds_path:
            ds = gdal.Open(self.ds_path)
            self.rast_r = ds.GetRasterBand(1).ReadAsArray()
            self.rast_g = ds.GetRasterBand(2).ReadAsArray()
            self.rast_b = ds.GetRasterBand(3).ReadAsArray()
            
        if self.dz_path:
            dz = gdal.Open(self.dz_path)
            self.rast_z = dz.GetRasterBand(1).ReadAsArray()*1000
            
        if self.dt_path:
            dt = gdal.Open(self.dt_path)
            self.rast_t = dt.GetRasterBand(1).ReadAsArray()
            
        self.is_loaded = True
    
    def clear_rasters(self):
        """Release rasters from memory."""
        self.rast_r = None
        self.rast_g = None
        self.rast_b = None
        self.rast_z = None
        self.rast_t = None
        self.is_loaded = False
        gc.collect()  # Force garbage collection to free memory
    
    def get_patch(self, x, y, size_patch):
        """
        Extract patches from all available rasters at the given coordinates.
        Returns tuple of (r, g, b, z, t) patches.
        """
        if not self.is_loaded:
            self.load_rasters()
            
        r = self.rast_r[x:x + size_patch, y:y + size_patch] if self.rast_r is not None else None
        g = self.rast_g[x:x + size_patch, y:y + size_patch] if self.rast_g is not None else None
        b = self.rast_b[x:x + size_patch, y:y + size_patch] if self.rast_b is not None else None
        z = self.rast_z[x:x + size_patch, y:y + size_patch] if self.rast_z is not None else None
        t = self.rast_t[x:x + size_patch, y:y + size_patch] if self.rast_t is not None else None
        
        return r, g, b, z, t
    
    def calculate_patch_stats(self, x, y, size_patch):
        """
        Calculate statistics for patches directly without storing the entire patch.
        Returns a dictionary with mean and variance values for r, g, b, z, t.
        """
        if not self.is_loaded:
            self.load_rasters()

        stats = {}
        
        # RGB statistics
        if self.rast_r is not None and self.rast_g is not None and self.rast_b is not None:
            r_patch = self.rast_r[x:x + size_patch, y:y + size_patch]
            g_patch = self.rast_g[x:x + size_patch, y:y + size_patch]
            b_patch = self.rast_b[x:x + size_patch, y:y + size_patch]
            
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
            stats['z_mean'] = float(np.mean(z_patch))
            stats['z_var'] = float(np.var(z_patch))
        else:
            stats['z_mean'] = stats['z_var'] = None
            
        # Température (t)
        if self.rast_t is not None:
            t_patch = self.rast_t[x:x + size_patch, y:y + size_patch]
            stats['t_mean'] = float(np.mean(t_patch))
            stats['t_var'] = float(np.var(t_patch))
        else:
            stats['t_mean'] = stats['t_var'] = None
            
        return stats