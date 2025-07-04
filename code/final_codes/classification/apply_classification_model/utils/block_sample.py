import numpy as np

class BlockSample:
    """
    Sample class adapted for block-based processing.
    
    This class represents a patch (small area) within a larger block of raster data.
    It stores the patch's location within the block and calculates various statistics
    about the patch (RGB values, altitude/DSM values, temperature/thermal values).
    These statistics are later used as features for the classification algorithm.
    
    The class also computes neighborhood statistics by examining surrounding patches,
    which helps to provide spatial context for the classification.
    """
    def __init__(self, i_x, i_y, x, y, size_patch, block_rasters):
        """
        Initialize a sample within a block.
        
        Args:
            i_x, i_y: Sample indices (position in the grid of samples)
            x, y: Pixel coordinates within the block (top-left corner of the patch)
            size_patch: Size of the patch in pixels (patches are square)
            block_rasters: BlockRastersManager instance that provides access to the raster data
        """
        # Sample identifiers and location information
        self.i_x = i_x  # Index in the x direction (column in the grid)
        self.i_y = i_y  # Index in the y direction (row in the grid)
        self.x = x      # X coordinate in pixels within the block
        self.y = y      # Y coordinate in pixels within the block
        self.size_patch = size_patch  # Size of the patch in pixels
        
        # Calculate statistics for this patch using the block_rasters manager
        stats = block_rasters.calculate_patch_stats(x, y, size_patch)
        
        # RGB statistics - mean and variance for each channel
        self.r_mean = stats['r_mean']  # Mean red channel value
        self.g_mean = stats['g_mean']  # Mean green channel value
        self.b_mean = stats['b_mean']  # Mean blue channel value
        self.r_var = stats['r_var']    # Variance in red channel
        self.g_var = stats['g_var']    # Variance in green channel
        self.b_var = stats['b_var']    # Variance in blue channel

        # Altitude (z) statistics from DSM
        self.z_mean = stats['z_mean']  # Mean elevation value
        self.z_var = stats['z_var']    # Variance in elevation

        # Temperature (t) statistics from thermal data if available
        self.t_mean = stats['t_mean']  # Mean temperature value
        self.t_var = stats['t_var']    # Variance in temperature

        # Neighborhood statistics (initialized as None)
        # These will be computed in the compute_neighbors method
        
        # Immediate neighborhood (adjacent patches)
        self.r_n_mean = self.g_n_mean = self.b_n_mean = None  # RGB means of neighbors
        self.t_n_mean = None  # Temperature mean of neighbors
        self.z_n_var = None   # Elevation variance of neighbors
        self.z_moins_z_n = None  # Difference between patch elevation and neighbor elevation
                               # (positive values indicate the patch is higher than neighbors)

        # Large neighborhood statistics (patches further away)
        self.r_large_mean = self.g_large_mean = self.b_large_mean = None  # RGB means of large neighborhood
        self.t_large_mean = None  # Temperature mean of large neighborhood
        self.z_moins_z_large = None  # Difference between patch elevation and large neighborhood
    
    def compute_neighbors(self, samples_dict, block_rasters, distance_large=3):
        """
        Compute neighborhood statistics for this sample.
        
        This method calculates statistics of neighboring patches and stores them as features.
        It handles two types of neighborhoods:
        1. Immediate neighbors (8 surrounding patches)
        2. Large neighborhood (patches within a larger distance)
        
        Args:
            samples_dict: Dictionary of samples indexed by (i_x, i_y) coordinates
            block_rasters: BlockRastersManager instance providing access to raster data
            distance_large: Distance (in patches) for the large neighborhood (default: 3)
                           A value of 3 means neighbors up to 3 patches away will be included
        """
        # Get block shape for bounds checking
        block_height, block_width = block_rasters.get_block_shape()
        size_patch = self.size_patch
        has_t = block_rasters.thermal_path is not None  # Check if thermal data is available
        has_z = block_rasters.dsm_path is not None      # Check if elevation data is available

        # --- Immediate Neighborhood (8 adjacent patches) ---
        # Use lists to collect valid values from neighbors
        r_values, g_values, b_values, t_values = [], [], [], []

        # Loop through the 3x3 grid centered on the current patch
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Skip the center (current patch)
                
                # Calculate neighbor indices
                neighbor_i_x = self.i_x + i
                neighbor_i_y = self.i_y + j
                
                # Calculate block-relative coordinates for the neighbor
                x_n = self.x + i * size_patch
                y_n = self.y + j * size_patch
                
                # Try to get the neighbor from the samples dictionary
                neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                if neighbor is not None:
                    # If the neighbor exists in the samples, use its pre-calculated stats
                    if neighbor.r_mean is not None: r_values.append(neighbor.r_mean)
                    if neighbor.g_mean is not None: g_values.append(neighbor.g_mean)
                    if neighbor.b_mean is not None: b_values.append(neighbor.b_mean)
                    if has_t and neighbor.t_mean is not None: t_values.append(neighbor.t_mean)
                elif (
                    0 <= x_n < block_height - size_patch and
                    0 <= y_n < block_width - size_patch
                ):
                    # If the neighbor doesn't exist in samples but is within block bounds,
                    # calculate its statistics directly from the raster data
                    stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                    if stats['r_mean'] is not None: r_values.append(stats['r_mean'])
                    if stats['g_mean'] is not None: g_values.append(stats['g_mean'])
                    if stats['b_mean'] is not None: b_values.append(stats['b_mean'])
                    if has_t and stats['t_mean'] is not None: t_values.append(stats['t_mean'])

        # Calculate neighbor means using numpy for efficiency
        self.r_n_mean = np.mean(r_values) if r_values else None
        self.g_n_mean = np.mean(g_values) if g_values else None
        self.b_n_mean = np.mean(b_values) if b_values else None
        self.t_n_mean = np.mean(t_values) if t_values else None

        # --- Large Neighborhood (patches up to distance_large away) ---
        if distance_large > 1:
            r_large_values, g_large_values, b_large_values, tL_values = [], [], [], []
            
            # Loop through a larger grid centered on the current patch
            for i in range(-distance_large, distance_large + 1):
                for j in range(-distance_large, distance_large + 1):
                    if i == 0 and j == 0:
                        continue  # Skip the center (current patch)
                    
                    # Calculate neighbor indices
                    neighbor_i_x = self.i_x + i
                    neighbor_i_y = self.i_y + j
                    
                    # Calculate block-relative coordinates
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    
                    # Try to get the neighbor from the samples dictionary
                    neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                    if neighbor is not None:
                        # Use pre-calculated stats from existing samples
                        if neighbor.r_mean is not None: r_large_values.append(neighbor.r_mean)
                        if neighbor.g_mean is not None: g_large_values.append(neighbor.g_mean)
                        if neighbor.b_mean is not None: b_large_values.append(neighbor.b_mean)
                        if has_t and neighbor.t_mean is not None: tL_values.append(neighbor.t_mean)
                    elif (
                        0 <= x_n < block_height - size_patch and
                        0 <= y_n < block_width - size_patch
                    ):
                        # Calculate stats directly for neighbors not in samples dictionary
                        stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        if stats['r_mean'] is not None: r_large_values.append(stats['r_mean'])
                        if stats['g_mean'] is not None: g_large_values.append(stats['g_mean'])
                        if stats['b_mean'] is not None: b_large_values.append(stats['b_mean'])
                        if has_t and stats['t_mean'] is not None: tL_values.append(stats['t_mean'])
            
            # Calculate large neighbor means
            self.r_large_mean = np.mean(r_large_values) if r_large_values else None
            self.g_large_mean = np.mean(g_large_values) if g_large_values else None
            self.b_large_mean = np.mean(b_large_values) if b_large_values else None
            self.t_large_mean = np.mean(tL_values) if tL_values else None
        else:
            # No large neighborhood requested, set values to None
            self.r_large_mean = self.g_large_mean = self.b_large_mean = None
            self.t_large_mean = None

        # --- Z (elevation) calculations for immediate neighborhood ---
        if has_z:
            z_values = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue  # Skip the center (current patch)
                    
                    neighbor_i_x = self.i_x + i
                    neighbor_i_y = self.i_y + j
                    
                    # Calculate block-relative coordinates
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    
                    # Collect Z values from neighbors
                    neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                    if neighbor is not None and neighbor.z_mean is not None:
                        z_values.append(neighbor.z_mean)
                    elif (
                        0 <= x_n < block_height - size_patch and
                        0 <= y_n < block_width - size_patch
                    ):
                        stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        if stats['z_mean'] is not None:
                            z_values.append(stats['z_mean'])
            
            # Calculate Z difference between patch and its neighbors
            z_n_mean = np.mean(z_values) if z_values else None
            self.z_moins_z_n = self.z_mean - z_n_mean if z_n_mean is not None and self.z_mean is not None else None
        else:
            self.z_moins_z_n = None

        # --- Z calculations for large neighborhood ---
        if distance_large > 1 and has_z:
            z_large_values = []
            for i in range(-distance_large, distance_large + 1):
                for j in range(-distance_large, distance_large + 1):
                    if i == 0 and j == 0:
                        continue  # Skip the center (current patch)
                    
                    neighbor_i_x = self.i_x + i
                    neighbor_i_y = self.i_y + j
                    
                    # Calculate block-relative coordinates
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    
                    # Collect Z values from large neighborhood
                    neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                    if neighbor is not None and neighbor.z_mean is not None:
                        z_large_values.append(neighbor.z_mean)
                    elif (
                        0 <= x_n < block_height - size_patch and
                        0 <= y_n < block_width - size_patch
                    ):
                        stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        if stats['z_mean'] is not None:
                            z_large_values.append(stats['z_mean'])
            
            # Calculate Z difference between patch and its large neighborhood
            z_large_mean = np.mean(z_large_values) if z_large_values else None
            self.z_moins_z_large = self.z_mean - z_large_mean if z_large_mean is not None and self.z_mean is not None else None
        else:
            self.z_moins_z_large = None
