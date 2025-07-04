import numpy as np

class Sample:
    """
    Simplified version of BlockSample that works directly with the interactive selection tool.
    """
    def __init__(self, i_x, i_y, x, y, size_patch, block_rasters):
        self.i_x = i_x
        self.i_y = i_y
        self.x = x
        self.y = y
        self.size_patch = size_patch
        self.block_rasters = block_rasters
        
        # Calculate statistics for this patch
        stats = block_rasters.calculate_patch_stats(x, y, size_patch)
        
        # RGB statistics
        self.r_mean = stats['r_mean']
        self.g_mean = stats['g_mean']
        self.b_mean = stats['b_mean']
        self.r_var = stats['r_var']
        self.g_var = stats['g_var']
        self.b_var = stats['b_var']
        
        # Z statistics
        self.z_mean = stats['z_mean']
        self.z_var = stats['z_var']
        
        # T statistics
        self.t_mean = stats['t_mean']
        self.t_var = stats['t_var']
        
        # Neighborhood statistics (initialized as None)
        self.r_n_mean = self.g_n_mean = self.b_n_mean = None
        self.t_n_mean = None
        self.z_n_var = None
        self.z_moins_z_n = None
        
        # Large neighborhood statistics
        self.r_large_mean = self.g_large_mean = self.b_large_mean = None
        self.t_large_mean = None
        self.z_moins_z_large = None
    
    def get_RGBZT(self):
        """Get RGB, Z and T data for this sample patch."""
        return self.block_rasters.get_patch(self.x, self.y, self.size_patch)
    
    def compute_neighbors(self, samples_dict, block_rasters, distance_large=3):
        """Compute neighborhood statistics for this sample."""
        # Get block shape for bounds checking
        block_height, block_width = block_rasters.get_block_shape()
        size_patch = self.size_patch
        has_t = block_rasters.thermal_path is not None
        has_z = block_rasters.dsm_path is not None
        
        # --- Immediate neighborhood (8 surrounding patches) ---
        r_values, g_values, b_values, t_values = [], [], [], []
        
        # Loop through immediate neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                
                # Calculate neighbor indices
                neighbor_i_x = self.i_x + i
                neighbor_i_y = self.i_y + j
                
                # Calculate coordinates
                x_n = self.x + i * size_patch
                y_n = self.y + j * size_patch
                
                # Try to get neighbor from samples dictionary
                neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                if neighbor is not None:
                    if neighbor.r_mean is not None: r_values.append(neighbor.r_mean)
                    if neighbor.g_mean is not None: g_values.append(neighbor.g_mean)
                    if neighbor.b_mean is not None: b_values.append(neighbor.b_mean)
                    if has_t and neighbor.t_mean is not None: t_values.append(neighbor.t_mean)
                elif (
                    0 <= x_n < block_width - size_patch and
                    0 <= y_n < block_height - size_patch
                ):
                    # Calculate stats directly from raster data
                    stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                    if stats['r_mean'] is not None: r_values.append(stats['r_mean'])
                    if stats['g_mean'] is not None: g_values.append(stats['g_mean'])
                    if stats['b_mean'] is not None: b_values.append(stats['b_mean'])
                    if has_t and stats['t_mean'] is not None: t_values.append(stats['t_mean'])
        
        # Calculate neighbor means
        self.r_n_mean = np.mean(r_values) if r_values else None
        self.g_n_mean = np.mean(g_values) if g_values else None
        self.b_n_mean = np.mean(b_values) if b_values else None
        self.t_n_mean = np.mean(t_values) if t_values else None
        
        # --- Large neighborhood ---
        if distance_large > 1:
            r_large_values, g_large_values, b_large_values, tL_values = [], [], [], []
            
            for i in range(-distance_large, distance_large + 1):
                for j in range(-distance_large, distance_large + 1):
                    if i == 0 and j == 0:
                        continue
                    
                    # Calculate neighbor indices
                    neighbor_i_x = self.i_x + i
                    neighbor_i_y = self.i_y + j
                    
                    # Calculate coordinates
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    
                    # Try to get neighbor from samples dictionary
                    neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                    if neighbor is not None:
                        if neighbor.r_mean is not None: r_large_values.append(neighbor.r_mean)
                        if neighbor.g_mean is not None: g_large_values.append(neighbor.g_mean)
                        if neighbor.b_mean is not None: b_large_values.append(neighbor.b_mean)
                        if has_t and neighbor.t_mean is not None: tL_values.append(neighbor.t_mean)
                    elif (
                        0 <= x_n < block_width - size_patch and
                        0 <= y_n < block_height - size_patch
                    ):
                        # Calculate stats directly from raster data
                        stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        if stats['r_mean'] is not None: r_large_values.append(stats['r_mean'])
                        if stats['g_mean'] is not None: g_large_values.append(stats['g_mean'])
                        if stats['b_mean'] is not None: b_large_values.append(stats['b_mean'])
                        if has_t and stats['t_mean'] is not None: tL_values.append(stats['t_mean'])
            
            # Calculate large neighborhood means
            self.r_large_mean = np.mean(r_large_values) if r_large_values else None
            self.g_large_mean = np.mean(g_large_values) if g_large_values else None
            self.b_large_mean = np.mean(b_large_values) if b_large_values else None
            self.t_large_mean = np.mean(tL_values) if tL_values else None
        
        # --- Z calculations (elevation differences) ---
        if has_z:
            # Immediate neighborhood Z
            z_values = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    
                    # Calculate coordinates
                    neighbor_i_x = self.i_x + i
                    neighbor_i_y = self.i_y + j
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    
                    # Collect Z values
                    neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                    if neighbor is not None and neighbor.z_mean is not None:
                        z_values.append(neighbor.z_mean)
                    elif (
                        0 <= x_n < block_width - size_patch and
                        0 <= y_n < block_height - size_patch
                    ):
                        stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        if stats['z_mean'] is not None:
                            z_values.append(stats['z_mean'])
            
            # Calculate Z difference
            z_n_mean = np.mean(z_values) if z_values else None
            self.z_moins_z_n = self.z_mean - z_n_mean if self.z_mean is not None and z_n_mean is not None else None
            
            # Large neighborhood Z
            if distance_large > 1:
                z_large_values = []
                for i in range(-distance_large, distance_large + 1):
                    for j in range(-distance_large, distance_large + 1):
                        if i == 0 and j == 0:
                            continue
                        
                        # Calculate coordinates
                        neighbor_i_x = self.i_x + i
                        neighbor_i_y = self.i_y + j
                        x_n = self.x + i * size_patch
                        y_n = self.y + j * size_patch
                        
                        # Collect Z values
                        neighbor = samples_dict.get((neighbor_i_x, neighbor_i_y))
                        if neighbor is not None and neighbor.z_mean is not None:
                            z_large_values.append(neighbor.z_mean)
                        elif (
                            0 <= x_n < block_width - size_patch and
                            0 <= y_n < block_height - size_patch
                        ):
                            stats = block_rasters.calculate_patch_stats(x_n, y_n, size_patch)
                            if stats['z_mean'] is not None:
                                z_large_values.append(stats['z_mean'])
                
                # Calculate Z large difference
                z_large_mean = np.mean(z_large_values) if z_large_values else None
                self.z_moins_z_large = self.z_mean - z_large_mean if self.z_mean is not None and z_large_mean is not None else None
