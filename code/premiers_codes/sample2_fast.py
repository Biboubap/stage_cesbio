import matplotlib.pyplot as plt
import numpy as np
from rasters_manager import RastersManager

class Sample2:
    
    """
    Représente un patch d'image et ses statistiques (moyennes, variances, voisinages, etc.)
    Permet d'extraire les valeurs RGB, Z, T et de calculer les statistiques des voisins.
    """
    # Class-level reference to the rasters manager
    rasters = RastersManager()
    
    def __init__(self, i_x, i_y, x, y, size_patch, category=None):
        # Indices et position du patch
        self.size_patch = size_patch
        self.i_x = i_x
        self.i_y = i_y
        self.x = x
        self.y = y
        self.category = category

        # Ensure rasters are loaded
        self.rasters.load_rasters()
        
        # Calculate statistics directly without storing patches
        stats = self.rasters.calculate_patch_stats(x, y, size_patch)
        
        # RGB statistics
        self.r_mean = stats['r_mean']
        self.g_mean = stats['g_mean']
        self.b_mean = stats['b_mean']
        self.r_var = stats['r_var']
        self.g_var = stats['g_var']
        self.b_var = stats['b_var']

        # Altitude (z)
        self.z_mean = stats['z_mean']
        self.z_var = stats['z_var']

        # Température (t)
        self.t_mean = stats['t_mean']
        self.t_var = stats['t_var']

        # Statistiques des voisins (initialisées à None)
        self.r_n_mean = self.g_n_mean = self.b_n_mean = None
        self.t_n_mean = None
        self.z_n_var = None
        self.z_moins_z_n = None

        # # Statistiques des voisins à grande distance (large)
        self.r_large_mean = self.g_large_mean = self.b_large_mean = None
        self.t_large_mean = None
        self.z_moins_z_large = None

    def get_RGBZT(self):
        """
        Retourne les patchs RGB, Z, T pour ce sample.
        """
        return self.rasters.get_patch(self.x, self.y, self.size_patch)
        
    def compute_neighbors_all(self, sample_set, distance_large=3):
        """
        Calcule les statistiques des voisins (moyennes RGB, T, Z) pour ce sample.
        Optimisé pour performance et réduction de mémoire.
        """
        # Ensure rasters are loaded
        self.rasters.load_rasters()
        
        # Get raster shapes for bounds checking
        rast_shape_r = self.rasters.rast_r.shape if self.rasters.rast_r is not None else (0, 0)
        rast_shape_z = self.rasters.rast_z.shape if self.rasters.rast_z is not None else (0, 0)
        size_patch = self.size_patch
        has_t = self.rasters.rast_t is not None
        has_z = self.rasters.rast_z is not None

        # Optimized: Pre-calculate neighbor coordinates for reuse
        neighbors_coords = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbors_coords.append((i, j))

        # --- Calculate statistics for close neighborhood (1) ---
        # Collect coordinates of all neighbors for batch processing
        close_neighbor_positions = []
        close_existing_neighbors = []
        
        for i, j in neighbors_coords:
            x_n = self.x + i * size_patch
            y_n = self.y + j * size_patch
            neighbor = sample_set.samples.get((x_n, y_n))
            if neighbor is not None:
                close_existing_neighbors.append(neighbor)
            elif (0 <= x_n < rast_shape_r[0] - size_patch and
                  0 <= y_n < rast_shape_r[1] - size_patch):
                close_neighbor_positions.append((x_n, y_n))
        
        # Get stats for neighbors in batch if any positions to calculate
        close_neighbor_stats = []
        if close_neighbor_positions:
            close_neighbor_stats = self.rasters.calculate_patches_stats_batch(close_neighbor_positions, size_patch)
        
        # Combine statistics from existing neighbors and calculated ones
        r_values = [n.r_mean for n in close_existing_neighbors if n.r_mean is not None]
        g_values = [n.g_mean for n in close_existing_neighbors if n.g_mean is not None]
        b_values = [n.b_mean for n in close_existing_neighbors if n.b_mean is not None]
        t_values = [n.t_mean for n in close_existing_neighbors if has_t and n.t_mean is not None]
        z_values = [n.z_mean for n in close_existing_neighbors if has_z and n.z_mean is not None]
        
        # Add calculated stats
        for stats in close_neighbor_stats:
            if stats['r_mean'] is not None: r_values.append(stats['r_mean'])
            if stats['g_mean'] is not None: g_values.append(stats['g_mean'])
            if stats['b_mean'] is not None: b_values.append(stats['b_mean'])
            if has_t and stats['t_mean'] is not None: t_values.append(stats['t_mean'])
            if has_z and stats['z_mean'] is not None: z_values.append(stats['z_mean'])
        
        # Calculate means using numpy's optimized functions
        self.r_n_mean = np.mean(r_values) if r_values else None
        self.g_n_mean = np.mean(g_values) if g_values else None
        self.b_n_mean = np.mean(b_values) if b_values else None
        self.t_n_mean = np.mean(t_values) if t_values else None
        
        # Z difference calculation
        z_n_mean = np.mean(z_values) if z_values and has_z else None
        self.z_moins_z_n = self.z_mean - z_n_mean if z_n_mean is not None and self.z_mean is not None else None

        # --- Calculate large neighborhood statistics if needed ---
        if distance_large > 1:
            # Pre-calculate all large neighborhood coordinates
            large_neighbors_coords = []
            for i in range(-distance_large, distance_large + 1):
                for j in range(-distance_large, distance_large + 1):
                    if i == 0 and j == 0:
                        continue
                    large_neighbors_coords.append((i, j))
            
            # Collect coordinates of all large neighbors for batch processing
            large_neighbor_positions = []
            large_existing_neighbors = []
            
            for i, j in large_neighbors_coords:
                x_n = self.x + i * size_patch
                y_n = self.y + j * size_patch
                neighbor = sample_set.samples.get((x_n, y_n))
                if neighbor is not None:
                    large_existing_neighbors.append(neighbor)
                elif (0 <= x_n < rast_shape_r[0] - size_patch and
                      0 <= y_n < rast_shape_r[1] - size_patch):
                    large_neighbor_positions.append((x_n, y_n))
            
            # Get stats for large neighbors in batch
            large_neighbor_stats = []
            if large_neighbor_positions:
                large_neighbor_stats = self.rasters.calculate_patches_stats_batch(large_neighbor_positions, size_patch)
            
            # Combine statistics
            r_large_values = [n.r_mean for n in large_existing_neighbors if n.r_mean is not None]
            g_large_values = [n.g_mean for n in large_existing_neighbors if n.g_mean is not None]
            b_large_values = [n.b_mean for n in large_existing_neighbors if n.b_mean is not None]
            t_large_values = [n.t_mean for n in large_existing_neighbors if has_t and n.t_mean is not None]
            z_large_values = [n.z_mean for n in large_existing_neighbors if has_z and n.z_mean is not None]
            
            # Add calculated stats
            for stats in large_neighbor_stats:
                if stats['r_mean'] is not None: r_large_values.append(stats['r_mean'])
                if stats['g_mean'] is not None: g_large_values.append(stats['g_mean'])
                if stats['b_mean'] is not None: b_large_values.append(stats['b_mean'])
                if has_t and stats['t_mean'] is not None: t_large_values.append(stats['t_mean'])
                if has_z and stats['z_mean'] is not None: z_large_values.append(stats['z_mean'])
            
            # Calculate means
            self.r_large_mean = np.mean(r_large_values) if r_large_values else None
            self.g_large_mean = np.mean(g_large_values) if g_large_values else None
            self.b_large_mean = np.mean(b_large_values) if b_large_values else None
            self.t_large_mean = np.mean(t_large_values) if t_large_values else None
            
            # Z difference calculation for large neighborhood
            z_large_mean = np.mean(z_large_values) if z_large_values and has_z else None
            self.z_moins_z_large = self.z_mean - z_large_mean if z_large_mean is not None and self.z_mean is not None else None
        else:
            self.r_large_mean = self.g_large_mean = self.b_large_mean = None
            self.t_large_mean = None
            self.z_moins_z_large = None
    
    @staticmethod
    def compute_neighbors_batch(samples_list, sample_set, distance_large=3):
        """
        Process neighborhood statistics for multiple samples in one batch.
        More efficient than processing each sample individually.
        """
        for sample in samples_list:
            sample.compute_neighbors_all(sample_set, distance_large)

    def __repr__(self):
        # Affichage lisible des principales statistiques du sample
        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"
        return (f"Sample2(x={self.x}, y={self.y}, "
                f"r_mean={fmt(self.r_mean)}, g_mean={fmt(self.g_mean)}, b_mean={fmt(self.b_mean)}, "
                f"z_mean={fmt(self.z_mean)}, z_var={fmt(self.z_var)}, t_mean={fmt(self.t_mean)}, "
                f"r_n_mean={fmt(self.r_n_mean)}, g_n_mean={fmt(self.g_n_mean)}, b_n_mean={fmt(self.b_n_mean)}, t_n_mean={fmt(self.t_n_mean)}, "
                f"z_moins_z_n={fmt(self.z_moins_z_n)}, z_moins_z_large={fmt(self.z_moins_z_large)}"
        )
    
    def plot_sample(self):
        """
        Affiche le patch RGB, ses statistiques et celles de ses voisins.
        """
        def fmt2(val):
            return f"{val:.2f}" if val is not None else "None"
        def fmt5(val):
            return f"{val:.5f}" if val is not None else "None"
        
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))  # 1 row, 3 columns

        r,g,b,z, t = self.get_RGBZT()

        rgb = np.dstack((r, g, b))
        axes[0].imshow(rgb)
        axes[0].set_title(f"x = {self.x}, y = {self.y}")
        axes[0].axis("off")

        # Statistiques de la cellule
        cell_text = (
            f"Sample(x={self.x}, y={self.y}, "
            f"r_mean={fmt2(self.r_mean)}, g_mean={fmt2(self.g_mean)}, b_mean={fmt2(self.b_mean)}, "
            f"z_mean={fmt2(self.z_mean)}, z_var={fmt5(self.z_var)}, t_mean={fmt2(self.t_mean)}, "
        )
        axes[1].text(0, 0.5, cell_text, fontsize=10, ha="left", va="center", wrap=True)
        axes[1].axis("off")

        # Statistiques des voisins et gradients
        neighbor_text = (
            f"R (neighbors): mean={fmt2(self.r_n_mean)}\n"
            f"G (neighbors): mean={fmt2(self.g_n_mean)}\n"
            f"B (neighbors): mean={fmt2(self.b_n_mean)}\n"
            f"T (neighbors): mean={fmt2(self.t_n_mean)}\n"
            f"z_moins_z_n: {fmt5(self.z_moins_z_n)}"
        )
        axes[2].text(0, 0.5, neighbor_text, fontsize=10, ha="left", va="center", wrap=True)
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    

if __name__ == "__main__":
    # Initialize the RastersManager
    rasters_mgr = RastersManager()
    
    # Set paths to raster files
    rasters_mgr.set_paths(
        ds_path=r'data/rgb_reshaped.tif',
        dz_path=r'data/dsm_reshaped.tif',
        dt_path=r'data/thermal_reshaped.tif'
    )
    
    # Create a sample
    sample = Sample2(i_x=0, i_y=0, x=1000, y=1000, size_patch=128)

    print(sample)
    sample.plot_sample()
    print(sample)
    sample.plot_sample()
