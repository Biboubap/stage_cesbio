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
        Si un voisin existe dans sample_set, utilise ses statistiques déjà calculées.
        Sinon, extrait et calcule la moyenne directement depuis les rasters.
        """
        # Ensure rasters are loaded
        self.rasters.load_rasters()
        
        # Get raster shapes for bounds checking
        rast_shape_r = self.rasters.rast_r.shape if self.rasters.rast_r is not None else (0, 0)
        rast_shape_z = self.rasters.rast_z.shape if self.rasters.rast_z is not None else (0, 0)
        size_patch = self.size_patch
        has_t = self.rasters.rast_t is not None
        has_z = self.rasters.rast_z is not None

        # --- Voisinage 1 ---
        r_sum = g_sum = b_sum = t_sum = 0
        nb_neighbors = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x_n = self.x + i * size_patch
                y_n = self.y + j * size_patch
                neighbor = sample_set.samples.get((x_n, y_n))
                if neighbor is not None:
                    r_sum += neighbor.r_mean
                    g_sum += neighbor.g_mean
                    b_sum += neighbor.b_mean
                    if has_t:
                        t_sum += neighbor.t_mean
                    nb_neighbors += 1
                elif (
                    0 <= x_n < rast_shape_r[0] - size_patch and
                    0 <= y_n < rast_shape_r[1] - size_patch
                ):
                    # Calculer directement les statistiques au lieu de charger tout le patch
                    stats = self.rasters.calculate_patch_stats(x_n, y_n, size_patch)
                    r_sum += stats['r_mean']
                    g_sum += stats['g_mean']
                    b_sum += stats['b_mean']
                    if has_t and stats['t_mean'] is not None:
                        t_sum += stats['t_mean']
                    nb_neighbors += 1

        # Calculate neighbor means if neighbors exist
        if nb_neighbors > 0:
            self.r_n_mean = r_sum / nb_neighbors
            self.g_n_mean = g_sum / nb_neighbors
            self.b_n_mean = b_sum / nb_neighbors
            if has_t:
                self.t_n_mean = t_sum / nb_neighbors
        else:
            self.r_n_mean = self.g_n_mean = self.b_n_mean = None
            self.t_n_mean = None

        if distance_large > 1:
            # --- Voisinage large (distance_large) ---
            r_large_sum = g_large_sum = b_large_sum = tL_sum = 0
            nb_neighborsL = 0
            for i in range(-distance_large, distance_large + 1):
                for j in range(-distance_large, distance_large + 1):
                    if i == 0 and j == 0:
                        continue
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    neighbor = sample_set.samples.get((x_n, y_n))
                    if neighbor is not None:
                        r_large_sum += neighbor.r_mean
                        g_large_sum += neighbor.g_mean
                        b_large_sum += neighbor.b_mean
                        if has_t:
                            tL_sum += neighbor.t_mean
                        nb_neighborsL += 1
                    elif (
                        0 <= x_n < rast_shape_r[0] - size_patch and
                        0 <= y_n < rast_shape_r[1] - size_patch
                    ):
                        # Calculer directement les statistiques au lieu de charger tout le patch
                        stats = self.rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        r_large_sum += stats['r_mean']
                        g_large_sum += stats['g_mean']
                        b_large_sum += stats['b_mean']
                        if has_t and stats['t_mean'] is not None:
                            tL_sum += stats['t_mean']
                        nb_neighborsL += 1
            
            # Calculate large neighbor means if neighbors exist
            if nb_neighborsL > 0:
                self.r_large_mean = r_large_sum / nb_neighborsL
                self.g_large_mean = g_large_sum / nb_neighborsL
                self.b_large_mean = b_large_sum / nb_neighborsL
                if has_t:
                    self.t_large_mean = tL_sum / nb_neighborsL
                else:
                    self.t_large_mean = None
            else:
                self.r_large_mean = self.g_large_mean = self.b_large_mean = None
                self.t_large_mean = None

        # --- Z voisinage 1 ---
        if has_z:
            z_sum = 0
            nb_neighbors_z = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    x_n = self.x + i * size_patch
                    y_n = self.y + j * size_patch
                    neighbor = sample_set.samples.get((x_n, y_n))
                    if neighbor is not None and neighbor.z_mean is not None:
                        z_sum += neighbor.z_mean
                        nb_neighbors_z += 1
                    elif (
                        0 <= x_n < rast_shape_z[0] - size_patch and
                        0 <= y_n < rast_shape_z[1] - size_patch
                    ):
                        # Calculer directement la moyenne Z au lieu de charger tout le patch
                        stats = self.rasters.calculate_patch_stats(x_n, y_n, size_patch)
                        if stats['z_mean'] is not None:
                            z_sum += stats['z_mean']
                            nb_neighbors_z += 1
            
            if nb_neighbors_z > 0:
                self.z_moins_z_n = self.z_mean - z_sum / nb_neighbors_z
            else:
                self.z_moins_z_n = None
        else:
            self.z_moins_z_n = None

        if distance_large > 1:
            # --- Z voisinage large ---
            if has_z:
                z_large_sum = 0
                nb_neighbors_z_large = 0
                for i in range(-distance_large, distance_large + 1):
                    for j in range(-distance_large, distance_large + 1):
                        if i == 0 and j == 0:
                            continue
                        x_n = self.x + i * size_patch
                        y_n = self.y + j * size_patch
                        neighbor = sample_set.samples.get((x_n, y_n))
                        if neighbor is not None and neighbor.z_mean is not None:
                            z_large_sum += neighbor.z_mean
                            nb_neighbors_z_large += 1
                        elif (
                            0 <= x_n < rast_shape_z[0] - size_patch and
                            0 <= y_n < rast_shape_z[1] - size_patch
                        ):
                            # Calculer directement la moyenne Z au lieu de charger tout le patch
                            stats = self.rasters.calculate_patch_stats(x_n, y_n, size_patch)
                            if stats['z_mean'] is not None:
                                z_large_sum += stats['z_mean']
                                nb_neighbors_z_large += 1
                
                if nb_neighbors_z_large > 0:
                    z_large_mean = z_large_sum / nb_neighbors_z_large
                    self.z_moins_z_large = self.z_mean - z_large_mean
                else:
                    self.z_moins_z_large = None
            else:
                self.z_moins_z_large = None

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
