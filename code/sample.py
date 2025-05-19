from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
ds = gdal.Open(r'data/rgb_reshaped.tif')
dz = gdal.Open(r'data/dsm_reshaped.tif')
dt = gdal.Open(r'data/thermal_reshaped.tif')
"""

ds = gdal.Open(r'data/twin_lake_mosaïc.tif')
dz = gdal.Open(r'data/twin_lake_dsm.tif')
dt = gdal.Open(r'data/twinLake_Thermal_Resampled.tif')

rast_r = ds.GetRasterBand(1).ReadAsArray()
rast_g = ds.GetRasterBand(2).ReadAsArray()
rast_b = ds.GetRasterBand(3).ReadAsArray()
rast_z = dz.GetRasterBand(1).ReadAsArray()
rast_t = dt.GetRasterBand(1).ReadAsArray()


class Sample:
    global rast_r, rast_g, rast_b, rast_z, rast_t

    def __init__(self, i_x, i_y, x, y, size_patch, category=None, sample_set = None):
 
        self.size_patch = size_patch
        self.i_x = i_x
        self.i_y = i_y
        self.x = x
        self.y = y
        self.category = category

        # Position réelle (optionnel)
       

        # Caractéristiques du patch
        r,g,b,z,t = Sample.get_RGBZT(self)

        self.r_mean = float(np.mean(r))
        self.g_mean = float(np.mean(g))
        self.b_mean = float(np.mean(b))

        self.r_var = float(np.var(r))
        self.g_var = float(np.var(g))
        self.b_var = float(np.var(b))

        self.z_mean = float(np.mean(z))
        self.z_var = float(np.var(z))

        self.t_mean = float(np.mean(t))

        # Moyennes RGB des voisins
        self.r_n_mean, self.g_n_mean, self.b_n_mean = None, None, None
        self.t_n_mean = None
        # Gradients d'altitude
        self.z_moins_z_n = None

    def __repr__(self):
        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"
        return (f"Sample(x={self.x}, y={self.y}, "
                f"r_mean={fmt(self.r_mean)}, g_mean={fmt(self.g_mean)}, b_mean={fmt(self.b_mean)}, "
                #f"r_var={fmt(self.r_var)}, g_var={fmt(self.g_var)}, b_var={fmt(self.b_var)}, "
                f"z_mean={fmt(self.z_mean)}, z_var={fmt(self.z_var)}, t_mean={fmt(self.t_mean)}, "
                f"r_n_mean={fmt(self.r_n_mean)}, g_n_mean={fmt(self.g_n_mean)}, b_n_mean={fmt(self.b_n_mean)}, t_n_mean={fmt(self.t_n_mean)}, "
                f"z_moins_z_n={fmt(self.z_moins_z_n)}")

    def get_RGBZ(self):
        global rast_r, rast_g, rast_b, rast_z
        """
        Get RGB values from the raster data
        """
        x = self.x
        y = self.y
        size_patch = self.size_patch
        r = rast_r[x:x + size_patch, y:y + size_patch]
        g = rast_g[x:x + size_patch, y:y + size_patch]
        b = rast_b[x:x + size_patch, y:y + size_patch]
        z = rast_z[x:x + size_patch, y:y + size_patch]*1000 #en mm
        return r, g, b, z

    def get_RGBZT(self):
        global rast_r, rast_g, rast_b, rast_z
        """
        Get RGB values from the raster data
        """
        x = self.x
        y = self.y
        size_patch = self.size_patch
        r = rast_r[x:x + size_patch, y:y + size_patch]
        g = rast_g[x:x + size_patch, y:y + size_patch]
        b = rast_b[x:x + size_patch, y:y + size_patch]
        z = rast_z[x:x + size_patch, y:y + size_patch]*1000 #en mm
        t  = rast_t[x:x + size_patch, y:y + size_patch]
        return r, g, b, z, t

    def plot_sample(self):
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
            #f"r_var={fmt2(self.r_var)}, g_var={fmt2(self.g_var)}, b_var={fmt2(self.b_var)}, "
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

    def compute_neighbors_color(self, sample_set=None, size_patch=None, depth_neighbors=1):
        """
        Calcule la moyenne des couleurs des voisins.
        Si un voisin existe dans sample_set, utilise sa moyenne déjà calculée.
        Sinon, calcule à partir du raster.
        """
        global rast_r, rast_g, rast_b
        if size_patch is None:
            size_patch = self.size_patch
        r_sum = 0
        g_sum = 0
        b_sum = 0
        nb_neighbors = 0
        for i in range(-depth_neighbors, depth_neighbors + 1):
            for j in range(-depth_neighbors, depth_neighbors + 1):
                if i == 0 and j == 0:
                    continue
                x_n = self.x + i * size_patch
                y_n = self.y + j * size_patch
                if sample_set is not None and (x_n, y_n) in sample_set.samples:
                    neighbor = sample_set.samples[(x_n, y_n)]
                    r_sum += neighbor.r_mean
                    g_sum += neighbor.g_mean
                    b_sum += neighbor.b_mean
                    nb_neighbors += 1
                elif 0 <= x_n < rast_r.shape[0] - size_patch and 0 <= y_n < rast_r.shape[1] - size_patch:
                    r_patch = rast_r[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    g_patch = rast_g[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    b_patch = rast_b[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    r_sum += np.mean(r_patch)
                    g_sum += np.mean(g_patch)
                    b_sum += np.mean(b_patch)
                    nb_neighbors += 1
        if nb_neighbors > 0:
            self.r_n_mean = r_sum / nb_neighbors
            self.g_n_mean = g_sum / nb_neighbors
            self.b_n_mean = b_sum / nb_neighbors
        else:
            self.r_n_mean = self.g_n_mean = self.b_n_mean = 0

    def compute_neighbors_all(self, sample_set=None, size_patch=None, depth_neighbors=1):
        """
        Calcule la moyenne des couleurs des voisins.
        Si un voisin existe dans sample_set, utilise sa moyenne déjà calculée.
        Sinon, calcule à partir du raster.
        """
        global rast_r, rast_g, rast_b, rast_z, rast_t
        if size_patch is None:
            size_patch = self.size_patch
        r_sum = 0
        g_sum = 0
        b_sum = 0
        z_sum = 0
        t_sum = 0
        nb_neighbors = 0
        for i in range(-depth_neighbors, depth_neighbors + 1):
            for j in range(-depth_neighbors, depth_neighbors + 1):
                if i == 0 and j == 0:
                    continue
                x_n = self.x + i * size_patch
                y_n = self.y + j * size_patch
                if sample_set is not None and (x_n, y_n) in sample_set.samples:
                    neighbor = sample_set.samples[(x_n, y_n)]
                    r_sum += neighbor.r_mean
                    g_sum += neighbor.g_mean
                    b_sum += neighbor.b_mean
                    z_sum += neighbor.z_mean
                    t_sum += neighbor.t_mean
                    nb_neighbors += 1
                elif 0 <= x_n < rast_r.shape[0] - size_patch and 0 <= y_n < rast_r.shape[1] - size_patch:
                    r_patch = rast_r[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    g_patch = rast_g[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    b_patch = rast_b[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    z_patch = rast_z[x_n:x_n + size_patch, y_n:y_n + size_patch]*1000
                    t_patch = rast_t[x_n:x_n + size_patch, y_n:y_n + size_patch]
                    r_sum += np.mean(r_patch)
                    g_sum += np.mean(g_patch)
                    b_sum += np.mean(b_patch)
                    z_sum += np.mean(z_patch)
                    t_sum += np.mean(t_patch)
                    nb_neighbors += 1
        if nb_neighbors > 0:
            self.r_n_mean = r_sum / nb_neighbors
            self.g_n_mean = g_sum / nb_neighbors
            self.b_n_mean = b_sum / nb_neighbors
            self.z_moins_z_n= self.z_mean - z_sum / nb_neighbors
            self.t_n_mean = t_sum / nb_neighbors
        else:
            self.r_n_mean = self.g_n_mean = self.b_n_mean = 0

    def get_z_mean(self, x, y, sample_set=None, size_patch=None):
        global rast_z
        if sample_set is not None and (x, y) in sample_set.samples:
            return sample_set.samples[(x, y)].z_mean
        elif 0 <= x < rast_z.shape[0] - size_patch and 0 <= y < rast_z.shape[1] - size_patch:
            z_patch = rast_z[x:x + size_patch, y:y + size_patch]*1000 #en mm
            return float(np.mean(z_patch))
        else:
            return None
            
    def compute_slope(self, sample_set=None, size_patch=None, depth_neighbors=1):
        """
        Calcule le gradient d'altitude (delta_z_x, delta_z_y) pour ce sample.
        - delta_z_x : (z voisin avant x - z voisin après x) / 2 (ou /1 si en bord)
        - delta_z_y : (z voisin avant y - z voisin après y) / 2*(ou /1 si en bord)
        Si un voisin existe dans sample_set, utilise sa moyenne déjà calculée, sinon le calcule à partir du raster.
        """
        global rast_z
        if size_patch is None:
            size_patch = self.size_patch

        distance = size_patch * depth_neighbors
        # Coordonnées des voisins
        neighbors = {
            "x_prev": (self.x - distance, self.y),
            "x_next": (self.x + distance, self.y),
            "y_prev": (self.x, self.y - distance),
            "y_next": (self.x, self.y + distance),
        }

        # Fonction pour récupérer la moyenne z d'un voisin

        # Calcul pour x
        z_prev_x = self.get_z_mean(*neighbors["x_prev"], sample_set, size_patch)
        z_next_x = self.get_z_mean(*neighbors["x_next"], sample_set, size_patch)
        if z_prev_x is not None and z_next_x is not None:
            self.delta_z_x = (z_next_x - z_prev_x) / (2 * depth_neighbors) 
        elif z_prev_x is not None:
            self.delta_z_x = (self.z_mean - z_prev_x) / (1 * depth_neighbors)
        elif z_next_x is not None:
            self.delta_z_x = (z_next_x - self.z_mean) / (1 * depth_neighbors)
        else:
            self.delta_z_x = None

        # Calcul pour y
        z_prev_y = self.get_z_mean(*neighbors["y_prev"], sample_set, size_patch)
        z_next_y = self.get_z_mean(*neighbors["y_next"], sample_set, size_patch)
        if z_prev_y is not None and z_next_y is not None:
            self.delta_z_y = (z_next_y - z_prev_y) / (2 * depth_neighbors) 
        elif z_prev_y is not None:
            self.delta_z_y = (self.z_mean - z_prev_y) / (1 * depth_neighbors)
        elif z_next_y is not None:
            self.delta_z_y = (z_next_y - self.z_mean) / (1 * depth_neighbors)
        else:
            self.delta_z_y = None

    

if __name__ == "__main__":
    # Parameters
    size_patch = 32
    n_samples_x = 10
    n_samples_y = 3
    x_start = 12800
    y_start = 12000

    sample = Sample(0, 0, x_start, y_start, size_patch)
    print(sample)
    sample.plot_sample()
    