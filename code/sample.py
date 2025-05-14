from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from carac_extract import get_mean_colors_neighbors, get_slope_neighbors

ds = gdal.Open(r'data/twin_lake_mosaïc.tif')
dz = gdal.Open(r'data/twin_lake_dsm.tif')

rast_r = ds.GetRasterBand(1).ReadAsArray() #SHAPE (49674, 23408) 
rast_g = ds.GetRasterBand(2).ReadAsArray()
rast_b = ds.GetRasterBand(3).ReadAsArray()
rast_z = dz.GetRasterBand(1).ReadAsArray()

class Sample:
    global rast_r, rast_g, rast_b, rast_z

    def __init__(self, i_x, i_y, x_start, y_start, size_patch=32, samples=None):
 
        self.size_patch = size_patch
        self.i_x = i_x
        self.i_y = i_y
        self.x_start = x_start 
        self.y_start = y_start 

        # Position réelle (optionnel)
        self.x = self.x_start + i_x * size_patch
        self.y = self.y_start + i_y * size_patch

        # Caractéristiques du patch
        r,g,b,z = get_RGBZ(self.x, self.y, size_patch)

        self.r_mean = float(np.mean(r))
        self.g_mean = float(np.mean(g))
        self.b_mean = float(np.mean(b))

        self.r_var = float(np.var(r))
        self.g_var = float(np.var(g))
        self.b_var = float(np.var(b))

        self.z_mean = float(np.mean(z))
        self.z_var = float(np.var(z))

        # Moyennes RGB des voisins
        self.r_n_mean, self.g_n_mean, self.b_n_mean = 0,0,0 #get_mean_colors_neighbors(samples, i_x, i_y)

        # Gradients d'altitude
        self.delta_z_x, self.delta_z_y = 0,0 #get_slope_neighbors(samples, i_x, i_y)

    def __repr__(self):
            return (f"Sample(x={self.x}, y={self.y}, r_mean={self.r_mean:.2f}, g_mean={self.g_mean:.2f}, "
                    f"b_mean={self.b_mean:.2f}, r_n_mean={self.r_n_mean:.2f}, g_n_mean={self.g_n_mean:.2f}, "
                    f"b_n_mean={self.b_n_mean:.2f}, delta_z_x={self.delta_z_x:.2f}, delta_z_y={self.delta_z_y:.2f})")

def get_RGBZ(x_start, y_start, size_patch=32):
    global rast_r, rast_g, rast_b, rast_z
    """
    Get RGB values from the raster data
    """
    r = rast_r[x_start:x_start + size_patch, y_start:y_start + size_patch]
    g = rast_g[x_start:x_start + size_patch, y_start:y_start + size_patch]
    b = rast_b[x_start:x_start + size_patch, y_start:y_start + size_patch]
    z = rast_z[x_start:x_start + size_patch, y_start:y_start + size_patch]*1000 #en mm
    return r, g, b, z


def create_samples(x_start, y_start, n_samples_x, n_samples_y, size_patch=32):
    """
    Create a list of samples from the raster data
    """
    x_samples = np.arange(x_start, x_start + n_samples_x*size_patch, size_patch)
    y_samples = np.arange(y_start, y_start + n_samples_y*size_patch, size_patch)

    samples = [[Sample(samples, i_x, i_y, x_start, y_start) for i_x in range(n_samples_x)] for i_y in range(n_samples_y)]

    return samples

def plot_sample(sample):
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))  # 1 row, 3 columns

    r,g,b,z = get_RGBZ(sample.x_start, sample.y_start, sample.size_patch)

    rgb = np.dstack((r, g, b))
    axes[0].imshow(rgb)
    axes[0].set_title(f"x = {sample.x_start}, y = {sample.y_start}")
    axes[0].axis("off")

    # Statistiques de la cellule
    cell_text = (
        f"R: mean={sample.r_mean:.2f}, var={sample.r_var:.2f}\n"
        f"G: mean={sample.g_mean:.2f}, var={sample.g_var:.2f}\n"
        f"B: mean={sample.b_mean:.2f}, var={sample.b_var:.2f}\n"
        f"Z: var={sample.z_var:.2f} (mm)"
    )
    axes[1].text(0, 0.5, cell_text, fontsize=10, ha="left", va="center", wrap=True)
    axes[1].axis("off")

    # Statistiques des voisins et gradients
        
    neighbor_text = (
        f"R (neighbors): mean={sample.r_n_mean:.2f}\n"
        f"G (neighbors): mean={sample.g_n_mean:.2f}\n"
        f"B (neighbors): mean={sample.b_n_mean:.2f}\n"
        f"delta_z_x: {sample.delta_z_x:.5f}\n"
        f"delta_z_y: {sample.delta_z_y:.5f}"
    )
    axes[2].text(0, 0.5, neighbor_text, fontsize=10, ha="left", va="center", wrap=True)
    axes[2].axis("off")


if __name__ == "__main__":
    # Parameters
    size_patch = 32
    n_samples_x = 10
    n_samples_y = 3
    x_start = 12800
    y_start = 12000

    sample = Sample([], 0, 0, x_start, y_start, size_patch)
    print(sample)
    plot_sample(sample)
    plt.show()