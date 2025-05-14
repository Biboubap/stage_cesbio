from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds = gdal.Open(r'data/twin_lake_mosaïc.tif')
dz = gdal.Open(r'data/twin_lake_dsm.tif')

rast_r = ds.GetRasterBand(1).ReadAsArray()[:15000] #SHAPE (49674, 23408) 
rast_g = ds.GetRasterBand(2).ReadAsArray()[:15000]
rast_b = ds.GetRasterBand(3).ReadAsArray()[:15000]
rast_z = dz.GetRasterBand(1).ReadAsArray()[:15000]



size_patch = 32
n_samples_x = 10
n_samples_y = 3
x_start = 12800
y_start = 12000
x_samples = np.arange(x_start, x_start + n_samples_x*size_patch, size_patch)
y_samples = np.arange(y_start, y_start + n_samples_y*size_patch, size_patch)

samples = [[(rast_r[x:x + size_patch, y:y + size_patch], 
             rast_g[x:x + size_patch, y:y + size_patch], 
             rast_b[x:x + size_patch, y:y + size_patch], 
             rast_z[x:x + size_patch, y:y + size_patch]*1000)
            for x in x_samples] for y in y_samples]

def get_mean_colors_neighbors(samples, i_x, i_y, neighborhood_size=1):
    r_mean = 0
    g_mean = 0
    b_mean = 0
    nb_neighbors = 0
    for i in range(-neighborhood_size, neighborhood_size + 1):
        for j in range(-neighborhood_size, neighborhood_size + 1):
            if i == 0 and j == 0:
                continue
            x = i_x + i
            y = i_y + j
            if 0 <= x < n_samples_x and 0 <= y < n_samples_y:
                r, g, b, z = samples[y][x]
                r_mean += np.mean(r)
                g_mean += np.mean(g)
                b_mean += np.mean(b)
                nb_neighbors += 1
    r_mean /= nb_neighbors
    g_mean /= nb_neighbors
    b_mean /= nb_neighbors
    return r_mean, g_mean, b_mean
        
def get_slope_neighbors(samples, i_x, i_y):
    """
    Calcule le gradient d'altitude (delta_x, delta_y) autour du patch (i_x, i_y)
    """
    # Vérification des bords
    n_x = len(samples[0])
    n_y = len(samples)
    
    # Gradient en x
    if i_x == 0 and n_x > 1:
        z_xplus = np.mean(samples[i_y][i_x + 1][3])
        z_center = np.mean(samples[i_y][i_x][3])
        delta_x = z_xplus - z_center
    elif i_x == n_x - 1 and n_x > 1:
        z_xminus = np.mean(samples[i_y][i_x - 1][3])
        z_center = np.mean(samples[i_y][i_x][3])
        delta_x = z_center - z_xminus
    elif 0 < i_x < n_x - 1:
        z_xplus = np.mean(samples[i_y][i_x + 1][3])
        z_xminus = np.mean(samples[i_y][i_x - 1][3])
        delta_x = (z_xplus - z_xminus) / 2
    else:
        delta_x = 0

    # Gradient en y
    if i_y == 0 and n_y > 1:
        z_yplus = np.mean(samples[i_y + 1][i_x][3])
        z_center = np.mean(samples[i_y][i_x][3])
        delta_y = z_yplus - z_center
    elif i_y == n_y - 1 and n_y > 1:
        z_yminus = np.mean(samples[i_y - 1][i_x][3])
        z_center = np.mean(samples[i_y][i_x][3])
        delta_y = z_center - z_yminus
    elif 0 < i_y < n_y - 1:
        z_yplus = np.mean(samples[i_y + 1][i_x][3])
        z_yminus = np.mean(samples[i_y - 1][i_x][3])
        delta_y = (z_yplus - z_yminus) / 2
    else:
        delta_y = 0

    return delta_x, delta_y 


# def get_slope(x, y, delta_l=1):
#     global rast_z, size_patch
#     rows, cols = rast_z.shape
#     pixel_size = delta_l * size_patch
#     # Calculate the gradient of z
#     if 0 + pixel_size <= x <= rows -1 - pixel_size and 0 + pixel_size <= y <= cols -1 - pixel_size:
#         z_up = mean(rast_z[x+pixel_size, y])
#         delta_z_x = (rast_z[x+pixel_size, y] - rast_z[x-pixel_size, y]) /2
#         delta_z_y = (rast_z[x, y+pixel_size] - rast_z[x, y-pixel_size]) /2
#     else :
#         delta_z_x = 0
#         delta_z_y = 0
#     return delta_z_x, delta_z_y

def plot_samples(samples):
    global rast_r, rast_g, rast_b, rast_z, n_samples_x, n_samples_y

    fig, axes = plt.subplots(n_samples_x, 3, figsize=(8, 2.5 * n_samples_x))  # 1 row, 2 columns
    for i, (r, g, b, z) in enumerate(samples[1]):
        rgb = np.dstack((r, g, b))
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"x = {x_samples[i]}, y = {y_samples[1]}")
        axes[i, 0].axis("off")

    
        # Statistiques de la cellule
        r_mean, r_var = np.mean(r), np.var(r)
        g_mean, g_var = np.mean(g), np.var(g)
        b_mean, b_var = np.mean(b), np.var(b)
        z_var = np.var(z)
        cell_text = (
            f"R: mean={r_mean:.2f}, var={r_var:.2f}\n"
            f"G: mean={g_mean:.2f}, var={g_var:.2f}\n"
            f"B: mean={b_mean:.2f}, var={b_var:.2f}\n"
            f"Z: var={z_var:.2f} (mm)"
        )
        axes[i, 1].text(0, 0.5, cell_text, fontsize=10, ha="left", va="center", wrap=True)
        axes[i, 1].axis("off")

        # Statistiques des voisins et gradients
        
        r_n_mean, g_n_mean, b_n_mean = get_mean_colors_neighbors(samples, i, 1)
        delta_z_x, delta_z_y = get_slope_neighbors(samples, i, 1)
        
        neighbor_text = (
            f"R (neighbors): mean={r_n_mean:.2f}\n"
            f"G (neighbors): mean={g_n_mean:.2f}\n"
            f"B (neighbors): mean={b_n_mean:.2f}\n"
            f"delta_z_x: {delta_z_x:.5f}\n"
            f"delta_z_y: {delta_z_y:.5f}"
        )
        axes[i, 2].text(0, 0.5, neighbor_text, fontsize=10, ha="left", va="center", wrap=True)
        axes[i, 2].axis("off")

    #plt.tight_layout()

if __name__ == "__main__":
    plot_samples(samples)
    plt.show()
    plt.close()
    #print(rast_z[12800, 12032:12042])
    # print(get_neighbors(12800, 12032))
    # print(get_neighbors(x_samples[0], y))