from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds = gdal.Open(r'data/twin lake mosaïc.tif')

r = ds.GetRasterBand(1).ReadAsArray()[:15000] #SHAPE (49674, 23408)
g = ds.GetRasterBand(2).ReadAsArray()[:15000]
b = ds.GetRasterBand(3).ReadAsArray()[:15000]

size_patch = 32
n_samples = 10
x_samples = np.arange(12800, 12800 + n_samples*size_patch, size_patch)
y = 12032

samples = [(r[x:x + size_patch, y:y + size_patch], g[x:x + size_patch, y:y + size_patch], b[x:x + size_patch, y:y + size_patch])
    for x in x_samples]



def plot_samples(samples):

    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 15))  # 1 row, 2 columns
    for i, (r, g, b) in enumerate(samples):
        rgb = np.dstack((r, g, b))
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"x = {x_samples[i]}, y = {y}")
        axes[i, 0].axis("off")

    
        # Adjust layout and show the figure
        
        #plt.show()

        # Calculate mean and variance for each band
        r_mean, r_var = np.mean(r), np.var(r)
        g_mean, g_var = np.mean(g), np.var(g)
        b_mean, b_var = np.mean(b), np.var(b)
        
        # Add text with mean and variance to the right of the sample
        text = (
            f"R: mean={r_mean:.2f}, var={r_var:.2f}\n"
            f"G: mean={g_mean:.2f}, var={g_var:.2f}\n"
            f"B: mean={b_mean:.2f}, var={b_var:.2f}"
        )
        axes[i, 1].text(0.5, 0.5, text, fontsize=10, ha="left", va="center", wrap=True)
        axes[i, 1].axis("off")

    plt.tight_layout()

if __name__ == "__main__":
    plot_samples(samples)
    plt.show()
    plt.close()