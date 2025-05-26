from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt


dt = gdal.Open(r'data/twinLake_Thermal_Resampled.tif')
ds = gdal.Open(r'data/twin_lake_mosaïc.tif')

t = dt.GetRasterBand(1).ReadAsArray()
r = ds.GetRasterBand(1).ReadAsArray()
g = ds.GetRasterBand(2).ReadAsArray()
b = ds.GetRasterBand(3).ReadAsArray()
                        


def plot_t(t, xmin = 10000, ymin = 10000, l = 1000):
   
    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(t[xmin:xmin+l, ymin:ymin + l])
    plt.title("Temperature Image")
    plt.axis("off")
    plt.show()

def plot_rgb_and_thermal(r, g, b, t, xmin=10000, ymin=10000, l=2000):
    rgb = np.dstack((r, g, b))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb[xmin:xmin+l, ymin:ymin + l])
    plt.title("Image RGB")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(t[xmin:xmin+l, ymin:ymin + l], cmap='inferno')
    plt.title("Image Thermique")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Plot the thermal image
    plot_rgb_and_thermal(r,g,b,t, xmin=5000, ymin=10000)

   
