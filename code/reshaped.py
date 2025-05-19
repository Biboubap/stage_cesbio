from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt


ds = gdal.Open(r'data/rgb_reshaped.tif')
dz = gdal.Open(r'data/dsm_reshaped.tif')
dt = gdal.Open(r'data/thermal_reshaped.tif')

r = ds.GetRasterBand(1).ReadAsArray()
g = ds.GetRasterBand(2).ReadAsArray()
b = ds.GetRasterBand(3).ReadAsArray()

def plot_shapes():
    global r, g, b
    print(f'r.shape = {r.shape}')
    print(f'g.shape = {g.shape}')
    print(f'b.shape = {b.shape}')

def plot_rgb():
    rgb = np.dstack((r, g, b))
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.show()

def plot_dsm():
    dsm = dz.GetRasterBand(1).ReadAsArray()
    plt.figure(figsize=(10, 10))
    plt.imshow(dsm[1000:2000, 1000:2000], cmap='gray')
    plt.title("DSM Image")
    plt.show()

def plot_thermal():
    thermal = dt.GetRasterBand(1).ReadAsArray()
    plt.figure(figsize=(10, 10))
    plt.imshow(thermal[1000:2000, 1000:2000], cmap='gray')
    plt.title("Thermal Image")
    plt.show()

if __name__ == "__main__":
    #print(b[10,0]) #t = -32767.0 ; z = -10000 ; r,g,b = 0
    # plot_shapes()
    # plot_rgb()
    # plot_dsm()
    # plot_thermal()


