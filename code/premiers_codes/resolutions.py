from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt

gdal.UseExceptions()

visible = gdal.Open(r'data/twin lake mosaïc.tif')
thermique = gdal.Open(r'data/twinLake_RadiometricThermal_modifié3.tif')
dsm = gdal.Open(r'data/twin lake dsm.tif')

r = visible.GetRasterBand(1).ReadAsArray()
g = visible.GetRasterBand(2).ReadAsArray()
b = visible.GetRasterBand(3).ReadAsArray()

t = thermique.GetRasterBand(1).ReadAsArray()
s = dsm.GetRasterBand(1).ReadAsArray()

def plot_shapes():
    global r, g, b, t, s
    print(f'r.shape = {r.shape}')
    print(f'g.shape = {g.shape}')
    print(f'b.shape = {b.shape}')
    print(f't.shape = {t.shape}')
    print(f's.shape = {s.shape}')

def plot_band():
    plt.figure()
    plt.imshow(r[100:500, 1000:1010])
    plt.axis("off")
    plt.show()

plot_band()
