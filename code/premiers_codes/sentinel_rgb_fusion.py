from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt

gdal.UseExceptions()

rasterR = gdal.Open(r'data/sentinel2/bandes/STACK_2023_BandB4_Twin_Lake_V2.tif')
rasterG = gdal.Open(r'data/sentinel2/bandes/STACK_2023_BandB3_Twin_Lake_V2.tif')
rasterB = gdal.Open(r'data/sentinel2/bandes/STACK_2023_BandB2_Twin_Lake_V2.tif')

def to_color(data_band):
    """
    Convert RGB array to color array
    """
    global rasterR, rasterG, rasterB
    r = rasterR.GetRasterBand(data_band).ReadAsArray()
    g = rasterG.GetRasterBand(data_band).ReadAsArray()
    b = rasterB.GetRasterBand(data_band).ReadAsArray()

    """
    Sauvegarde un raster RGB (3 bandes) au format .tif
    r, g, b : tableaux numpy (hauteur, largeur)
    out_path : chemin de sortie
    ref_ds : dataset GDAL de référence pour la géolocalisation
    """

    driver = gdal.GetDriverByName('GTiff')
    out_path = f"data/sentinel2/rgb/RGB_Twin_Lake_databand{data_band}.tif"
    rows, cols = r.shape
    out_ds = driver.Create(out_path, cols, rows, 3, gdal.GDT_UInt16)  # ou GDT_Byte selon le type

    # Copier la géotransformation et la projection du raster de référence
    out_ds.SetGeoTransform(rasterR.GetGeoTransform())
    out_ds.SetProjection(rasterR.GetProjection())

    out_ds.GetRasterBand(1).WriteArray(r)
    out_ds.GetRasterBand(2).WriteArray(g)
    out_ds.GetRasterBand(3).WriteArray(b)

    out_ds.FlushCache()
    out_ds = None  # Fermer le fichier

to_color(1)


