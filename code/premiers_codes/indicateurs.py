from osgeo import gdal
import numpy as np

def compute_gcc_tiff(rgb_path, out_path):
    ds = gdal.Open(rgb_path)
    r = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    g = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    b = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)

    # GCC = G / (R + G + B)
    sum_rgb = r + g + b
    gcc = np.zeros_like(g, dtype=np.float32)
    mask = sum_rgb > 0
    gcc[mask] = g[mask] / sum_rgb[mask]
    gcc[~mask] = np.nan  # ou 0 si tu préfères

    # Sauvegarde en TIFF (float32)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(gcc)
    out_ds.FlushCache()
    out_ds = None
    print(f"GCC sauvegardé dans {out_path}")

if __name__ == "__main__":
    compute_gcc_tiff(
        rgb_path="data/rgb_reshaped.tif",
        out_path="data/gcc_drone.tif"
    )