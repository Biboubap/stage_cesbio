import numpy as np
from osgeo import gdal

def map_temperature_to_rgb(ds_rgb_path, ds_temp_path):
    ds_rgb = gdal.Open(ds_rgb_path)
    ds_temp = gdal.Open(ds_temp_path)
    gt_rgb = ds_rgb.GetGeoTransform()
    gt_temp = ds_temp.GetGeoTransform()
    temp_array = ds_temp.GetRasterBand(1).ReadAsArray()
    x_size = ds_rgb.RasterXSize
    y_size = ds_rgb.RasterYSize

    temp_mapped = np.full((y_size, x_size), np.nan, dtype=np.float32)

    for y_pix_rgb in range(y_size):
        for x_pix_rgb in range(x_size):
            # Coordonnées géo du centre du pixel RGB
            x_geo = gt_rgb[0] + (x_pix_rgb + 0.5) * gt_rgb[1]
            y_geo = gt_rgb[3] + (y_pix_rgb + 0.5) * gt_rgb[5]
            # Indices dans le raster température
            col_temp = int((x_geo - gt_temp[0]) / gt_temp[1])
            row_temp = int((y_geo - gt_temp[3]) / gt_temp[5])
            # Vérifie que l'indice est dans les bornes
            if (0 <= row_temp < temp_array.shape[0]) and (0 <= col_temp < temp_array.shape[1]):
                temp_mapped[y_pix_rgb, x_pix_rgb] = temp_array[row_temp, col_temp]
            # sinon, on laisse NaN

    return temp_mapped


def save_array_as_tif(array, reference_tif_path, output_tif_path):

    ref_ds = gdal.Open(reference_tif_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_tif_path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(array)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

# if __name__ == "__main__":
#     # Exemple d'utilisation :
#     temp_mapped = map_temperature_to_rgb('data/twin_lake_mosaïc.tif', 'data/twinLake_RadiometricThermal_modifié3.tif')
#     save_array_as_tif(temp_mapped, 'data/twin_lake_mosaïc.tif', 'data/temp_resampled.tif')

def test_map_temperature_to_rgb_subset(ds_rgb_path, ds_temp_path, output_tif_path, size=1000):
    ds_rgb = gdal.Open(ds_rgb_path)
    ds_temp = gdal.Open(ds_temp_path)
    gt_rgb = ds_rgb.GetGeoTransform()
    gt_temp = ds_temp.GetGeoTransform()
    temp_array = ds_temp.GetRasterBand(1).ReadAsArray()
    x_size = min(ds_rgb.RasterXSize, size)
    y_size = min(ds_rgb.RasterYSize, size)

    temp_mapped = np.full((y_size, x_size), np.nan, dtype=np.float32)

    for y_pix_rgb in range(y_size):
        for x_pix_rgb in range(x_size):
            x_geo = gt_rgb[0] + (x_pix_rgb + 0.5) * gt_rgb[1]
            y_geo = gt_rgb[3] + (y_pix_rgb + 0.5) * gt_rgb[5]
            col_temp = int((x_geo - gt_temp[0]) / gt_temp[1])
            row_temp = int((y_geo - gt_temp[3]) / gt_temp[5])
            if (0 <= row_temp < temp_array.shape[0]) and (0 <= col_temp < temp_array.shape[1]):
                temp_mapped[y_pix_rgb, x_pix_rgb] = temp_array[row_temp, col_temp]

    # Sauvegarde du sous-raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_tif_path,
        x_size,
        y_size,
        1,
        gdal.GDT_Float32
    )
    # Adapter la géotransformée pour le sous-raster
    new_gt = (
        gt_rgb[0],
        gt_rgb[1],
        gt_rgb[2],
        gt_rgb[3],
        gt_rgb[4],
        gt_rgb[5]
    )
    out_ds.SetGeoTransform(new_gt)
    out_ds.SetProjection(ds_rgb.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(temp_mapped)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    out_ds = None

# Exemple d'utilisation :
if __name__ == "__main__":
    test_map_temperature_to_rgb_subset(
        'data/twin_lake_mosaïc.tif',
        'data/twinLake_RadiometricThermal_modifié3.tif',
        'data/temp_resampled_subset.tif',
        size=1000
    )