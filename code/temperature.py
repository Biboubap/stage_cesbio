from osgeo import gdal

# Ouvre les deux rasters
ds_rgb = gdal.Open('rgb.tif')
ds_temp = gdal.Open('temp.tif')

# Récupère les géotransformations
gt_rgb = ds_rgb.GetGeoTransform()
gt_temp = ds_temp.GetGeoTransform()

# Pour chaque patch RGB (i, j)
x_pix_rgb, y_pix_rgb = ... # indices du patch
# Coordonnées géo du centre du patch
x_geo = gt_rgb[0] + (x_pix_rgb + 0.5) * gt_rgb[1]
y_geo = gt_rgb[3] + (y_pix_rgb + 0.5) * gt_rgb[5]

# Indices dans le raster température
col_temp = int((x_geo - gt_temp[0]) / gt_temp[1])
row_temp = int((y_geo - gt_temp[3]) / gt_temp[5])

# Lecture de la température
temp_array = ds_temp.GetRasterBand(1).ReadAsArray()
temperature = temp_array[row_temp, col_temp]