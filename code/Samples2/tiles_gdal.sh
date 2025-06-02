gdal_retile.py -targetDir drone_treated/WAP32_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/WAP32_full_transparent_mosaic_group1.tif 

gdal_retile.py -targetDir drone_treated/WAP32_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/WAP32_full_dsm.tif 

gdalwarp ./drone_treated/WAP32_tiles/rgb/*.tif ./drone_treated/WAP32_tiles/Wap32_rgb_mosaic.tif 