gdal_retile.py -targetDir drone_treated/WAP32_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/WAP32_full_transparent_mosaic_group1.tif 
gdal_retile.py -targetDir drone_treated/WAP32_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/WAP32_full_dsm.tif 
gdalwarp drone_treated/WAP32_tiles/classification/*.tif drone_treated/WAP32_tiles/WAP32_classif.tif 



gdal_retile.py -targetDir drone_treated/WAP23_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/Wap23_main_transparent_mosaic_group1.tif 
gdal_retile.py -targetDir drone_treated/WAP23_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/Wap23_main_dsm.tif 
gdalwarp drone_treated/WAP23_tiles/classification/*.tif drone_treated/WAP23_tiles/WAP23_classif.tif

gdal_retile.py -targetDir drone_treated/WAP12_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/Wap12_Main_transparent_mosaic_group1.tif 
gdal_retile.py -targetDir drone_treated/WAP12_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/Wap12_Main_dsm.tif 