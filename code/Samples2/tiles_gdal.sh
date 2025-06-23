gdal_retile.py -targetDir drone_treated/WAP32_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/WAP32_full_transparent_mosaic_group1.tif 
gdal_retile.py -targetDir drone_treated/WAP32_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/WAP32_full_dsm.tif 
gdalwarp drone_treated/WAP32_tiles/classification/*.tif drone_treated/WAP32_tiles/WAP32_classif.tif 



gdal_retile.py -targetDir drone_treated/WAP23_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/Wap23_main_transparent_mosaic_group1.tif 
gdal_retile.py -targetDir drone_treated/WAP23_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/Wap23_main_dsm.tif 
gdalwarp drone_treated/WAP23_tiles/classification/*.tif drone_treated/WAP23_tiles/WAP23_classif.tif

gdal_retile.py -targetDir drone_treated/WAP12_tiles/rgb -ps 4992 4992 -of GTiff drone_treated/Wap12_Main_transparent_mosaic_group1.tif 
gdal_retile.py -targetDir drone_treated/WAP12_tiles/dsm -ps 4992 4992 -of GTiff drone_treated/Wap12_Main_dsm.tif 


gdal_retile.py -targetDir drone_treated/TL_tiles/rgb -ps 4992 4992 -of GTiff data/twin_lake_mosaïc.tif 
gdal_retile.py -targetDir drone_treated/TL_tiles/dsm -ps 4992 4992 -of GTiff data/twin_lake_dsm.tif 
gdalwarp drone_treated/WAP32_tiles/classification_wap32_5wd/*.tif drone_treated/WAP32_tiles/WAP32_classif_5wd.tif 
gdalwarp drone_treated/WAP23_tiles/classification_wap32_5wd/*.tif drone_treated/WAP23_tiles/WAP32_classif_5wd.tif 
gdalwarp drone_treated/WAP12_tiles/classification_wap32_5wd/*.tif drone_treated/WAP12_tiles/WAP32_classif_5wd.tif 
gdalwarp drone_treated/WAP99_tiles/classification_wap32_5wd/*.tif drone_treated/WAP99_tiles/WAP32_classif_5wd.tif 

mkdir -p Konstantin/Chesnay_tiles_tiles/rgb
mkdir -p Konstantin/Chesnay_tiles/dsm
mkdir -p Konstantin/Chesnay_tiles/thermal
gdal_retile.py -targetDir Konstantin/Chesnay_tiles/rgb -ps 4992 4992 -of GTiff Konstantin/UAV_Konstantin_Tabatha/Chesnay/ChesnayAugust2023_ElevationToolbox_export_MonJun16161652839886_32615.tif
gdal_retile.py -targetDir Konstantin/Chesnay_tiles/dsm -ps 4992 4992 -of GTiff Konstantin/UAV_Konstantin_Tabatha/Chesnay/Chesnay_DSM_Resampled.tif
gdal_retile.py -targetDir Konstantin/Chesnay_tiles/thermal -ps 4992 4992 -of GTiff Konstantin/UAV_Konstantin_Tabatha/Chesnay/ChesnayAugust2023Thermal_RadiometricThermal_export_MonJun16174733351905_32615.tif
gdalwarp Konstantin/Chesnay_tiles/merged_classification/*.tif Konstantin/Chesnay_tiles/merged_classification.tif


mkdir -p Konstantin/Belcher_tiles/rgb
mkdir -p Konstantin/Belcher_tiles/dsm
# mkdir -p Konstantin/Belcher_tiles/thermal
gdal_retile.py -targetDir Konstantin/Belcher_tiles/rgb -ps 4992 4992 -of GTiff Konstantin/UAV_Konstantin_Tabatha/Belcher/BelcherAugust2023_ortho_export_TueJun17212958144821_32615.tif
gdal_retile.py -targetDir Konstantin/Belcher_tiles/dsm -ps 4992 4992 -of GTiff Konstantin/UAV_Konstantin_Tabatha/Belcher/Belcher_DSM_Resampled.tif
# gdal_retile.py -targetDir Konstantin/Belcher_tiles/thermal -ps 4992 4992 -of GTiff Konstantin/UAV_Konstantin_Tabatha/Belcher/BelcherAugust2023Thermal_RadiometricThermal_export_MonJun16174733351905_32615.tif
gdalwarp Konstantin/Belcher_tiles/merged_classification/*.tif Konstantin/Belcher_tiles/merged_classification.tif


## Créer le dossier de sortie s'il n'existe pas
mkdir -p DataCubeS2/WAP32_10m/mediane_indices_10m/

# Traiter chaque fichier .tif
for input_file in DataCubeS2/WAP32_10m/mediane_indices/*.tif; do
    # Extraire le nom du fichier sans le chemin
    filename=$(basename "$input_file")
    
    # Créer un fichier temporaire en supprimant la première ligne de pixels
    temp_file="/tmp/temp_$filename"
    
    # Obtenir les dimensions avec une commande simple
    width=$(gdalinfo $input_file | grep "Size is" | cut -d' ' -f3 | cut -d',' -f1)
    height=$(gdalinfo $input_file | grep "Size is" | cut -d' ' -f4)
    
    echo "Suppression de la première ligne de pixels de $input_file"
    gdal_translate -srcwin 0 1 $width $((height-1)) "$input_file" "$temp_file"
    
    # Rééchantillonner à 10m
    output_file="DataCubeS2/WAP32_10m/mediane_indices_10m/$filename"
    echo "Rééchantillonnage de $temp_file vers $output_file"
    gdalwarp -tr 10.0 10.0 -r average -overwrite "$temp_file" "$output_file"
    
    # Supprimer le fichier temporaire
    rm "$temp_file"
done

mkdir -p DataCubeS2/WAP32_10m/mediane_bands_10m/

# Traiter chaque fichier .tif
for input_file in DataCubeS2/WAP32_10m/mediane_bands/*.tif; do
    # Extraire le nom du fichier sans le chemin
    filename=$(basename "$input_file")
    
    # Créer un fichier temporaire en supprimant la première ligne de pixels
    temp_file="/tmp/$filename"
    
    # Obtenir les dimensions avec une commande simple
    width=$(gdalinfo $input_file | grep "Size is" | cut -d' ' -f3 | cut -d',' -f1)
    height=$(gdalinfo $input_file | grep "Size is" | cut -d' ' -f4)
    
    echo "Suppression de la première ligne de pixels de $input_file"
    gdal_translate -srcwin 0 1 $width $((height-1)) "$input_file" "$temp_file"
    
    Rééchantillonner à 10m
    output_file="DataCubeS2/WAP32_10m/mediane_bands_10m/$filename"
    echo "Rééchantillonnage de $filename vers $output_file"
    gdalwarp -tr 10.0 10.0 -r average -overwrite "$temp_file" "$output_file"

done
