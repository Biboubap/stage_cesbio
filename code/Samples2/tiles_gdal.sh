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

## Créer le dossier de sortie s'il n'existe pas
mkdir -p IndicesS22023_WAP32/mediane_10m/

# Traiter chaque fichier .tif
for input_file in IndicesS22023_WAP32/mediane/*.tif; do
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
    output_file="IndicesS22023_WAP32/mediane_10m/10m_$filename"
    echo "Rééchantillonnage de $temp_file vers $output_file"
    gdalwarp -tr 10.0 10.0 -r average -overwrite "$temp_file" "$output_file"
    
    # Supprimer le fichier temporaire
    rm "$temp_file"
done

mkdir -p BandsS22023_WAP32/mediane_10m/

# Traiter chaque fichier .tif
for input_file in BandsS22023_WAP32/mediane/*.tif; do
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
    output_file="BandsS22023_WAP32/mediane_10m/10m_$filename"
    echo "Rééchantillonnage de $temp_file vers $output_file"
    gdalwarp -tr 10.0 10.0 -r average -overwrite "$temp_file" "$output_file"
    
    # Supprimer le fichier temporaire
    rm "$temp_file"
done
