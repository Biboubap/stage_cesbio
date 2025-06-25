# Extract contour of non-transparent area in WAP23 mosaic
echo "Extracting contour of non-transparent area from WAP32 mosaic..."
gdal_calc.py --overwrite -A drone_treated/WAP32_tiles/WAP32_classif_merged.tif --A_band=1 --outfile=drone_treated/WAP32_tiles/mask_WAP32.tif --calc="A>0" --NoDataValue=0

# Convert the binary mask to a polygon
echo "Converting mask to polygon contour..."
gdal_polygonize.py drone_treated/WAP32_tiles/mask_WAP32.tif -f GeoJSON drone_treated/WAP32_tiles/WAP32_contour.json

# Create directories for clipped bands and indices
mkdir -p DataCubeS2/WAP32/clipped_bands
mkdir -p DataCubeS2/WAP32/clipped_indices

# Clip Sentinel-2 bands using the WAP32 contour
echo "Clipping Sentinel-2 bands with WAP32 contour..."
for f in DataCubeS2/Bands_S2_10m_2023/*WAP32_deflate.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP32_tiles/WAP32_contour.json -crop_to_cutline "$f" "DataCubeS2/WAP32/clipped_bands/clipped_${base_filename}"
    echo "Clipped $base_filename"
done

# Clip Sentinel-2 indices using the WAP32 contour
echo "Clipping Sentinel-2 indices with WAP32 contour..."
for f in DataCubeS2/Indices_S2_10m_2023/*WAP32_deflate.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP32_tiles/WAP32_contour.json -crop_to_cutline "$f" "DataCubeS2/WAP32/clipped_indices/clipped_${base_filename}"
    echo "Clipped $base_filename"
done

# Create directories for median calculation
mkdir -p DataCubeS2/WAP32/mediane_bands
mkdir -p DataCubeS2/WAP32/mediane_indices

# Compute median for clipped bands
echo "Computing median for clipped Sentinel-2 bands..."
for f in DataCubeS2/WAP32/clipped_bands/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/WAP32/mediane_bands/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

# Compute median for clipped indices
echo "Computing median for clipped Sentinel-2 indices..."
for f in DataCubeS2/WAP32/clipped_indices/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/WAP32/mediane_indices/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

echo "All processing completed for WAP32 data clipping and median calculation."



# Extract contour of non-transparent area in WAP23 mosaic
echo "Extracting contour of non-transparent area from WAP23 mosaic..."
gdal_calc.py --overwrite -A drone_treated/WAP23_tiles/WAP23_classif_all_peat.tif --A_band=1 --outfile=drone_treated/WAP23_tiles/mask_WAP23_peat.tif --calc="A>0" --NoDataValue=0

# Convert the binary mask to a polygon
echo "Converting mask to polygon contour..."
gdal_polygonize.py drone_treated/WAP23_tiles/mask_WAP23_peat.tif -f GeoJSON drone_treated/WAP23_tiles/WAP23_contour_peat.json

# Create directories for clipped bands and indices
_
mkdir -p DataCubeS2/IndicesS22023_WAP23_peat/clipped

# Clip Sentinel-2 bands using the WAP23 contour
echo "Clipping Sentinel-2 bands with WAP23 contour..."
for f in DataCubeS2/BandsS22023/*WAP23_deflate.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP23_tiles/WAP23_contour_peat.json -crop_to_cutline "$f" "DataCubeS2/BandsS22023_WAP23_peat/clipped/clipped_${base_filename}"
    echo "Clipped $base_filename"
done

# Clip Sentinel-2 indices using the WAP23 contour
echo "Clipping Sentinel-2 indices with WAP23 contour..."
for f in DataCubeS2/IndicesS22023/*WAP23_deflate.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP23_tiles/WAP23_contour.json -crop_to_cutline "$f" "DataCubeS2/IndicesS22023_WAP23_peat/clipped/clipped_${base_filename}"
    echo "Clipped $base_filename"
done

# Create directories for median calculation
mkdir -p DataCubeS2/BandsS22023_WAP23_peat/mediane
mkdir -p DataCubeS2/IndicesS22023_WAP23_peat/mediane

# Compute median for clipped bands
echo "Computing median for clipped Sentinel-2 bands..."
for f in DataCubeS2/BandsS22023_WAP23_peat/clipped/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/BandsS22023_WAP23_peat/mediane/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

# Compute median for clipped indices
echo "Computing median for clipped Sentinel-2 indices..."
for f in DataCubeS2/IndicesS22023_WAP23_peat/clipped/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/IndicesS22023_WAP23_peat/mediane/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

echo "All processing completed for WAP23 data clipping and median calculation."






# Process all WAP sites in a loop
echo "Starting batch processing for WAP12, WAP23, and WAP32..."
for wap in 12 23 32; do
    # if [ "$wap" == "12" ]; then
    #         mosaic_file="Wap12_Main_transparent_mosaic_group1"
    #     elif [ "$wap" == "23" ]; then
    #         mosaic_file="Wap23_main_transparent_mosaic_group1"
    #     elif [ "$wap" == "32" ]; then
    #         mosaic_file="WAP32_full_transparent_mosaic_group1"
    #     fi
    # # Extract contour of non-transparent area from mosaic using Band 4 (comparing with 255)
        # echo "Extracting contour of non-transparent area from WAP${wap} mosaic (${mosaic_file})..."
        # gdal_calc.py --overwrite -A drone_treated/${mosaic_file}.tif --A_band=4 --outfile=drone_treated/WAP${wap}_tiles/mask_WAP${wap}.tif --calc="A==255" --NoDataValue=0

        # # Convert the binary mask to a polygon
        # echo "Converting mask to polygon contour..."
        # gdal_polygonize.py drone_treated/WAP${wap}_tiles/mask_WAP${wap}.tif -f GeoJSON drone_treated/WAP${wap}_tiles/WAP${wap}_contour.json

    
    # Set the appropriate mosaic file name based on WAP site
   
    
    #  Create directories for clipped bands and indices
    # mkdir -p DataCubeS2/WAP${wap}_5m/clipped_bands
    # mkdir -p DataCubeS2/WAP${wap}_5m/clipped_indices

    # echo "Clipping Sentinel-2 bands with WAP contour..."
    # for f in DataCubeS2/Bands_S2_5m_2023/*WAP${wap}_deflate.tif; do
    #     base_filename=$(basename "$f")
    #     gdalwarp -overwrite -of GTiff -tr 5 -5.0 -tap -cutline drone_treated/WAP${wap}_tiles/WAP${wap}_contour.json -crop_to_cutline "$f" "DataCubeS2/WAP${wap}_5m/clipped_bands/clipped_${base_filename}"
    #     echo "Clipped $base_filename"
    # done

    echo "Clipping Sentinel-2 indices with WAP32 contour..."
    for f in DataCubeS2/Indices_S2_10m_2023/*WAP${wap}_deflate.tif; do
        base_filename=$(basename "$f")
        gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP${wap}_tiles/WAP${wap}_contour.json -crop_to_cutline "$f" "DataCubeS2/WAP${wap}_5m/clipped_indices/clipped_${base_filename}"
        echo "Clipped $base_filename"
    done

        
    # # Create directories for median calculation
    # mkdir -p DataCubeS2/WAP${wap}_5m/mediane_bands
    # mkdir -p DataCubeS2/WAP${wap}_5m/mediane_indices

    # # Compute median for clipped bands
    # echo "Computing median for clipped Sentinel-2 bands..."
    # for f in DataCubeS2/WAP${wap}_5m/clipped_bands/clipped_*.tif; do
    #     base_filename=$(basename "$f")
    #     python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/WAP${wap}_5m/mediane_bands/mediane_${base_filename}"
    #     echo "Computed median for $base_filename"
    # done

    # Compute median for clipped indices
    echo "Computing median for clipped Sentinel-2 indices..."
    for f in DataCubeS2/WAP${wap}_5m/clipped_indices/clipped_*.tif; do
        base_filename=$(basename "$f")
        python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/WAP${wap}_5m/mediane_indices/mediane_${base_filename}"
        echo "Computed median for $base_filename"
    done
    
    # # Compute median for clipped bands
    # echo "Computing median for clipped Sentinel-2 bands..."
    # for f in DataCubeS2/BandsS22023_WAP${wap}_10m/clipped/clipped_*.tif; do
    #     base_filename=$(basename "$f")
    #     python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/BandsS22023_WAP${wap}_10m/mediane/mediane_${base_filename}"
    #     echo "Computed median for $base_filename"
    # done

    echo "Completed processing for WAP${wap}"
done

echo "All WAP sites processing completed."


mkdir -p DataCubeS2/IndicesS22023_WAP23_peat_10m/clipped
echo "Clipping Sentinel-2 bands with WAP23 contour..."
for f in DataCubeS2/BandsS22023_10m/*WAP23_deflate.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 10.0 -10.0 -tap -cutline drone_treated/WAP23_tiles/WAP23_contour_peat.json -crop_to_cutline "$f" "DataCubeS2/BandsS22023_WAP23_peat_10m/clipped/clipped_${base_filename}"
    echo "Clipped $base_filename"
done
mkdir -p DataCubeS2/BandsS22023_WAP23_peat_10m/mediane
# Compute median for clipped bands
echo "Computing median for clipped Sentinel-2 bands..."
for f in DataCubeS2/BandsS22023_WAP23_peat_10m/clipped/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/BandsS22023_WAP23_peat_10m/mediane/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done