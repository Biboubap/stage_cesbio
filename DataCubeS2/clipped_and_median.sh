
# Extract contour of non-transparent area in WAP23 mosaic
echo "Extracting contour of non-transparent area from WAP23 mosaic..."
gdal_calc.py --overwrite -A drone_treated/WAP32_full_transparent_mosaic_group1.tif --A_band=4 --outfile=drone_treated/WAP32_tiles/mask_WAP32.tif --calc="A>0" --NoDataValue=0

# Convert the binary mask to a polygon
echo "Converting mask to polygon contour..."
gdal_polygonize.py drone_treated/WAP32_tiles/mask_WAP32.tif -f GeoJSON drone_treated/WAP32_tiles/WAP32_contour.json

# Create directories for clipped bands and indices
mkdir -p DataCubeS2/BandsS22023_WAP32/clipped
mkdir -p DataCubeS2/IndicesS22023_WAP32/clipped

# Clip Sentinel-2 bands using the WAP32 contour
echo "Clipping Sentinel-2 bands with WAP32 contour..."
for f in DataCubeS2/BandsS22023/*.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP32_tiles/WAP32_contour.json -crop_to_cutline "$f" "DataCubeS2/BandsS22023_WAP32/clipped/clipped_${base_filename}"
    echo "Clipped $base_filename"
done

# Clip Sentinel-2 indices using the WAP32 contour
echo "Clipping Sentinel-2 indices with WAP32 contour..."
for f in DataCubeS2/IndicesS22023/*.tif; do
    base_filename=$(basename "$f")
    gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline drone_treated/WAP32_tiles/WAP32_contour.json -crop_to_cutline "$f" "DataCubeS2/IndicesS22023_WAP32/clipped/clipped_${base_filename}"
    echo "Clipped $base_filename"
done

# Create directories for median calculation
mkdir -p DataCubeS2/BandsS22023_WAP32/mediane
mkdir -p DataCubeS2/IndicesS22023_WAP32/mediane

# Compute median for clipped bands
echo "Computing median for clipped Sentinel-2 bands..."
for f in DataCubeS2/BandsS22023_WAP32/clipped/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/BandsS22023_WAP32/mediane/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

# Compute median for clipped indices
echo "Computing median for clipped Sentinel-2 indices..."
for f in DataCubeS2/IndicesS22023_WAP32/clipped/clipped_*.tif; do
    base_filename=$(basename "$f")
    python /home/lcousin/stage_cesbio/DataCubeS2/TwinLakeCubeIndex/compute_median.py "$f" "DataCubeS2/IndicesS22023_WAP32/mediane/mediane_${base_filename}"
    echo "Computed median for $base_filename"
done

echo "All processing completed for WAP32 data clipping and median calculation."