# mkdir -p clipped
# for f in *.tif; do
#     gdalwarp -overwrite -of GTiff -tr 5.0 -5.0 -tap -cutline contour_polygon.json -cl out -crop_to_cutline "$f" "clipped/clipped_${f}"
# done

mkdir -p mediane
for f in clipped2/*.tif; do
  python compute_median.py "$f" "mediane2/mediane_$(basename "$f")"
done