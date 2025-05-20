from osgeo import gdal
import pickle

def pixel_to_geo(transform, px, py):
    """Convertit des indices pixel (col, row) en coordonnées géo (x, y) du coin haut gauche"""
    x = transform[0] + px * transform[1] + py * transform[2]
    y = transform[3] + px * transform[4] + py * transform[5]
    return x, y

def geo_to_pixel(transform, x, y):
    """Convertit des coordonnées géo (x, y) en indices pixel (col, row)"""
    inv_det = 1 / (transform[1] * transform[5] - transform[2] * transform[4])
    px = inv_det * (transform[5] * (x - transform[0]) - transform[2] * (y - transform[3]))
    py = inv_det * (-transform[4] * (x - transform[0]) + transform[1] * (y - transform[3]))
    return int(round(px)), int(round(py))

# Ouvre les deux rasters
ds_5m = gdal.Open("data/sentinel2/rgb/databand1_reshaped.tif")
ds_1cm = gdal.Open("data/rgb_reshaped.tif")

gt_5m = ds_5m.GetGeoTransform()
gt_1cm = ds_1cm.GetGeoTransform()

cols_5m = ds_5m.RasterXSize
rows_5m = ds_5m.RasterYSize

sentinel_to_drone = {}
drone_to_sentinel = {}

for row in range(rows_5m):
    for col in range(cols_5m):
        # Coordonnées géographiques du coin haut gauche du pixel 5m
        x, y = pixel_to_geo(gt_5m, col, row)
        # Indices dans le raster 1cm
        col_1cm, row_1cm = geo_to_pixel(gt_1cm, x, y)
        sentinel_to_drone[(col, row)] = (col_1cm, row_1cm)
        drone_to_sentinel[(col_1cm, row_1cm)] = (col, row)
        #print(f"Pixel 5m ({col},{row}) -> Coord ({x:.2f},{y:.2f}) -> Pixel 1cm ({col_1cm},{row_1cm})")

## Pixel 5m (9,9) -> Coord (452525.00,6499530.00) -> Pixel 1cm (4001,3934)


# Sauvegarde des dictionnaires
with open("data/sentinel2/sentinel_to_drone.pkl", "wb") as f:
    pickle.dump(sentinel_to_drone, f)
with open("data/sentinel2/drone_to_sentinel.pkl", "wb") as f:
    pickle.dump(drone_to_sentinel, f)

print("Dictionnaires de correspondance sauvegardés.")


# def load_pickle_dict(path):
#     with open(path, "rb") as f:
#         d = pickle.load(f)
#     return d

# # Exemple d'utilisation :
# sentinel_to_drone = load_pickle_dict("data/sentinel2/sentinel_to_drone.pkl")
# drone_to_sentinel = load_pickle_dict("drone_to_sentinel.pkl")