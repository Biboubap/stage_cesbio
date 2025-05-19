import numpy as np
from osgeo import gdal
import joblib
from tqdm import tqdm
from samples_set import SamplesSet

def raster_to_samples_json(rgb_path, dsm_path, thermal_path, out_json, size_patch=32):
    ds = gdal.Open(rgb_path)
    dz = gdal.Open(dsm_path)
    dt = gdal.Open(thermal_path)
    # Lecture de la fenêtre voulue
    r = ds.GetRasterBand(1).ReadAsArray()[:15000]
    g = ds.GetRasterBand(2).ReadAsArray()[:15000]
    b = ds.GetRasterBand(3).ReadAsArray()[:15000]
    t = dt.GetRasterBand(1).ReadAsArray()[:15000]
    z = dz.GetRasterBand(1).ReadAsArray()[:15000]

    width = 15000#ds.RasterXSize
    height = 15000#ds.RasterYSize
    n_samples_x = width // size_patch
    n_samples_y = height // size_patch

    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    samples_set.create_samples_grid(x_start=0, y_start=0, size_patch=size_patch)
    samples_set.fill_neighbors_all(depth_neighbors=1)
    samples_set.save_samples_to_json(out_json)
    print(f"JSON des samples sauvegardé dans {out_json}")

def predict_json_to_map(json_path, model_path, apply_filter=True):
    samples_set = SamplesSet.load_samples_from_json(json_path)
    n_samples_x = samples_set.n_samples_x
    n_samples_y = samples_set.n_samples_y
    samples_matrix = samples_set.get_samples_matrix()
    features = []
    positions = []
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            s = samples_matrix[i_y][i_x]
            feat = [
                s.r_mean, s.g_mean, s.b_mean,
                s.r_var, s.g_var, s.b_var,
                s.r_n_mean, s.g_n_mean, s.b_n_mean,
                s.t_mean, s.t_n_mean,
                s.z_var, s.z_moins_z_n
            ]
            features.append(feat)
            positions.append((i_x, i_y))
    features = np.array(features)
    clf = joblib.load(model_path)
    preds = clf.predict(features)
    class_to_val = {
        "lichen": 1,
        "sphegnes": 2,
        "sphaignes": 2,
        "crevasse": 3,
        "foret": 4,
        "flaque": 5,
        "lac": 6
    }
    pred_map = np.zeros((n_samples_y, n_samples_x), dtype=np.uint8)
    for idx, (i_x, i_y) in enumerate(positions):
        pred_map[i_y, i_x] = class_to_val.get(preds[idx], 0)

    if apply_filter:
        from scipy.ndimage import generic_filter
        def filter_func(values):
            center = values[4]
            neighbors = np.delete(values, 4)
            if center == 0:
                return center
            if not np.any(neighbors == center):
                nonzero_neighbors = neighbors[neighbors != 0]
                if len(nonzero_neighbors) == 0:
                    return center
                vals, counts = np.unique(nonzero_neighbors, return_counts=True)
                return vals[np.argmax(counts)]
            else:
                return center
        pred_map = generic_filter(pred_map, filter_func, size=3, mode='constant', cval=0)
    return pred_map

def save_map_to_tif(pred_map, ref_tif_path, out_tif_path, size_patch=32):
    ds = gdal.Open(ref_tif_path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    n_samples_y, n_samples_x = pred_map.shape
    out_img = np.zeros((height, width), dtype=np.uint8)
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            x0 = i_x * size_patch
            y0 = i_y * size_patch
            out_img[y0:y0+size_patch, x0:x0+size_patch] = pred_map[i_y, i_x]
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_tif_path, width, height, 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(out_img)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"Carte de classification sauvegardée dans {out_tif_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    rgb_path = "data/rgb_reshaped.tif"
    dsm_path = "data/twin_lake_dsm.tif"
    thermal_path = "data/twinLake_Thermal_Resampled.tif"
    json_path = "data/samples/selection3/full_samples.json"
    model_path = "data/samples/selection3/random_forest_model3.joblib"
    out_tif_path = "data/samples/selection3/classification_full.tif"
    size_patch = 32

    raster_to_samples_json(rgb_path, dsm_path, thermal_path, json_path, size_patch=size_patch)
    # pred_map = predict_json_to_map(json_path, model_path, apply_filter=True)
    # save_map_to_tif(pred_map, rgb_path, out_tif_path, size_patch=size_patch)