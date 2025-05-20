from osgeo import gdal
import pickle
import numpy as np
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def save_correspondance():
    """fonction beaucoup trop longue, pas exécutée"""
    # Ouvre les deux rasters
    ds_5m = gdal.Open("data/sentinel2/rgb/databand1_reshaped.tif")
    ds_1cm = gdal.Open("data/rgb_reshaped.tif")

    gt_5m = ds_5m.GetGeoTransform()
    gt_1cm = ds_1cm.GetGeoTransform()

    cols_5m = ds_5m.RasterXSize
    rows_5m = ds_5m.RasterYSize
    cols_1cm = ds_1cm.RasterXSize
    rows_1cm = ds_1cm.RasterYSize

    sentinel_to_drone = {}
    drone_to_sentinel = {}

    # Ajout de la barre de progression sur les lignes drone
    for row_d in tqdm(range(rows_1cm), desc="Correspondance drone->sentinel"):
        for col_d in range(cols_1cm):
            # Coordonnées géographiques du coin haut gauche du pixel drone
            x, y = pixel_to_geo(gt_1cm, col_d, row_d)
            # Indices dans le raster 5m
            col_s, row_s = geo_to_pixel(gt_5m, x, y)
            # On vérifie que le pixel sentinel est dans l'image
            if 0 <= col_s < cols_5m and 0 <= row_s < rows_5m:
                # Ajoute la correspondance drone -> sentinel
                drone_to_sentinel[(col_d, row_d)] = (col_s, row_s)
                # Ajoute la correspondance sentinel -> drone (sous forme de liste)
                key = (col_s, row_s)
                if key not in sentinel_to_drone:
                    sentinel_to_drone[key] = []
                sentinel_to_drone[key].append((col_d, row_d))

    # Convertit les listes en numpy arrays pour chaque pixel sentinel
    for key in sentinel_to_drone:
        sentinel_to_drone[key] = np.array(sentinel_to_drone[key], dtype=np.int32)

    print(sentinel_to_drone.get((0, 0)))  # Affiche la correspondance pour le pixel (0, 0)
    print(drone_to_sentinel.get((0, 0)))  # Affiche la correspondance pour le pixel (0, 0)

    # Sauvegarde des dictionnaires
    with open("data/sentinel2/sentinel_to_drone.pkl", "wb") as f:
        pickle.dump(sentinel_to_drone, f)
    with open("data/sentinel2/drone_to_sentinel.pkl", "wb") as f:
        pickle.dump(drone_to_sentinel, f)

    print("Dictionnaires de correspondance sauvegardés.")
    
# # Exemple d'utilisation :
# save_correspondance()


def load_pickle_dict(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d

# # Exemple d'utilisation :
# sentinel_to_drone = load_pickle_dict("data/sentinel2/sentinel_to_drone.pkl")
# drone_to_sentinel = load_pickle_dict("drone_to_sentinel.pkl")

def sentinel_to_drone_bounds(col_s, row_s):
    ds_5m = gdal.Open("data/sentinel2/rgb/databand1_reshaped.tif")
    ds_1cm = gdal.Open("data/rgb_reshaped.tif")
    gt_5m = ds_5m.GetGeoTransform()
    gt_1cm = ds_1cm.GetGeoTransform()

    """
    Pour un pixel Sentinel (col_s, row_s), retourne les bornes xmin, xmax, ymin, ymax
    des pixels drone (1cm) couverts par ce pixel Sentinel (5m).
    """
    # Coordonnées géographiques du coin haut-gauche du pixel Sentinel
    x_min, y_max = pixel_to_geo(gt_5m, col_s, row_s)
    # Coordonnées géographiques du coin bas-droit du pixel Sentinel
    x_max, y_min = pixel_to_geo(gt_5m, col_s + 1, row_s + 1)

    # Indices pixel drone correspondants
    col_min, row_min = geo_to_pixel(gt_1cm, x_min, y_min)  # coin bas-gauche
    col_max, row_max = geo_to_pixel(gt_1cm, x_max, y_max)  # coin haut-droit

    # Correction pour garantir xmin < xmax et ymin < ymax
    xmin = min(col_min, col_max)
    xmax = max(col_min, col_max)
    ymin = min(row_min, row_max)
    ymax = max(row_min, row_max)

    return xmin, xmax, ymin, ymax

# # Exemple d'utilisation :
# if __name__ == "__main__":
#     xmin, xmax, ymin, ymax = sentinel_to_drone_bounds(10, 20)
#     print(xmin, xmax, ymin, ymax)



def compute_lichen_proportion_per_sentinel_pixel(mask_path, sentinel_path):
    # Charge le masque lichen drone (1 = lichen, 0 = autre, 255 = nodata)
    ds_mask = gdal.Open(mask_path)
    mask = ds_mask.GetRasterBand(1).ReadAsArray()
    ds_5m = gdal.Open(sentinel_path)
    cols_5m = ds_5m.RasterXSize
    rows_5m = ds_5m.RasterYSize

    result_dict = {}

    for row_s in range(rows_5m):
        for col_s in range(cols_5m):
            xmin, xmax, ymin, ymax = sentinel_to_drone_bounds(col_s, row_s)
            # Vérifie que la fenêtre est dans les bornes du masque
            if xmin < 0 or ymin < 0 or xmax > mask.shape[1] or ymax > mask.shape[0]:
                result_dict[(col_s, row_s)] = None
                continue
            submask = mask[ymin:ymax, xmin:xmax]
            if np.any(submask == 255) or submask.size == 0:
                result_dict[(col_s, row_s)] = None
            else:
                prop = np.sum(submask == 1) / submask.size
                result_dict[(col_s, row_s)] = prop

    return result_dict

def save_proportion_dict_to_csv(result_dict, out_csv):
    df = pd.DataFrame([
        {"col_s": k[0], "row_s": k[1], "proportion_lichen": v}
        for k, v in result_dict.items()
    ])
    df.to_csv(out_csv, index=False)
    print(f"Résultat sauvegardé dans {out_csv}")

import pandas as pd

def load_proportion_csv(csv_path):
    """
    Charge le fichier CSV des proportions de lichen par pixel sentinel dans un DataFrame pandas.
    """
    df = pd.read_csv(csv_path)
    return df

# Exemple d'utilisation :
# df = load_proportion_csv("data/sentinel2/lichen_proportion_per_sentinel_pixel.csv")
# print(df.head())

# # # Exemple d'utilisation :
# if __name__ == "__main__":
#     result = compute_lichen_proportion_per_sentinel_pixel(
#         mask_path="data/samples/selection4/lichen_mask.tif",
#         sentinel_path="data/sentinel2/rgb/databand1_reshaped.tif"
#     )
#     save_proportion_dict_to_csv(result, "data/sentinel2/lichen_proportion_per_sentinel_pixel.csv")


def load_sentinel_rgb_bands(sentinel_path):
    ds = gdal.Open(sentinel_path)
    r = ds.GetRasterBand(1).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    b = ds.GetRasterBand(3).ReadAsArray()
    return r, g, b

def plot_rgb_vs_lichen_proportion(csv_path, sentinel_path, out_png, distance_bord=0, show_mask=False):
    # Charge les données
    df = load_proportion_csv(csv_path)
    r, g, b = load_sentinel_rgb_bands(sentinel_path)

    # Ne garde que les pixels avec une proportion définie (non None et non NaN)
    df = df[df["proportion_lichen"].notnull()]

    # Filtre les pixels "intérieurs" selon la distance au bord
    if distance_bord > 0:
        mask = mask_interior_pixels(df, distance=distance_bord, show = show_mask)
        df = df[mask]

    # Récupère les valeurs RGB et la proportion pour chaque pixel Sentinel
    reds = []
    greens = []
    blues = []
    means = []
    props = []

    for _, row in df.iterrows():
        col_s = int(row["col_s"])
        row_s = int(row["row_s"])
        prop = row["proportion_lichen"]
        reds.append(r[row_s, col_s])
        greens.append(g[row_s, col_s])
        blues.append(b[row_s, col_s])
        means.append(np.mean([r[row_s, col_s], g[row_s, col_s], b[row_s, col_s]]))
        props.append(prop)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].scatter(props, reds, color='red', alpha=0.5, s=5)
    axs[0].set_title("Rouge Sentinel vs proportion de lichen")
    axs[0].set_xlabel("Proportion de lichen")
    axs[0].set_ylabel("Valeur Rouge")

    axs[1].scatter(props, greens, color='green', alpha=0.5, s=5)
    axs[1].set_title("Vert Sentinel vs proportion de lichen")
    axs[1].set_xlabel("Proportion de lichen")
    axs[1].set_ylabel("Valeur Vert")

    axs[2].scatter(props, blues, color='blue', alpha=0.5, s=5)
    axs[2].set_title("Bleu Sentinel vs proportion de lichen")
    axs[2].set_xlabel("Proportion de lichen")
    axs[2].set_ylabel("Valeur Bleu")

    axs[3].scatter(props, means, color='gray', alpha=0.5, s=5)
    axs[3].set_title("Moyenne RGB Sentinel vs proportion de lichen")
    axs[3].set_xlabel("Proportion de lichen")
    axs[3].set_ylabel("Moyenne RGB")

    plt.suptitle("Valeurs RGB Sentinel2 en fonction de la proportion de lichen (drone) par pixel Sentinel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"Graphe sauvegardé dans {out_png}")



from scipy.ndimage import binary_dilation

def mask_interior_pixels(df, distance=0, show=False):
    """
    Renvoie un masque booléen de la même taille que df, True si le pixel est "intérieur"
    (aucun pixel None à moins de 'distance' en distance de Manhattan).
    """
    # Création de la matrice des proportions (None -> nan)
    max_row = df["row_s"].max() + 1
    max_col = df["col_s"].max() + 1
    mat = np.full((max_row, max_col), np.nan)
    for _, row in df.iterrows():
        mat[int(row["row_s"]), int(row["col_s"])] = row["proportion_lichen"]

    # Pixels None (bord)
    mask_none = np.isnan(mat)
    if distance == 0:
        # Tous les pixels non-None sont valides
        mask_valid = ~mask_none
    else:
        # Dilate les pixels None pour marquer les pixels proches du bord
        struct = np.zeros((2*distance+1, 2*distance+1), dtype=bool)
        for i in range(2*distance+1):
            for j in range(2*distance+1):
                if abs(i-distance) + abs(j-distance) <= distance:
                    struct[i, j] = True
        mask_border = binary_dilation(mask_none, structure=struct)
        mask_valid = (~mask_none) & (~mask_border)
    # Création d'un set des indices valides
    valid_indices = set(zip(*np.where(mask_valid)))
    # Masque pour le DataFrame
    mask_df = df.apply(lambda row: (int(row["row_s"]), int(row["col_s"])) in valid_indices, axis=1)

    if show:
        # Affichage du masque
        plt.imshow(mask_none, cmap='gray', interpolation='nearest')
        plt.title("Masque des pixels None")
        plt.colorbar()
        plt.show()

        plt.imshow(mask_valid, cmap='gray', interpolation='nearest')
        plt.title("Masque des pixels valides")
        plt.colorbar()
        plt.show()

    return mask_df
    
# # Exemple d'utilisation :
# if __name__ == "__main__":
#     distance_bord=1
#     plot_rgb_vs_lichen_proportion(
#         csv_path="data/sentinel2/sentinel_analysis/lichen_proportion_per_sentinel_pixel.csv",
#         sentinel_path="data/sentinel2/rgb/databand1_reshaped.tif",
#         out_png=f"data/sentinel2/sentinel_analysis/rgb_vs_lichen_proportion_distance{distance_bord}.png",
#         distance_bord=distance_bord,
#         show_mask=True
#     )