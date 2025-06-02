import numpy as np
import matplotlib.pyplot as plt
import joblib
from samples_set2 import SamplesSet2
from sample2 import Sample2
from rasters_manager_tiles import RastersManager
from osgeo import gdal
import pickle
import os
import math
import rasterio

def create_samples_and_compute(x_start, y_start, x_end, y_end, size_patch, ds_path, dz_path=None, dt_path=None, window_mode=False, distance_large=3):
    n_samples_x = (x_end - x_start) // size_patch
    n_samples_y = (y_end - y_start) // size_patch
    samples_set = SamplesSet2(
        ds_path=ds_path,
        dz_path=dz_path,
        dt_path=dt_path,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y
    )
    
    if window_mode:
        # Configure RastersManager to load only the needed window
        rasters_manager = RastersManager()
        # Définir le chemin des rasters pour assurer qu'ils sont disponibles pour la lecture par fenêtre
        rasters_manager.set_paths(ds_path=ds_path, dz_path=dz_path, dt_path=dt_path)
        # Charger seulement la fenêtre nécessaire, en mode optimisé
        rasters_manager.load_raster_window(x_start, y_start, x_end - x_start, y_end - y_start, 
                                          optimized=True)
    
    print("Création des samples...")
    samples_set.create_samples_grid(x_start=x_start, y_start=y_start, size_patch=size_patch)
    print("Samples créés, calcul des paramètres en cours...")
    samples_set.fill_neighbors_all(distance_large=distance_large)
    return samples_set, n_samples_x, n_samples_y


def extract_features(samples_set, n_samples_x, n_samples_y, include_large=True):
    samples_matrix = samples_set.get_samples_matrix()
    features = []
    positions = []
    
    # Define feature names
    feature_names = [
        "r_mean", "g_mean", "b_mean",
        "r_var", "g_var", "b_var",
        "r_n_mean", "g_n_mean", "b_n_mean",
        "t_mean", "t_n_mean", "t_var",
    ]
    
    # Ajouter les features de voisinage large seulement si elles sont demandées
    if include_large:
        feature_names.extend([
            "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean",
            "z_var", "z_moins_z_n", "z_moins_z_large"
        ])
    else:
        feature_names.extend([
            "z_var", "z_moins_z_n"
        ])
    
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            s = samples_matrix[i_y][i_x]
            if include_large:
                feat = [
                    s.r_mean, s.g_mean, s.b_mean,
                    s.r_var, s.g_var, s.b_var,
                    s.r_n_mean, s.g_n_mean, s.b_n_mean,
                    s.t_mean, s.t_n_mean, s.t_var,
                    # Large neighborhood features (only included if requested)
                    s.r_large_mean, s.g_large_mean, s.b_large_mean, s.t_large_mean,
                    s.z_var, s.z_moins_z_n, s.z_moins_z_large
                ]
            else:
                feat = [
                    s.r_mean, s.g_mean, s.b_mean,
                    s.r_var, s.g_var, s.b_var,
                    s.r_n_mean, s.g_n_mean, s.b_n_mean,
                    s.t_mean, s.t_n_mean, s.t_var,
                    # Pas de large neighborhood features
                    s.z_var, s.z_moins_z_n
                ]
            features.append(feat)
            positions.append((i_x, i_y))
    return np.array(features), positions, feature_names

def load_mask_tiff(mask_path):
    ds = gdal.Open(mask_path)
    mask = ds.GetRasterBand(1).ReadAsArray()
    return mask

def predict_samples_2(clf, features, positions, n_samples_x, n_samples_y, samples_set=None, mask=None, size_patch=32):
    preds = clf.predict(features)
    pred_map = np.zeros((n_samples_y, n_samples_x), dtype=np.uint8)
    class_to_val = {
        "lichen": 1,
        "chicoutai": 2,
        "crevasse": 3,
        "sphaignes": 4,
        "None": 0
    }
    samples_matrix = samples_set.get_samples_matrix() if samples_set is not None else None

    for idx, (i_x, i_y) in enumerate(positions):
        s = samples_matrix[i_y][i_x] if samples_matrix is not None else None
        # Cas "pas de données" via masque
        if s is not None and mask is not None:
            cx = s.x + size_patch // 2
            cy = s.y + size_patch // 2
            if mask[cx, cy] == 0:
                pred_map[i_y, i_x] = 255
                continue
        # Cas "patch noir" (hors masque ou masque absent)
        if s is not None and s.r_mean == 0 and s.g_mean == 0 and s.b_mean == 0:
            pred_map[i_y, i_x] = 0
            continue
        pred_map[i_y, i_x] = class_to_val.get(preds[idx], 0)
    return pred_map

def create_classification_map(pred_map, size_patch):
    n_samples_y, n_samples_x = pred_map.shape
    color_map = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 3), dtype=np.uint8)
    color_dict = {
        1: [200, 200, 200],   # lichen : gris clair
        2: [0, 100, 0],       # chicoutai : vert foncé
        3: [60, 60, 60],      # crevasse : gris foncé
        4: [181, 101, 29],    # sphaignes : brun
        0: [0, 0, 0],       # mask out : noir
        255: [0, 0, 0]      # no data : noir
    }
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            val = pred_map[i_y, i_x]
            color = color_dict.get(val, [0, 0, 0])
            color_map[i_x*size_patch:(i_x+1)*size_patch, i_y*size_patch:(i_y+1)*size_patch] = color
    return color_map

def plot_results(rgb_img, color_map, x_start, y_start, x_end, y_end, save_path=None):
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    print("rgb_img shape:", rgb_img.shape)
    print("color_map shape:", color_map.shape)
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Image RGB\nFenêtre x: {x_start}-{x_end}, y: {y_start}-{y_end}")
    axes[0].axis("off")
    axes[1].imshow(color_map)
    axes[1].set_title("Carte de classification\nFenêtre x: {}-{}, y: {}-{}".format(x_start, x_end, y_start, y_end))
    axes[1].axis("off")
    # Légende des couleurs
    color_labels = [
        ("lichen",      [200, 200, 200]),
        ("chicoutai",   [0, 100, 0]),
        ("sphaignes",    [181, 101, 29]),
        ("crevasse",    [60, 60, 60]),
        
        # ("foret",       [0, 80, 0]),
        #("flaque",      [0, 0, 120]),
        # ("lac",         [64, 224, 208]),
    ]
    patches = [mpatches.Patch(color=np.array(rgb)/255, label=label) for label, rgb in color_labels]
    axes[1].legend(handles=patches, loc='lower right', fontsize=10, title="Écozones")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch):
    samples_matrix = samples_set.get_samples_matrix()
    #rgb_img = np.zeros((n_samples_y*size_patch, n_samples_x*size_patch, 3), dtype=np.uint8)
    rgb_img = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 3), dtype=np.uint8)
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            s = samples_matrix[i_y][i_x]
            r, g, b, _, _ = s.get_RGBZT()
            rgb = np.dstack((r, g, b)).astype(np.uint8)
            rgb_img[i_x*size_patch:(i_x+1)*size_patch, i_y*size_patch:(i_y+1)*size_patch, :] = rgb
    return rgb_img

def filter_isolated_samples(pred_map):
    """Si un sample est seul de sa classe parmi ses 8 voisins, il prend la classe majoritaire de ses voisins."""
    from scipy.ndimage import generic_filter

    def filter_func(values):
        center = values[4]
        neighbors = np.delete(values, 4)
        if center == 0:
            return center
        # Si aucun voisin n'a la même classe que le centre
        if not np.any(neighbors == center):
            # Prend la classe majoritaire des voisins (hors fond/0)
            nonzero_neighbors = neighbors[neighbors != 0]
            if len(nonzero_neighbors) == 0:
                return center
            vals, counts = np.unique(nonzero_neighbors, return_counts=True)
            return vals[np.argmax(counts)]
        else:
            return center

    filtered = generic_filter(pred_map, filter_func, size=3, mode='constant', cval=0)
    return filtered.astype(pred_map.dtype)

from sklearn.tree import plot_tree

# Supposons que clf est ton RandomForestClassifier déjà entraîné
# On affiche le premier arbre de la forêt (index 0)
def plot_tree_model(clf):
    plt.figure(figsize=(20, 10))
    plot_tree(clf.estimators_[0], 
            feature_names=["r_mean", "g_mean", "b_mean", "r_var", "g_var", "b_var", 
                           "r_n_mean", "g_n_mean", "b_n_mean", 
                           "t_mean", "t_n_mean", "t_var", 
                           "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean",
                           "z_var", "z_moins_z_n", "z_moins_z_large"],
            class_names=clf.classes_,
            filled=True, rounded=True, max_depth=3)  # max_depth=3 pour lisibilité
    plt.show()

def plot_feature_importances(clf, save_path, feature_names=None):
    """
    Affiche un tableau et un graphique des importances des features pour un RandomForestClassifier.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if feature_names is None:
        feature_names = ["r_mean", "g_mean", "b_mean", "r_var", "g_var", "b_var", 
                         "r_n_mean", "g_n_mean", "b_n_mean", 
                         "t_mean", "t_n_mean", "t_var", 
                         "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean",
                         "z_var", "z_moins_z_n", "z_moins_z_large"]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature importances :")
    for idx in indices:
        print(f"{feature_names[idx]:15s} : {importances[idx]*100:.2f} %")

    plt.figure(figsize=(10, 5))
    plt.bar([feature_names[i] for i in indices], importances[indices]*100)
    plt.ylabel("Importance (%)")
    plt.title("Importance des features dans le Random Forest")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)

def save_classification_to_tif(pred_map, ref_tif_path, out_tif_path, size_patch=32, x_offset=0, y_offset=0):
    """
    Sauvegarde la carte de classification (pred_map) au format .tif,
    en utilisant la géoréférence et la taille du raster de référence.
    Supporte maintenant les offsets pour le placement correct des tuiles.
    
    Version optimisée utilisant Rasterio.
    """
    n_samples_y, n_samples_x = pred_map.shape
    width_tile = n_samples_x * size_patch
    height_tile = n_samples_y * size_patch
    
    out_img = np.zeros((height_tile, width_tile), dtype=np.uint8)
    
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            x0 = i_x * size_patch
            y0 = i_y * size_patch
            out_img[y0:y0+size_patch, x0:x0+size_patch] = pred_map[i_y, i_x]
    
    try:
        # Utiliser Rasterio pour extraire les métadonnées
        with rasterio.open(ref_tif_path) as src:
            # Ajuster la transformation pour prendre en compte les offsets
            transform = src.transform
            
            # Appliquer les offsets
            new_transform = rasterio.Affine(
                transform.a, transform.b, transform.c + x_offset * transform.a,
                transform.d, transform.e, transform.f + y_offset * transform.e
            )
            
            # Créer un nouveau raster géoréférencé
            with rasterio.open(
                out_tif_path, 'w',
                driver='GTiff',
                height=height_tile,
                width=width_tile,
                count=1,
                dtype=np.uint8,
                crs=src.crs,
                transform=new_transform,
                nodata=0
            ) as dst:
                dst.write(out_img, 1)
                
        print(f"Carte de classification sauvegardée dans {out_tif_path}")
        
    except Exception as e:
        print(f"Erreur avec Rasterio: {e}, utilisation de GDAL comme fallback")
        # Fallback vers GDAL si Rasterio échoue
        ds = gdal.Open(ref_tif_path)
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_tif_path, width_tile, height_tile, 1, gdal.GDT_Byte)
        out_ds.GetRasterBand(1).WriteArray(out_img)
        
        # Adjust geotransform to account for tile offset
        geotransform = list(ds.GetGeoTransform())
        geotransform[0] = geotransform[0] + x_offset * geotransform[1]  # adjust X origin
        geotransform[3] = geotransform[3] + y_offset * geotransform[5]  # adjust Y origin
        
        out_ds.SetGeoTransform(tuple(geotransform))
        out_ds.SetProjection(ds.GetProjection())
        out_ds.GetRasterBand(1).SetNoDataValue(0)
        out_ds.FlushCache()
        out_ds = None
        print(f"Carte de classification sauvegardée dans {out_tif_path} (méthode GDAL)")

def filter_features(features, all_feature_names, needed_feature_names):
    """
    Filtre les features pour ne garder que celles dont on a besoin pour le modèle.
    
    Args:
        features: Tableau numpy des features (n_samples × n_features)
        all_feature_names: Liste des noms de toutes les features disponibles
        needed_feature_names: Liste des noms des features dont on a besoin
    
    Returns:
        Tableau numpy des features filtrées
    """
    # Vérifier si tous les noms requis existent
    for name in needed_feature_names:
        if name not in all_feature_names:
            print(f"Attention: la feature '{name}' n'existe pas dans les données")
    
    # Trouver les indices des features à conserver
    indices_to_keep = [all_feature_names.index(name) for name in needed_feature_names 
                      if name in all_feature_names]
    
    # Filtrer les features
    filtered_features = features[:, indices_to_keep]
    
    print(f"Features filtrées : {features.shape[1]} → {filtered_features.shape[1]}")
    return filtered_features

def save_precomputed_data(samples_set, n_samples_x, n_samples_y, rgb_img, features, positions, ds_path, dz_path, dt_path, path_prefix, feature_names=None):
    """
    Sauvegarde les objets nécessaires pour éviter de tout recalculer, y compris les chemins des rasters et les noms des features.
    """
    with open(f"{path_prefix}_meta.pkl", "wb") as f:
        pickle.dump({
            "samples_set": samples_set,
            "n_samples_x": n_samples_x,
            "n_samples_y": n_samples_y,
            "features": features,
            "positions": positions,
            "feature_names": feature_names,  # Ajout des noms de features
            "ds_path": ds_path,
            "dz_path": dz_path,
            "dt_path": dt_path
        }, f)
    np.save(f"{path_prefix}_rgb.npy", rgb_img)
    print(f"Pré-calculs sauvegardés avec préfixe {path_prefix}")

def load_precomputed_data(path_prefix):
    """
    Charge les objets nécessaires pour éviter de tout recalculer, y compris les chemins des rasters et les noms des features.
    """
    with open(f"{path_prefix}_meta.pkl", "rb") as f:
        data = pickle.load(f)
    rgb_img = np.load(f"{path_prefix}_rgb.npy")
    print(f"Pré-calculs chargés depuis préfixe {path_prefix}")
    return (data["samples_set"], data["n_samples_x"], data["n_samples_y"], rgb_img,
            data["features"], data["positions"], data.get("ds_path"), data.get("dz_path"), data.get("dt_path"),
            data.get("feature_names"))  # Retourne aussi les noms de features

def create_classification_map_transparent(pred_map, size_patch):
    n_samples_y, n_samples_x = pred_map.shape
    color_map = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 4), dtype=np.uint8)  # 4 canaux (RGBA)
    color_dict = {
        1: [200, 200, 200, 255],   # lichen : gris clair
        2: [0, 100, 0, 255],       # chicoutai : vert foncé
        3: [60, 60, 60, 255],      # crevasse : gris foncé
        4: [181, 101, 29, 255],    # sphaignes : brun
        0: [0, 0, 0, 0],         # mask out : transparent
    }
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            val = pred_map[i_y, i_x]
            color = color_dict.get(val, [0, 0, 0, 0])
            color_map[i_x*size_patch:(i_x+1)*size_patch, i_y*size_patch:(i_y+1)*size_patch, :] = color
    return color_map

def merge_classif(classif1_path, classif2_path, out_path, nodata_val=0, both_val=255):
    """
    Fusionne deux rasters de classification :
    - Si un pixel est non nul dans une seule classif, on prend sa valeur.
    - Si un pixel est nul dans les deux, on met 0 (et transparent si RGBA).
    - Si un pixel est non nul dans les deux, on met 255.
    """
    ds1 = gdal.Open(classif1_path)
    ds2 = gdal.Open(classif2_path)
    arr1 = ds1.GetRasterBand(1).ReadAsArray()
    arr2 = ds2.GetRasterBand(1).ReadAsArray()
    assert arr1.shape == arr2.shape, "Les deux rasters doivent avoir la même taille"

    merged = np.zeros_like(arr1, dtype=np.uint8)

    # Cas 1 : non nul dans les deux
    arr1[arr1 == 255] = 0
    arr2[arr2 == 255] = 0
    arr2[arr2 != 0] += 10
    merged = arr1 + arr2 
   
    # Cas 4 : nul dans les deux => déjà à 0 (transparent si RGBA)

    # Sauvegarde
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, arr1.shape[1], arr1.shape[0], 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(merged)
    out_ds.GetRasterBand(1).SetNoDataValue(nodata_val)
    out_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_ds.SetProjection(ds1.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"Carte fusionnée sauvegardée dans {out_path}")

def main_prediction():
    # Paramètres de la fenêtre à tester
    
    x_start = 0
    y_start = 0
    x_end = 12742 
    y_end = 12796
    size_patch = 16

    # Chemins vers les rasters
    # ds_path = "data/rgb_reshaped.tif"
    # dz_path = "data/dsm_reshaped.tif"
    # dt_path = None

    ds_path = "drone_treated/WAP32_partial_rgb2.tif"
    dz_path = "drone_treated/WAP32_partial_dsm2.tif"
    dt_path = None
    
    # 1. Créer les samples et calculer les paramètres
    print("Création des samples et calcul des paramètres...")
    samples_set, n_samples_x, n_samples_y = create_samples_and_compute(
        x_start, y_start, x_end, y_end, size_patch,
        ds_path=ds_path, dz_path=dz_path, dt_path=dt_path
    )
    print("Samples créés et paramètres calculés. Génération de l'image RGB en cours...")

    # 2. Générer l'image RGB à partir des samples
    rgb_img = create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch)
    print(f"Image RGB générée de taille : {rgb_img.shape}")
   
   
    # 3. Extraire les features
    features, positions, feature_names = extract_features(samples_set, n_samples_x, n_samples_y)
    print(f"Features extraites avec taille : {features.shape}")

    # Pour sauvegarder
    save_precomputed_data(samples_set, n_samples_x, n_samples_y, rgb_img, features=features, positions=positions,
                          ds_path=ds_path, dz_path=dz_path, dt_path=dt_path, path_prefix="data/samples/selection10/precalc_Wap32_2",
                          feature_names=feature_names)

    
    # # # 4. Charger le modèle
    model_data = joblib.load("data/samples/selection10/model10.joblib")
    clf = model_data["model"]  # Extraire le modèle du dictionnaire
    feature_names_model = model_data["feature_names"]  # Récupérer aussi les noms de features
    print("Modèle RF chargé.")

    # # #Pour charger 1.2.3
    # samples_set, n_samples_x, n_samples_y, rgb_img, features, positions, ds_path, dz_path, dt_path, feature_names = \
    #     load_precomputed_data("data/samples/selection8/precalc")
    # print(f"Pré-calculs chargés")


    #5 Filtrer les features pour qu'elles correspondent au modèle
    features_filtered = filter_features(features, feature_names, feature_names_model)
    print(f"Features extraites avec taille après filtrage: {features_filtered.shape}")


    # 6 Définir la zone de prédiction
    mask = None
    # mask_path = "data/samples/selection9/non_lichen_mask.tif"
    # mask = load_mask_tiff(mask_path)
    print("Masque chargé." if mask is not None else "Aucun masque utilisé.")
    
    # 7. Prédire
    pred_map = predict_samples_2(
        clf, features_filtered, positions, n_samples_x, n_samples_y, samples_set,
        mask=mask, size_patch=size_patch
    )
    print("Prédictions effectuées.")
    
    # 8. Filtrage des samples isolés
    pred_map_filtered = pred_map#filter_isolated_samples(pred_map)
    #print("Samples isolés filtrés.")

    # 9. Créer la carte de classification
    color_map = create_classification_map(pred_map_filtered, size_patch)
    print("Carte de classification créée.")

    # 10. Afficher et sauvegarder les résultats
    plot_results(rgb_img, color_map, x_start, y_start, x_end, y_end, save_path="data/samples/selection10/classif_WAP32_2.png")
    print("Résultats affichés et sauvegardés.")

    # 11. Sauvegarder la carte de classification au format .tif
    save_classification_to_tif(
        pred_map_filtered,
        ref_tif_path="ds_path",
        out_tif_path="data/samples/selection10/classif_WAP32_2.tif",
        size_patch = size_patch
    )

    # 12. Sauvegarder les importances des features
    plot_feature_importances(clf, save_path="data/samples/selection10/features_WAP32_2.png", 
                         feature_names=feature_names)

 
# if __name__ == "__main__":
#     main_prediction()
#     #merge_classif("data/samples/selection6/classification_result_2.tif", "data/samples/selection9/classification_result_nonlichen.tif", "data/samples/selection9/fusion_classif.tif")


def save_tile_data(tile_id, samples_set, n_samples_x, n_samples_y, rgb_img, features, positions, feature_names, output_dir):
    """
    Sauvegarde les données calculées pour une tuile spécifique.
    
    Args:
        tile_id: Identifiant unique de la tuile (ex: "x1000_y2000")
        samples_set: L'objet SamplesSet2 contenant les échantillons
        n_samples_x, n_samples_y: Dimensions de la grille d'échantillons
        rgb_img: Image RGB générée à partir des échantillons
        features: Features extraites des échantillons
        positions: Positions des échantillons
        feature_names: Noms des features extraites
        output_dir: Répertoire de sortie pour les fichiers de cache
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, f"tile_{tile_id}.pkl")
    
    # Sauvegarde des données dans un seul fichier pickle
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'samples_set': samples_set,
            'n_samples_x': n_samples_x,
            'n_samples_y': n_samples_y,
            'rgb_img': rgb_img,
            'features': features,
            'positions': positions,
            'feature_names': feature_names
        }, f)
    
    print(f"Données de la tuile {tile_id} sauvegardées dans {cache_file}")
    return cache_file

def load_tile_data(tile_id, output_dir):
    """
    Charge les données précalculées pour une tuile spécifique.
    
    Args:
        tile_id: Identifiant unique de la tuile (ex: "x1000_y2000")
        output_dir: Répertoire contenant les fichiers de cache
        
    Returns:
        Un dictionnaire contenant toutes les données de la tuile ou None si le fichier n'existe pas
    """
    cache_file = os.path.join(output_dir, f"tile_{tile_id}.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Données de la tuile {tile_id} chargées depuis {cache_file}")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données de la tuile {tile_id}: {e}")
        return None

def process_tile(x_start, y_start, tile_width, tile_height, size_patch, ds_path, dz_path, dt_path, 
                model_path, output_prefix, mask=None, feature_names_model=None, distance_large=3, 
                include_large=True, use_cache=True, cache_dir=None):
    """
    Traite une tuile spécifique du raster.
    
    Args:
        use_cache: Si True, tente de charger les données précalculées ou les sauvegarde
        cache_dir: Répertoire pour stocker/charger le cache (par défaut: output_prefix + "_cache")
    """
    import gc  # Import garbage collector
    
    tile_id = f"x{x_start}_y{y_start}"
    
    if cache_dir is None:
        cache_dir = output_prefix + "_cache"
    
    # Vérifier si les données précalculées existent
    tile_data = None
    if use_cache:
        tile_data = load_tile_data(tile_id, cache_dir)
    
    # Si les données existent déjà dans le cache, les utiliser
    if tile_data is not None:
        samples_set = tile_data['samples_set']
        n_samples_x = tile_data['n_samples_x']
        n_samples_y = tile_data['n_samples_y']
        rgb_img = tile_data['rgb_img']
        features = tile_data['features']
        positions = tile_data['positions']
        feature_names = tile_data['feature_names']
        print(f"Données précalculées chargées pour la tuile {tile_id}")
    else:
        # Sinon, calculer les données normalement
        print(f"Traitement de la tuile: x={x_start}, y={y_start}, largeur={tile_width}, hauteur={tile_height}")
        
        # 1. Créer les samples et calculer les paramètres
        print("Création des samples et calcul des paramètres...")
        samples_set, n_samples_x, n_samples_y = create_samples_and_compute(
            x_start, y_start, x_start + tile_width, y_start + tile_height, size_patch,
            ds_path=ds_path, dz_path=dz_path, dt_path=dt_path, window_mode=True,
            distance_large=distance_large
        )
        print("Samples créés et paramètres calculés. Génération de l'image RGB en cours...")

        # 2. Générer l'image RGB à partir des samples
        rgb_img = create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch)
        print(f"Image RGB générée de taille : {rgb_img.shape}")
       
        # 3. Extraire les features
        features, positions, feature_names = extract_features(samples_set, n_samples_x, n_samples_y, include_large=include_large)
        print(f"Features extraites avec taille : {features.shape}")
        
        # Sauvegarder les données calculées si demandé
        if use_cache:
            save_tile_data(tile_id, samples_set, n_samples_x, n_samples_y, rgb_img, features, 
                          positions, feature_names, cache_dir)
    
    # 4. Charger le modèle
    model_data = joblib.load(model_path)
    clf = model_data["model"]  # Extraire le modèle du dictionnaire
    if feature_names_model is None:
        feature_names_model = model_data.get("feature_names", feature_names)  # Récupérer aussi les noms de features
    print("Modèle RF chargé.")

    # 5. Filtrer les features pour qu'elles correspondent au modèle
    features_filtered = filter_features(features, feature_names, feature_names_model)
    print(f"Features extraites avec taille après filtrage: {features_filtered.shape}")

    # 6. Prédire
    pred_map = predict_samples_2(
        clf, features_filtered, positions, n_samples_x, n_samples_y, samples_set,
        mask=mask, size_patch=size_patch
    )
    print("Prédictions effectuées.")
    
    # 7. Filtrage des samples isolés (optionnel)
    pred_map_filtered = pred_map  # ou filter_isolated_samples(pred_map) si souhaité

    # 8. Créer la carte de classification
    color_map = create_classification_map(pred_map_filtered, size_patch)
    print("Carte de classification créée.")

    # 9. Noms des fichiers de sortie avec indicateurs de position
    png_out_path = f"{output_prefix}_{tile_id}.png"
    tif_out_path = f"{output_prefix}_{tile_id}.tif"

    # 10. Afficher et sauvegarder les résultats
    plot_results(rgb_img, color_map, x_start, y_start, x_start + tile_width, y_start + tile_height, save_path=png_out_path)
    print(f"Résultats sauvegardés dans {png_out_path}")

    # 11. Sauvegarder la carte de classification au format .tif
    save_classification_to_tif(
        pred_map_filtered,
        ref_tif_path=ds_path,
        out_tif_path=tif_out_path,
        size_patch=size_patch,
        x_offset=x_start,
        y_offset=y_start
    )
    
    # Nettoyer les ressources
    pred_map = pred_map_filtered = None
    samples_set = None
    rgb_img = None
    positions = None
    feature_names = None
    
    # Force garbage collection 
    gc.collect()
    
    return tif_out_path

# Cache pour les informations de raster
_raster_info_cache = {}

def get_raster_info(raster_path):
    """
    Récupère les informations d'un raster sans l'ouvrir complètement
    en utilisant un cache pour éviter les ouvertures répétées.
    
    Args:
        raster_path: Chemin vers le fichier raster
        
    Returns:
        Tuple (width, height, geotransform, projection)
    """
    global _raster_info_cache
    
    # Vérifier si les informations sont déjà en cache
    if raster_path in _raster_info_cache:
        return _raster_info_cache[raster_path]
    
    # Ouvrir le raster avec rasterio uniquement pour les métadonnées
    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        geotransform = src.transform
        projection = src.crs
    
    # Mettre en cache les informations
    _raster_info_cache[raster_path] = (width, height, geotransform, projection)
    
    return width, height, geotransform, projection

def process_raster_by_tiles(ds_path, dz_path, dt_path, model_path, output_prefix, 
                          max_tile_size=9000, size_patch=16, max_tiles=None, 
                          distance_large=3, include_large=True, use_cache=True, cache_dir=None):
    """
    Divise un raster en tuiles et traite chaque tuile individuellement.
    
    Args:
        use_cache: Si True, tente de charger les données précalculées ou les sauvegarde
        cache_dir: Répertoire pour stocker/charger le cache
    """
    # Obtenir les dimensions du raster sans charger les données
    width, height, _, _ = get_raster_info(ds_path)
    print(f"Dimensions totales du raster : {width}x{height}")
    
    # Calculer le nombre de tuiles nécessaires
    n_tiles_x = math.ceil(width / max_tile_size)
    n_tiles_y = math.ceil(height / max_tile_size)
    print(f"Nombre de tuiles : {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y} tuiles")
    
    # Ajuster la taille des tuiles pour une répartition équilibrée
    tile_width = width // n_tiles_x
    tile_height = height // n_tiles_y
    
    # S'assurer que la taille de la tuile est un multiple de size_patch
    tile_width = (tile_width // size_patch) * size_patch
    tile_height = (tile_height // size_patch) * size_patch
    
    print(f"Taille des tuiles ajustée : {tile_width}x{tile_height}")
    
    # Charger le modèle une fois pour récupérer les noms des features
    model_data = joblib.load(model_path)
    feature_names_model = model_data.get("feature_names", None)
    
    # Traiter chaque tuile
    tif_paths = []
    tile_count = 0
    
    for i_y in range(n_tiles_y):
        for i_x in range(n_tiles_x):
            # Si max_tiles est défini et qu'on a déjà traité ce nombre de tuiles, on s'arrête
            if max_tiles is not None and tile_count >= max_tiles:
                break
                
            x_start = i_x * tile_width
            y_start = i_y * tile_height
            
            # Ajuster la taille de la dernière tuile si nécessaire
            curr_tile_width = min(tile_width, width - x_start)
            curr_tile_height = min(tile_height, height - y_start)
            
            # S'assurer que la taille de la tuile est un multiple de size_patch
            curr_tile_width = (curr_tile_width // size_patch) * size_patch
            curr_tile_height = (curr_tile_height // size_patch) * size_patch
            
            # Ne pas traiter les tuiles trop petites
            if curr_tile_width < size_patch or curr_tile_height < size_patch:
                continue
                
            # Traiter la tuile
            tif_path = process_tile(
                x_start, y_start, curr_tile_width, curr_tile_height, size_patch,
                ds_path, dz_path, dt_path, model_path, output_prefix,
                feature_names_model=feature_names_model,
                distance_large=distance_large, include_large=include_large,
                use_cache=use_cache, cache_dir=cache_dir
            )
            tif_paths.append(tif_path)
            tile_count += 1
            print(f"Tuile {tile_count} traitée : {tif_path}")

    return tif_paths

def main_prediction_tiled():
    """
    Version améliorée de main_prediction qui traite le raster par tuiles.
    """
    # Chemins vers les rasters
    ds_path = "drone_treated/WAP32_full_transparent_mosaic_group1.tif"
    dz_path = "drone_treated/WAP32_full_dsm.tif"
    dt_path = None
    
    # Paramètres
    size_patch = 16
    max_tile_size = 10000 # Taille maximale d'une tuile en pixels
    model_path = "data/samples/selection10/model10_no_large.joblib"
    output_prefix = "data/samples/selection10/classif_WAP32_tiled"
    
    # Paramètre pour tester sur un nombre limité de tuiles (None pour toutes les traiter)
    max_tiles = None  # Mettre à None pour traiter toutes les tuiles
    
    # Paramètres voisinage large
    distance_large = 0  # Mettre à 0 pour ne pas calculer ces attributs
    include_large = False  # Mettre à False pour ne pas utiliser ces features même si calculées
    
    # Paramètres du cache
    use_cache = True
    cache_dir = output_prefix + "_cache"
    
    # Traiter le raster par tuiles
    tif_paths = process_raster_by_tiles(
        ds_path, dz_path, dt_path, model_path, output_prefix,
        max_tile_size=max_tile_size, size_patch=size_patch, max_tiles=max_tiles,
        distance_large=distance_large, include_large=include_large,
        use_cache=use_cache, cache_dir=cache_dir
    )
    
    # Fusionner les tuiles en un seul raster
    if len(tif_paths) > 0:
        merge_tif_tiles(
            tif_paths, 
            f"{output_prefix}_merged.tif", 
            ds_path
        )
        
        print("Traitement par tuiles terminé avec succès !")

def merge_tif_tiles(tif_paths, output_path, ref_tif_path):
    """
    Fusionne plusieurs fichiers TIF en un seul raster, en tenant compte de leurs positions.
    
    Args:
        tif_paths: Liste des chemins vers les fichiers TIF à fusionner.
        output_path: Chemin de sortie pour le fichier TIF fusionné.
        ref_tif_path: Chemin vers le raster de référence pour la géoréférence.
    """
    print(f"Fusion de {len(tif_paths)} tuiles TIF...")
    
    # Utiliser GDAL pour créer un VRT (Virtual Dataset)
    vrt_path = output_path.replace('.tif', '.vrt')
    gdal.BuildVRT(vrt_path, tif_paths)
    
    # Convertir le VRT en GeoTIFF
    translate_options = gdal.TranslateOptions(
        format='GTiff', 
        creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES']
    )
    gdal.Translate(output_path, vrt_path, options=translate_options)
    
    # Nettoyer le fichier VRT temporaire
    os.remove(vrt_path)
    
    print(f"Fusion terminée: {output_path}")

if __name__ == "__main__":
    #main_prediction()
    main_prediction_tiled()
    #merge_classif("data/samples/selection6/classification_result_2.tif", "data/samples/selection9/classification_result_nonlichen.tif", "data/samples/selection9/fusion_classif.tif")
    #process_raster_by_tiles("drone_treated/WAP32_full_transparent_mosaic_group1.tif", "drone_treated/WAP32_full_dsm.tif", None, "data/samples/selection10/model10_no_large.joblib", "data/samples/selection10/classif_WAP32_tiled")