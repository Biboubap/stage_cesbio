import numpy as np
import matplotlib.pyplot as plt
import joblib
from samples_set2 import SamplesSet2
from sample2 import Sample2
from rasters_manager import RastersManager
from osgeo import gdal
import pickle

def create_samples_and_compute(x_start, y_start, x_end, y_end, size_patch, ds_path, dz_path=None, dt_path=None):
    n_samples_x = (x_end - x_start) // size_patch
    n_samples_y = (y_end - y_start) // size_patch
    samples_set = SamplesSet2(
        ds_path=ds_path,
        dz_path=dz_path,
        dt_path=dt_path,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y
    )
    samples_set.create_samples_grid(x_start=x_start, y_start=y_start, size_patch=size_patch)
    samples_set.fill_neighbors_all(distance_large=3)  # Use the new parameter name
    return samples_set, n_samples_x, n_samples_y


def extract_features(samples_set, n_samples_x, n_samples_y):
    samples_matrix = samples_set.get_samples_matrix()
    features = []
    positions = []
    
    # Define feature names
    feature_names = [
        "r_mean", "g_mean", "b_mean",
        "r_var", "g_var", "b_var",
        "r_n_mean", "g_n_mean", "b_n_mean",
        # We keep temperature features in the list, but we'll comment them out in the feature extraction
        "t_mean", "t_n_mean", "t_var",
        "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean",
        "z_var", "z_moins_z_n", "z_moins_z_large"
    ]
    
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            s = samples_matrix[i_y][i_x]
            feat = [
                s.r_mean, s.g_mean, s.b_mean,
                s.r_var, s.g_var, s.b_var,
                s.r_n_mean, s.g_n_mean, s.b_n_mean,
                # Temperature features - commented out as they are not used in the model
                # (dt_path is None, so these values would be None anyway)
                0, 0, 0,  # s.t_mean, s.t_n_mean, s.t_var - using 0 as placeholder
                # Other features
                s.r_large_mean, s.g_large_mean, s.b_large_mean, 
                0,  # s.t_large_mean - using 0 as placeholder
                s.z_var, s.z_moins_z_n, s.z_moins_z_large
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
    # Update class_to_val to match the order specified in the classification report
    class_to_val = {
        "chicoutai": 1,
        "dry_depression": 2,
        "green_depression": 3,
        "lichen": 4,
        "sphaignes": 5,
        "watered_depression": 6,
        "black_depression" : 7,
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
    # Update color_dict to match the colors in the QML
    color_dict = {
        1: [0, 100, 0],       # chicoutai: vert foncé 
        2: [153, 136, 0],     # dry_depression: noir-jaune
        3: [50, 205, 50],     # green_depression: vert clair/flashy
        4: [200, 200, 200],   # lichen: gris clair
        5: [139, 69, 19],     # sphaignes: marron/orange foncé
        6: [80, 80, 80],      # watered_depression: gris
        7: [50, 45, 10],      # black_depression: updated color to dark brown
        0: [0, 0, 0],         # mask out: noir
        255: [0, 0, 0]        # no data: noir
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
        ("chicoutai",         [0, 100, 0]),
        ("dry_depression",    [153, 136, 0]),
        ("green_depression",  [50, 205, 50]),
        ("lichen",            [200, 200, 200]),
        ("sphaignes",         [139, 69, 19]),
        ("watered_depression",[80, 80, 80]),
        ("black_depression",  [25, 20, 0]),
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
    """
    Remplace chaque pixel isolé de sa classe par la classe majoritaire parmi ses voisins.
    Un pixel est considéré isolé s'il n'a aucun voisin de la même classe dans son voisinage 3x3.
    """
    from scipy.ndimage import generic_filter

    def filter_func(values):
        center = values[4]  # La valeur centrale
        if center == 0 or center == 255:  # Ne pas modifier les pixels de fond ou sans données
            return center
        
        neighbors = np.delete(values, 4)  # Tous les voisins sauf le centre
        
        # Si aucun voisin n'a la même classe que le centre, le pixel est isolé
        if not np.any(neighbors == center):
            # Trouver la classe majoritaire parmi les voisins non-nuls
            nonzero_neighbors = neighbors[neighbors > 0]
            if len(nonzero_neighbors) == 0:
                return center  # Si tous les voisins sont nuls, garder la valeur originale
            
            # Compter les occurrences de chaque classe
            unique_vals, counts = np.unique(nonzero_neighbors, return_counts=True)
            # Retourne la classe la plus fréquente
            return unique_vals[np.argmax(counts)]
        else:
            # Le pixel n'est pas isolé, garder sa valeur originale
            return center

    # Appliquer le filtre sur chaque pixel avec un noyau 3x3
    filtered_map = generic_filter(pred_map, filter_func, size=3, mode='constant', cval=0)
    
    # Vérifier combien de pixels ont été modifiés
    changed = np.sum(filtered_map != pred_map)
    print(f"Filtrage des pixels isolés: {changed} pixels modifiés ({changed/(pred_map.size)*100:.2f}%)")
    
    return filtered_map.astype(pred_map.dtype)

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
                           "z_var", "z_moins_z_n", "z_moins_z_large"
                           ],
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
                         "z_var", "z_moins_z_n" "z_moins_z_large"]
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

def create_qgis_colormap(output_qml, class_labels=None):
    """
    Creates a QGIS color map file (.qml) for the classified raster.
    
    Args:
        output_qml: Path to save the QML file
        class_labels: Optional dictionary mapping class values to labels
    """
    if class_labels is None:
        class_labels = {
            1: "Chicoutai",
            2: "Dry Depression",
            3: "Green Depression",
            4: "Lichen",
            5: "Sphaignes",
            6: "Watered Depression",
            7: "Black Depression",
            0: "No Data",
            255: "No Data"
        }
    
    # Define colors for each class (matching our visualization)
    color_dict = {
        1: [0, 100, 0],       # chicoutai: vert foncé 
        2: [153, 136, 0],     # dry_depression: noir-jaune
        3: [50, 205, 50],     # green_depression: vert clair/flashy
        4: [200, 200, 200],   # lichen: gris clair
        5: [139, 69, 19],     # sphaignes: marron/orange foncé
        6: [80, 80, 80],      # watered_depression: gris
        7: [50, 45, 10],      # black_depression: dark brown
        0: [0, 0, 0],         # mask out: noir
        255: [0, 0, 0]        # no data: noir
    }

    # QGIS QML template matching the provided example format
    qml_template = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis hasScaleBasedVisibilityFlag="0" maxScale="0" version="3.22.4-Białowieża" minScale="1e+08" styleCategories="AllStyleCategories">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" fetchMode="0" mode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option type="bool" value="false" name="WMSBackgroundLayer"/>
      <Option type="bool" value="false" name="WMSPublishDataSourceUrl"/>
      <Option type="int" value="0" name="embeddedWidgets/count"/>
      <Option type="QString" value="Value" name="identify/format"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" value="" name="name"/>
      <Option name="properties"/>
      <Option type="QString" value="collection" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" enabled="false" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer alphaBand="-1" nodataColor="" type="paletted" opacity="1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
{color_entries}
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0" gamma="1"/>
    <huesaturation invertColors="0" saturation="0" colorizeRed="255" colorizeOn="0" colorizeGreen="128" colorizeBlue="128" grayscaleMode="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    
    # Generate color entries for the XML in the exact format from the example
    color_entries = []
    for class_value, label in sorted(class_labels.items()):
        rgb = color_dict.get(class_value, [0, 0, 0])
        # Special handling for transparency
        alpha = 0 if class_value in [0, 255] else 255
        
        # Format: <paletteEntry color="#RRGGBB" alpha="255" value="1" label="1"/>
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        entry = f'        <paletteEntry color="{hex_color}" alpha="{alpha}" value="{class_value}" label="{class_value}"/>'
        color_entries.append(entry)
    
    # Insert color entries into template
    qml_content = qml_template.format(color_entries="\n".join(color_entries))
    
    # Write the QML file
    with open(output_qml, 'w') as f:
        f.write(qml_content)
    
    print(f"QGIS color map saved to {output_qml}")

def save_classification_to_tif(pred_map, ref_tif_path, out_tif_path, size_patch=32, create_qml=True):
    """
    Sauvegarde la carte de classification (pred_map) au format .tif,
    en utilisant la géoréférence et la taille du raster de référence.
    
    Args:
        pred_map: Carte de classification sous forme d'un tableau numpy
        ref_tif_path: Chemin vers le raster de référence pour la géoréférence
        out_tif_path: Chemin de sortie pour sauvegarder le raster de classification
        size_patch: Taille du patch (par défaut: 32)
        create_qml: Si True, crée un fichier QML pour QGIS avec la même palette de couleurs
    """
    ds = gdal.Open(ref_tif_path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    n_samples_y, n_samples_x = pred_map.shape
    out_img = np.zeros((height, width), dtype=np.uint8)
    
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            x0 = i_x * size_patch
            y0 = i_y * size_patch
            out_img[x0:x0+size_patch, y0:y0+size_patch] = pred_map[i_y, i_x]
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_tif_path, width, height, 1, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(out_img)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"Carte de classification sauvegardée dans {out_tif_path}")
    
    # Create QML file if requested
    if create_qml:
        qml_path = out_tif_path.replace('.tif', '.qml')
        create_qgis_colormap(qml_path)

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
        1: [0, 100, 0, 255],       # chicoutai: vert foncé
        2: [153, 136, 0, 255],     # dry_depression: noir-jaune
        3: [50, 205, 50, 255],     # green_depression: vert clair/flashy
        4: [200, 200, 200, 255],   # lichen: gris clair
        5: [139, 69, 19, 255],     # sphaignes: marron/orange foncé
        6: [80, 80, 80, 255],      # watered_depression: gris
        7: [25, 20, 0, 255],       # black_depression: noir
        0: [0, 0, 0, 0],           # mask out: transparent
        255: [0, 0, 0, 0]          # no data: transparent
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
    x_end = 4992 
    y_end = 4992
    size_patch = 16

    # Chemins vers les rasters
    # ds_path = "data/rgb_reshaped.tif"
    # dz_path = "data/dsm_reshaped.tif"
    # dt_path = None

    ds_path = "drone_treated/WAP32_partial_rgb2.tif"
    dz_path = "drone_treated/WAP32_partial_dsm2.tif"
    dt_path = None
    
    # # 1. Créer les samples et calculer les paramètres
    # print("Création des samples et calcul des paramètres...")
    # samples_set, n_samples_x, n_samples_y = create_samples_and_compute(
    #     x_start, y_start, x_end, y_end, size_patch,
    #     ds_path=ds_path, dz_path=dz_path, dt_path=dt_path
    # )
    # print("Samples créés et paramètres calculés. Génération de l'image RGB en cours...")

    # # 2. Générer l'image RGB à partir des samples
    # rgb_img = create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch)
    # print(f"Image RGB générée de taille : {rgb_img.shape}")
   
   
    # # 3. Extraire les features
    # features, positions, feature_names = extract_features(samples_set, n_samples_x, n_samples_y)
    # print(f"Features extraites avec taille : {features.shape}")

    # # Pour sauvegarder
    # save_precomputed_data(samples_set, n_samples_x, n_samples_y, rgb_img, features=features, positions=positions,
    #                       ds_path=ds_path, dz_path=dz_path, dt_path=dt_path, path_prefix="data/samples/selection12/pop_merged/precalc",
    #                       feature_names=feature_names)

    # # # 4. Charger le modèle
    model_data = joblib.load("data/samples/selection13/merged/model_wap_32_5.joblib")
    clf = model_data["model"]  # Extraire le modèle du dictionnaire
    feature_names_model = model_data["feature_names"]  # Récupérer aussi les noms de features
    print("Modèle RF chargé.")

    # # #Pour charger 1.2.3
    samples_set, n_samples_x, n_samples_y, rgb_img, features, positions, ds_path, dz_path, dt_path, feature_names = \
        load_precomputed_data("data/samples/selection12/pop_merged/precalc")
    print(f"Pré-calculs chargés")


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
    
    # 8. Filtrage des samples isolés - activer le filtre
    pred_map_filtered = filter_isolated_samples(pred_map)
    print("Samples isolés filtrés.")

    # 9. Créer la carte de classification
    color_map = create_classification_map(pred_map_filtered, size_patch)
    print("Carte de classification créée.")

    # 10. Afficher et sauvegarder les résultats
    plot_results(rgb_img, color_map, x_start, y_start, x_end, y_end, save_path="data/samples/selection13/merged/classif_WAP32_5_filtered.png")
    print("Résultats affichés et sauvegardés.")

    # 11. Sauvegarder la carte de classification au format .tif with QML color map
    save_classification_to_tif(
        pred_map_filtered,
        ref_tif_path=ds_path,
        out_tif_path="data/samples/selection13/merged/classif_WAP32_5_filtered.tif",
        size_patch = size_patch,
        create_qml=True
    )

    # 12. Sauvegarder les importances des features
    # When plotting feature importances, always use the feature_names from your model
    plot_feature_importances(clf, 
                         save_path="data/samples/selection13/merged/features_WAP32_5.png", 
                         feature_names=feature_names_model)

 
if __name__ == "__main__":
    main_prediction()
    #merge_classif("data/samples/selection6/classification_result_2.tif", "data/samples/selection9/classification_result_nonlichen.tif", "data/samples/selection9/fusion_classif.tif")