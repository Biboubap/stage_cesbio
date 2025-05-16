import numpy as np
import matplotlib.pyplot as plt
import joblib
from samples_set import SamplesSet
from sample import Sample

def create_samples_and_compute(x_start, y_start, x_end, y_end, size_patch):
    n_samples_x = (x_end - x_start) // size_patch
    n_samples_y = (y_end - y_start) // size_patch
    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    samples_set.create_samples_grid(x_start=x_start, y_start=y_start, size_patch=size_patch)
    samples_set.fill_neighbors_all(depth_neighbors=1)
    samples_set.fill_slope(depth_neighbors=1)
    return samples_set, n_samples_x, n_samples_y

def extract_features(samples_set, n_samples_x, n_samples_y):
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
    return np.array(features), positions

def predict_samples(clf, features, positions, n_samples_x, n_samples_y):
    preds = clf.predict(features)
    pred_map = np.zeros((n_samples_y, n_samples_x), dtype=np.uint8)
    class_to_val = {
        "lichen": 1,
        "sphegnes": 2,
        "crevasse": 3,
        "foret": 4,
        "flaque": 5,
        "lac": 6
    }
    for idx, (i_x, i_y) in enumerate(positions):
        pred_map[i_y, i_x] = class_to_val.get(preds[idx], 0)
    return pred_map

def create_classification_map(pred_map, size_patch):
    n_samples_y, n_samples_x = pred_map.shape
    #color_map = np.zeros((n_samples_y*size_patch, n_samples_x*size_patch, 3), dtype=np.uint8)
    color_map = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 3), dtype=np.uint8)
    # Couleurs : lichen (gris clair), sphegnes (marron clair), crevasse (gris foncé), foret (vert foncé), flaque (bleu foncé), lac (turquoise)
    color_dict = {
        1: [200, 200, 200],   # lichen : gris clair
        2: [181, 101, 29],    # sphegnes : marron clair
        3: [60, 60, 60],      # crevasse : gris foncé
        4: [0, 80, 0],        # foret : vert foncé
        5: [0, 0, 120],       # flaque : bleu foncé
        6: [64, 224, 208],    # lac : turquoise
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
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Image RGB\nFenêtre x: {x_start}-{x_end}, y: {y_start}-{y_end}")
    axes[0].axis("off")
    axes[1].imshow(color_map)
    axes[1].set_title("Carte de classification\nFenêtre x: {}-{}, y: {}-{}".format(x_start, x_end, y_start, y_end))
    axes[1].axis("off")

    # Légende des couleurs
    color_labels = [
        ("lichen",      [200, 200, 200]),
        ("sphegnes",    [181, 101, 29]),
        ("crevasse",    [60, 60, 60]),
        ("foret",       [0, 80, 0]),
        ("flaque",      [0, 0, 120]),
        ("lac",         [64, 224, 208]),
    ]
    patches = [mpatches.Patch(color=np.array(rgb)/255, label=label) for label, rgb in color_labels]
    axes[1].legend(handles=patches, loc='lower right', fontsize=10, title="Écozones")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch):
    samples_matrix = samples_set.get_samples_matrix()
    #rgb_img = np.zeros((n_samples_y*size_patch, n_samples_x*size_patch, 3), dtype=np.uint8)
    rgb_img = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 3), dtype=np.uint8)
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            s = samples_matrix[i_y][i_x]
            r, g, b, _ = s.get_RGBZ()
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

def main():
    # Paramètres de la fenêtre à tester
    x_start = 5000
    y_start = 7000
    x_end = 12000
    y_end = 14000
    size_patch = 32

    # 1. Créer les samples et calculer les paramètres
    samples_set, n_samples_x, n_samples_y = create_samples_and_compute(
        x_start, y_start, x_end, y_end, size_patch
    )

    # 2. Générer l'image RGB à partir des samples
    rgb_img = create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch)
    print(f"Image RGB générée de taille : {rgb_img.shape}")
    # 3. Extraire les features
    features, positions = extract_features(samples_set, n_samples_x, n_samples_y)

    print(f"Features extraites avec taille : {features.shape}")
    # 4. Charger le modèle
    clf = joblib.load("data/samples/selection3/random_forest_model3.joblib")
    print("Modèle chargé.")
    # 5. Prédire
    pred_map = predict_samples(clf, features, positions, n_samples_x, n_samples_y)
    print("Prédictions effectuées.")
    # 6. Filtrage des samples isolés
    pred_map_filtered = filter_isolated_samples(pred_map)

    # 7. Créer la carte de classification
    color_map = create_classification_map(pred_map_filtered, size_patch)
    print("Carte de classification créée.")
    # 8. Afficher et sauvegarder les résultats
    plot_results(rgb_img, color_map, x_start, y_start, x_end, y_end, save_path="data/samples/selection3/classification_result4.png")
    print("Résultats affichés et sauvegardés.")
if __name__ == "__main__":
    main()