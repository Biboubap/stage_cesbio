import numpy as np
import matplotlib.pyplot as plt
import joblib
from samples_set import SamplesSet
from sample import Sample
from PIL import Image
from scipy.ndimage import generic_filter

def load_real_image(img_path, target_shape):
    img = Image.open(img_path)
    img_resized = img.resize(target_shape)
    return img_resized

def create_samples_and_compute(x_start, y_start, x_end, y_end, size_patch):
    n_samples_x = (x_end - x_start) // size_patch
    n_samples_y = (y_end - y_start) // size_patch
    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    samples_set.create_samples_grid(x_start=x_start, y_start=y_start, size_patch=size_patch)
    samples_set.fill_neighbors_colors(depth_neighbors=1)
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
                s.r_n_mean, s.g_n_mean, s.b_n_mean,
                s.r_var, s.g_var, s.b_var,
                s.z_var,
                s.delta_z_x, s.delta_z_y
            ]
            features.append(feat)
            positions.append((i_x, i_y))
    return np.array(features), positions

def predict_samples(clf, features, positions, n_samples_x, n_samples_y):
    preds = clf.predict(features)
    pred_map = np.zeros((n_samples_y, n_samples_x), dtype=np.uint8)
    for idx, (i_x, i_y) in enumerate(positions):
        if preds[idx] == "lichen":
            pred_map[i_y, i_x] = 1
        elif preds[idx] == "sphegnes":
            pred_map[i_y, i_x] = 2
    return pred_map

def create_classification_map(pred_map, size_patch):
    n_samples_y, n_samples_x = pred_map.shape
    color_map = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 3), dtype=np.uint8)
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            if pred_map[i_y, i_x] == 1:
                color_map[i_x*size_patch:(i_x+1)*size_patch, i_y*size_patch:(i_y+1)*size_patch] = [160, 160, 160]  # gris
            elif pred_map[i_y, i_x] == 2:
                color_map[i_x*size_patch:(i_x+1)*size_patch, i_y*size_patch:(i_y+1)*size_patch] = [120, 70, 30]    # marron
    return color_map

def plot_results(img_real, color_map, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img_real)
    axes[0].set_title("Image réelle")
    axes[0].axis("off")
    axes[1].imshow(color_map)
    axes[1].set_title("Carte de classification\nGris: lichen, Marron: sphegnes")
    axes[1].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch):
    samples_matrix = samples_set.get_samples_matrix()
    rgb_img = np.zeros((n_samples_x*size_patch, n_samples_y*size_patch, 3), dtype=np.uint8)
    for i_x in range(n_samples_x):
        for i_y in range(n_samples_y):
            s = samples_matrix[i_y][i_x]
            r, g, b, _ = s.get_RGBZ()
            rgb = np.dstack((r, g, b)).astype(np.uint8)
            rgb_img[i_x*size_patch:(i_x+1)*size_patch, i_y*size_patch:(i_y+1)*size_patch, :] = rgb
    return rgb_img


def eliminate_residues(pred_map):
    def filter_func(values):
        center = values[4]
        neighbors = np.delete(values, 4)
        # Si aucun voisin n'a la même classe que le centre ET le centre n'est pas fond (0)
        if center != 0 and not np.any(neighbors == center):
            # Prend la classe majoritaire des voisins (hors fond/0)
            nonzero_neighbors = neighbors[neighbors != 0]
            if len(nonzero_neighbors) == 0:
                return center
            vals, counts = np.unique(nonzero_neighbors, return_counts=True)
            return vals[np.argmax(counts)]
        else:
            return center

    # Appliquer le filtre sur la carte
    filtered = generic_filter(pred_map, filter_func, size=3, mode='constant', cval=0)
    return filtered.astype(pred_map.dtype)

def main():
    # Paramètres de la zone
    x_start, y_start = 12832, 11744
    x_end, y_end = 13792, 12704
    size_patch = 32
    apply_residue_filter = True

    # 1. Créer les samples et calculer les paramètres
    samples_set, n_samples_x, n_samples_y = create_samples_and_compute(
        x_start, y_start, x_end, y_end, size_patch
    )

    # 2. Générer l'image RGB à partir des samples
    img_real = create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch)

    # 3. Extraire les features
    features, positions = extract_features(samples_set, n_samples_x, n_samples_y)

    # 4. Charger le modèle
    clf = joblib.load("data/samples/lichen_sphegnes_balanced/random_forest_model.joblib")

    # 5. Prédire
    pred_map = predict_samples(clf, features, positions, n_samples_x, n_samples_y)

    # 6. Créer la carte de classification
    if apply_residue_filter:  # booléen à définir selon ton besoin
        pred_map = eliminate_residues(pred_map)
    color_map = create_classification_map(pred_map, size_patch)
    # 7. Afficher et sauvegarder les résultats
    plot_results(img_real, color_map, save_path="data/samples/lichen_sphegnes_selection/classification_result_filtered.png")

if __name__ == "__main__":
    main()