import numpy as np
from osgeo import gdal
from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_rgb_thermal(rgb_path, thermal_path):
    """
    Charge l'image RGB et l'image thermique, et retourne un tableau (H, W, 4) : r, g, b, t.
    Les deux images doivent être alignées et de même taille.
    """
    rgb_ds = gdal.Open(rgb_path)
    rgb = np.stack([rgb_ds.GetRasterBand(i+1).ReadAsArray() for i in range(3)], axis=-1)
    thermal_ds = gdal.Open(thermal_path)
    thermal = thermal_ds.GetRasterBand(1).ReadAsArray()
    # Vérifie la taille
    assert rgb.shape[:2] == thermal.shape, "RGB et thermique doivent avoir la même taille"
    # Fusionne
    rgbt = np.concatenate([rgb, thermal[..., None]], axis=-1)
    return rgbt

def load_and_prepare_label_mask(mask_lichen_path, mask_sphaignes_path, mask_crevasse_path, rgb_path, size_patch=32, distance_borders=2):
    """
    Charge les trois masques, retire les bords (2*size_patch), et retourne un masque global :
    1=lichen, 2=sphaignes, 3=crevasse, 0=non classé, 255=hors champ (pixels RGB transparents).
    """
    
    # Lecture des masques
    ds_lichen = gdal.Open(mask_lichen_path)
    ds_sphaignes = gdal.Open(mask_sphaignes_path)
    ds_crevasse= gdal.Open(mask_crevasse_path)
    mask_lichen = ds_lichen.GetRasterBand(1).ReadAsArray()
    mask_sphaignes = ds_sphaignes.GetRasterBand(1).ReadAsArray()
    mask_crevasse = ds_crevasse.GetRasterBand(1).ReadAsArray()
    # Lecture alpha du RGB pour la zone d'intérêt
    
    rgb_ds = gdal.Open(rgb_path)
    alpha = rgb_ds.GetRasterBand(4).ReadAsArray() if rgb_ds.RasterCount >= 4 else None

    h, w = mask_lichen.shape
    label = np.zeros((h, w), dtype=np.uint8)
    label[mask_lichen == 1] = 1
    label[mask_sphaignes == 1] = 2
    label[mask_crevasse == 1] = 3

    # Hors champ (transparence RGB)
    if alpha is not None:
        label[alpha == 0] = 255
    
    # Retire les bords (2*size_patch pixels)
    border = distance_borders * size_patch
    label[:border, :] = 0
    label[-border:, :] = 0
    label[:, :border] = 0
    label[:, -border:] = 0

    # Remet la zone hors champ à 255 même sur les bords
    if alpha is not None:
        label[alpha == 0] = 255

    return label

def count_categories_in_patches(label, size_patch_cnn=128, size_patch_rf=32):
    """
    Balaye l'image label par une fenêtre de size_patch_cnn x size_patch_cnn,
    en avançant de size_patch_rf à chaque pas.
    Pour chaque patch, ajoute dans un dictionnaire la clé (x, y) (coin haut gauche)
    et la valeur Counter des catégories présentes dans le patch.
    """
    h, w = label.shape
    patch_cnn_dict = {}
    for x in range(0, h - size_patch_cnn + 1, size_patch_rf):
        for y in range(0, w - size_patch_cnn + 1, size_patch_rf):
            patch = label[x:x+size_patch_cnn, y:y+size_patch_cnn]
            # Compte les catégories (hors 0 et 255 si tu veux ignorer non classé et hors champ)
            vals, counts = np.unique(patch, return_counts=True)
            counter = Counter()
            for v, c in zip(vals, counts):
                counter[int(v)] = int(c)
            patch_cnn_dict[(x, y)] = counter
    return patch_cnn_dict

def save_patch_cnn_dict(patch_cnn_dict, json_path):
    """
    Sauvegarde le dictionnaire patch_cnn_dict au format JSON.
    Les clés (x, y) sont converties en chaînes "x_y".
    Les Counter sont convertis en dicts classiques.
    """
    serializable_dict = {f"{x}_{y}": dict(counter) for (x, y), counter in patch_cnn_dict.items()}
    with open(json_path, "w") as f:
        json.dump(serializable_dict, f)
    print(f"Dictionnaire sauvegardé dans {json_path}")

def load_patch_cnn_dict(json_path):
    """
    Charge le dictionnaire patch_cnn_dict depuis un fichier JSON.
    Les clés sont reconverties en tuples (x, y) et les valeurs en Counter.
    """
    from collections import Counter
    with open(json_path, "r") as f:
        data = json.load(f)
    patch_cnn_dict = {}
    for key, val in data.items():
        x, y = map(int, key.split("_"))
        patch_cnn_dict[(x, y)] = Counter({int(k): int(v) for k, v in val.items()})
    print(f"Dictionnaire chargé depuis {json_path}")
    return patch_cnn_dict

def filter_patch_cnn_dict(patch_cnn_dict, patch_size=128, min_labelled_ratio=0.7, min_nonlichen_ratio=0.1, min_lichen_ratio=0.1):
    """
    Ne garde que les patchs avec au moins min_labelled_ratio de pixels labellisés (1,2,3)
    et au moins min_nonlichen_ratio de pixels non-lichen (2 ou 3).
    """
    filtered = {}
    total_pixels = patch_size * patch_size
    for coord, counter in patch_cnn_dict.items():
        n_lichen = counter.get(1, 0)
        n_sphaignes = counter.get(2, 0)
        n_crevasse = counter.get(3, 0)
        n_labelled = n_lichen + n_sphaignes + n_crevasse
        n_nonlichen = n_sphaignes + n_crevasse

        labelled_ratio = n_labelled / total_pixels
        nonlichen_ratio = n_nonlichen / total_pixels
        lichen_ratio = n_lichen / total_pixels

        if labelled_ratio >= min_labelled_ratio and nonlichen_ratio >= min_nonlichen_ratio and lichen_ratio >= min_lichen_ratio:
            filtered[coord] = counter
    return filtered


def count_total_categories(patch_cnn_dict):
    """
    Compte le nombre total de pixels de chaque catégorie (1, 2, 3) dans tous les patchs du dictionnaire.
    """
    total_counter = Counter()
    for counter in patch_cnn_dict.values():
        for cat in [1, 2, 3]:
            total_counter[cat] += counter.get(cat, 0)
    return total_counter


def select_lichen_samples(patch_cnn, patch_cnn_filtered, patch_size=128, min_lichen_ratio=0.9):
    """
    Sélectionne les patchs contenant du lichen avec au moins 90% de pixels labellisés,
    jusqu'à ce que le total de pixels lichen dépasse celui des sphaignes.
    """
    patch_cnn_balanced = patch_cnn_filtered.copy()
    total_counts = count_total_categories(patch_cnn_filtered)
    total_lichen = total_counts[1]
    total_sphaignes = total_counts[2]
    total_crevasse = total_counts[3]

    # Trie les patchs contenant du lichen et >= 90% labellisés
    
    total_pixels = patch_size * patch_size
    for coord, counter in patch_cnn.items():
        n_lichen = counter.get(1, 0)
        labelled_ratio = n_lichen / total_pixels
        if labelled_ratio > min_lichen_ratio :
            patch_cnn_balanced[coord] = counter
            total_lichen += n_lichen
            if total_lichen > total_sphaignes:
                return patch_cnn_balanced
    raise Exception("Pas assez de patchs lichen pour équilibrer")


def plot_rgb_and_mask_samples(rgb, label_masks, coords, patch_size=128, n_blocks=5, samples_per_line=10):
    """
    Affiche une ligne de n patchs RGB et en dessous la ligne des masques correspondants.
    """
    # Définition de la colormap personnalisée
    # 0 et 255 -> noir, 1 -> blanc, 2 -> marron, 3 -> bleu
    colors = [
        (0, 0, 0),      # 0 : noir
        (1, 1, 1),      # 1 : blanc
        (0.5, 0.25, 0), # 2 : marron
        (0, 0.3, 1),    # 3 : bleu
        (0, 0, 0)       # 255 : noir (sera mappé à l'indice 4)
    ]
    # Pour que 255 soit mappé à la dernière couleur, on crée un tableau de labels où 255 -> 4
    def mask_for_display(mask):
        mask_disp = mask.copy()
        mask_disp[mask_disp == 255] = 4
        return mask_disp

    cmap = ListedColormap(colors)

    
    fig, axs = plt.subplots(n_blocks * 2, samples_per_line, figsize=(2*samples_per_line, 2*n_blocks*2))
    coords_list = list(coords)[:n_blocks * 2 * samples_per_line]
    for block in range(n_blocks):
        for i in range(samples_per_line):
            idx = block * 2 * samples_per_line + i
            if idx >= len(coords_list):
                continue
            x, y = coords_list[idx]
            patch_rgb = rgb[x:x+patch_size, y:y+patch_size, :]
            patch_mask = label_masks[x:x+patch_size, y:y+patch_size]
            axs[block*2, i].imshow(patch_rgb)
            axs[block*2, i].axis("off")
            axs[block*2+1, i].imshow(mask_for_display(patch_mask), cmap=cmap, vmin=0, vmax=4)
            axs[block*2+1, i].axis("off")
        axs[block*2, 0].set_ylabel(f"RGB {block+1}", fontsize=14)
        axs[block*2+1, 0].set_ylabel(f"Masque {block+1}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rgbt = load_rgb_thermal("data/rgb_reshaped.tif", "data/thermal_reshaped.tif")
    # print("Image shape (r,g,b,t):", rgbt.shape)
    label_masks = load_and_prepare_label_mask(
        "data/samples/selection5/masks/mask_01_lichen_l95_nl95.tif",
        "data/samples/selection5/masks/mask_01_sphaignes_l95_nl95.tif",
        "data/samples/selection5/masks/mask_01_crevasse_l95_nl95.tif",
        "data/rgb_reshaped.tif",
        size_patch=32,
        distance_borders=5
    )
    print("Label shape:", label_masks.shape)

    size_patch_cnn = 256

    # patch_cnn_dict = count_categories_in_patches(label_masks, size_patch_cnn=size_patch_cnn, size_patch_rf=32)
    # save_patch_cnn_dict(patch_cnn_dict, "data/samples/selection5/masks/patch_cnn_dict.json")
    patch_cnn_dict_loaded = load_patch_cnn_dict("data/samples/selection5/masks/patch_cnn_dict.json")
    

    filtered_dict = filter_patch_cnn_dict(patch_cnn_dict_loaded, 
        patch_size=size_patch_cnn, 
        min_labelled_ratio=0.7, 
        min_nonlichen_ratio=8/64,
        min_lichen_ratio=2/64)
    save_patch_cnn_dict(filtered_dict, "data/samples/selection5/masks/patch_cnn_dict_filtered2.json")
    print(f"{len(filtered_dict)} patchs gardés après filtrage")
    
    filtered_dict_loaded = load_patch_cnn_dict("data/samples/selection5/masks/patch_cnn_dict_filtered2.json")
    total_counts = count_total_categories(filtered_dict_loaded)
    print(f"Total lichen (1)  : {total_counts[1]}")
    print(f"Total sphaignes(2): {total_counts[2]}")
    print(f"Total crevasse(3) : {total_counts[3]}")

        
    # # 1. Sélectionne les patchs lichen jusqu'à équilibre
    # balanced_dict = select_lichen_samples(patch_cnn_dict_loaded, filtered_dict_loaded, patch_size=128, min_lichen_ratio=0.9)

    # # 2. Sauvegarde le dictionnaire
    # save_patch_cnn_dict(balanced_dict, "data/samples/selection5/masks/patch_cnn_balanced.json")

    # # 3. Affiche les stats
    # print(f"{len(balanced_dict)} patchs gardés")
    # total_counts = count_total_categories(balanced_dict)
    # print(f"Total lichen (1)  : {total_counts[1]}")
    # print(f"Total sphaignes(2): {total_counts[2]}")
    # print(f"Total crevasse(3) : {total_counts[3]}")
    balanced_dict = filtered_dict_loaded #load_patch_cnn_dict("data/samples/selection5/masks/patch_cnn_balanced.json")
    # 4. Affiche 100 patchs en RGB
    # Charge l'image RGB+T pour extraire les patchs
    ds = gdal.Open("data/rgb_reshaped.tif")
    rgb = np.stack([ds.GetRasterBand(i+1).ReadAsArray() for i in range(3)], axis=-1)
    plot_rgb_and_mask_samples(rgb, label_masks, balanced_dict.keys(), patch_size=size_patch_cnn, n_blocks=5, samples_per_line=10)