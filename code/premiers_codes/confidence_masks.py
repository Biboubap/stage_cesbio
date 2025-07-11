from evaluation3 import load_precomputed_data
import numpy as np
import matplotlib.pyplot as plt
import joblib
from osgeo import gdal


def get_class_confidence_masks(clf, features, positions, n_samples_x, n_samples_y, class_names, proba_lichen, proba_non_lichen=None, samples_set=None):
    """
    Renvoie un masque binaire pour chaque classe selon la règle :
    - lichen : proba(lichen) >= proba_lichen
    - sphaignes/crevasse : proba(lichen) <= 1-proba_non_lichen et proba(classe) == max(proba autres classes)
    - Si le pixel est noir (r=g=b=0), il n'est classé dans aucune classe (None)
    """
    proba = clf.predict_proba(features)
    class_indices = {name: idx for idx, name in enumerate(clf.classes_)}
    masks = {name: np.zeros((n_samples_x, n_samples_y), dtype=np.uint8) for name in class_names}

    # Pour vérifier la couleur, on a besoin de samples_set
    samples_matrix = samples_set.get_samples_matrix() if samples_set is not None else None

    for idx, (i_x, i_y) in enumerate(positions):
        # Vérifie si le pixel est noir (r=g=b=0)
        if samples_matrix is not None:
            s = samples_matrix[i_y][i_x]
            if s.r_mean == 0 and s.g_mean == 0 and s.b_mean == 0:
                continue  # Ne classe pas ce pixel

        # Vérifie les 8 voisins
            voisin_noir = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = i_x + dx, i_y + dy
                    if 0 <= nx < n_samples_x and 0 <= ny < n_samples_y:
                        voisin = samples_matrix[ny][nx]
                        if voisin.r_mean == 0 and voisin.g_mean == 0 and voisin.b_mean == 0:
                            voisin_noir = True
                            break
                if voisin_noir:
                    break
            if voisin_noir:
                continue  # Ignore ce pixel si un voisin est noir

        p_lichen = proba[idx, class_indices["lichen"]]
        # Lichen : proba >= 0.95
        if p_lichen >= proba_lichen:
            masks["lichen"][i_x, i_y] = 1
        else:
            # Pour sphaignes et crevasse : proba(lichen) <= 0.10 et c'est la classe la plus probable
            for cname in ["sphaignes", "crevasse"]:
                p_class = proba[idx, class_indices[cname]]
                if p_lichen <= 1 - proba_non_lichen and p_class == np.max(proba[idx]):
                    masks[cname][i_x, i_y] = 1
    return masks

def save_masked_rgb(mask, rgb_img, ref_tif_path, out_tif_path, size_patch=32):
    """
    Sauvegarde le produit de l'image RGB par le masque binaire au format tif,
    en utilisant la géoréférence d'un raster de référence.
    Les pixels où mask == 0 sont rendus transparents (alpha=0).
    """
    # Étend le masque à la taille de l'image RGB
    mask_expanded = np.kron(mask, np.ones((size_patch, size_patch), dtype=mask.dtype))
    mask_expanded = mask_expanded[:rgb_img.shape[0], :rgb_img.shape[1]]
    masked = rgb_img * mask_expanded[..., None]

    # Crée le canal alpha : 255 si mask==1, 0 sinon
    alpha = (mask_expanded * 255).astype(np.uint8)

    ds = gdal.Open(ref_tif_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_tif_path, masked.shape[1], masked.shape[0], 4, gdal.GDT_Byte)
    for i in range(3):
        out_ds.GetRasterBand(i+1).WriteArray(masked[..., i])
    out_ds.GetRasterBand(4).WriteArray(alpha)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None

    print(f"Image masquée (avec transparence) sauvegardée dans {out_tif_path}")


def save_mask(mask, ref_tif_path, out_tif_path, size_patch=32):
    """
    Sauvegarde le masque binaire (0/1) au format tif,
    en utilisant la géoréférence d'un raster de référence.
    """
    # Étend le masque à la taille de l'image de référence
    mask_expanded = np.kron(mask, np.ones((size_patch, size_patch), dtype=mask.dtype))
    ds = gdal.Open(ref_tif_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_tif_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    # Coupe si besoin pour correspondre exactement à la taille du raster de référence
    mask_expanded = mask_expanded[:ds.RasterYSize, :ds.RasterXSize]
    out_ds.GetRasterBand(1).WriteArray(mask_expanded)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"Masque binaire sauvegardé dans {out_tif_path}")


# def save_masked_rgb_transparence(mask, rgb_img, ref_tif_path, out_tif_path, size_patch=32):
#     """
#     Sauvegarde le produit de l'image RGB par le masque binaire au format tif,
#     en utilisant la géoréférence d'un raster de référence.
#     Les pixels où mask == 0 sont rendus transparents (alpha=0).
#     """
#     # Étend le masque à la taille de l'image RGB
#     mask_expanded = np.kron(mask, np.ones((size_patch, size_patch), dtype=mask.dtype))
#     mask_expanded = mask_expanded[:rgb_img.shape[0], :rgb_img.shape[1]]
#     masked = rgb_img * mask_expanded[..., None]

#     # Crée le canal alpha : 255 si mask==1, 0 sinon
#     alpha = (mask_expanded * 255).astype(np.uint8)

#     ds = gdal.Open(ref_tif_path)
#     driver = gdal.GetDriverByName('GTiff')
#     out_ds = driver.Create(out_tif_path, masked.shape[1], masked.shape[0], 4, gdal.GDT_Byte)
#     for i in range(3):
#         out_ds.GetRasterBand(i+1).WriteArray(masked[..., i])
#     out_ds.GetRasterBand(4).WriteArray(alpha)
#     out_ds.SetGeoTransform(ds.GetGeoTransform())
#     out_ds.SetProjection(ds.GetProjection())
#     out_ds.FlushCache()
#     out_ds = None

#     print(f"Image masquée (avec transparence) sauvegardée dans {out_tif_path}")

def plot_rgb_and_masks(rgb_img, masks, class_names, xmin, xmax, ymin, ymax, out_png):
    """
    Affiche et sauvegarde un subplot : RGB + 3 masques de confiance pour la fenêtre demandée.
    """
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(rgb_img[xmin:xmax, ymin:ymax])
    axs[0].set_title("Image RGB")
    axs[0].axis("off")
    for i, cname in enumerate(class_names):
        axs[i+1].imshow(masks[cname][xmin:xmax,ymin:ymax], cmap='gray', vmin=0, vmax=1)
        axs[i+1].set_title(f"Masque confiance {cname}")
        axs[i+1].axis("off")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Subplot RGB + masques sauvegardé dans {out_png}")


def main_mask():
   
    x_start = 0
    y_start = 0
    x_end = 31715
    y_end = 17416
    size_patch = 32
    confidence_lichen = 0.7
    confidence_non_lichen = 0.7
    class_names = ["lichen", "sphaignes", "crevasse"]
    samples_set, n_samples_x, n_samples_y, rgb_img, features, positions = load_precomputed_data("data/samples/selection5/precalc")
    # # 1. Créer les samples et calculer les paramètres
    # samples_set, n_samples_x, n_samples_y = create_samples_and_compute(
    #     x_start, y_start, x_end, y_end, size_patch
    # )

    # # 2. Générer l'image RGB à partir des samples
    # rgb_img = create_rgb_image_from_samples(samples_set, n_samples_x, n_samples_y, size_patch)
    # print(f"Image RGB générée de taille : {rgb_img.shape}")
    
    # # 3. Extraire les features
    # features, positions = extract_features(samples_set, n_samples_x, n_samples_y)
    # print(f"Features extraites avec taille : {features.shape}")
    
    # 4. Charger le modèle
    clf = joblib.load("data/samples/selection5/model5.joblib")
    print("Modèle chargé.")
   
     # 5. Obtenir les masques de confiance
    masks = get_class_confidence_masks(clf, 
        features, positions, 
        n_samples_x, n_samples_y, class_names, 
        proba_lichen = confidence_lichen, 
        proba_non_lichen = confidence_non_lichen, 
        samples_set = samples_set)
    print("Masques de confiance obtenus.")

    # 6. Sauvegarder les images RGB masquées pour chaque classe
    for cname in class_names:
        save_masked_rgb(
            masks[cname],
            rgb_img,
            ref_tif_path="data/rgb_reshaped.tif",
            out_tif_path=f"data/samples/selection5/masks70/mask_rgb_{cname}_l{int(confidence_lichen*100)}_nl{int(confidence_non_lichen*100)}.tif",
            size_patch=size_patch
        )

    # 6.5 Sauvegarder les masques au format .tif
    for cname in class_names:
        save_mask(
            masks[cname],
            ref_tif_path="data/rgb_reshaped.tif",
            out_tif_path=f"data/samples/selection5/masks70/mask_01_{cname}_l{int(confidence_lichen*100)}_nl{int(confidence_non_lichen*100)}.tif",
            size_patch=size_patch
        )
        
    # 7. Plot la fenêtre demandée
    # Exemple : fenêtre centrée sur 1000:2000, 1000:2000

    xmin, xmax, ymin, ymax = 0, x_end-x_start, 0, y_end-y_start   
    plot_rgb_and_masks(
        rgb_img, masks, class_names, xmin, xmax, ymin, ymax,
        out_png=f"data/samples/selection5/masks70/masks_l{int(confidence_lichen*100)}_nl{int(confidence_non_lichen*100)}.png"
    )


 
if __name__ == "__main__":
    main_mask()