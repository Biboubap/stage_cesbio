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

def sentinel_to_drone_bounds(col_s, row_s, ds_5m=None, ds_1cm=None):
    if ds_5m is None :
        ds_5m = gdal.Open("data/sentinel2/rgb/databand1_reshaped.tif")
    if ds_1cm is None:
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
            xmin, xmax, ymin, ymax = sentinel_to_drone_bounds(col_s, row_s, ds_5m = ds_5m)
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


def load_sentinel_rgb_bands(sentinel_path):
    ds = gdal.Open(sentinel_path)
    r = ds.GetRasterBand(1).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    b = ds.GetRasterBand(3).ReadAsArray()
    return r, g, b

def plot_rgb_vs_lichen_proportion(csv_path, sentinel_path, out_png):
    # Charge les données
    df = load_proportion_csv(csv_path)
    r, g, b = load_sentinel_rgb_bands(sentinel_path)

    # Ne garde que les pixels avec une proportion définie (non None et non NaN)
    df = df[df["proportion_lichen"].notnull()]

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
from osgeo import gdal

def mask_interior_pixel(sentinel_tif, csv_in=None, csv_out=None, show=False, out_mask_tif=None):
    """
    Ne garde que les pixels Sentinel "intérieurs" (aucun voisin 4-connecté à 0).
    Filtre le CSV pour ne garder que ces pixels.
    Affiche le masque si show=True.
    Sauvegarde le masque intérieur au format tif si out_mask_tif est fourni.
    """
    # Charge le raster Sentinel (première bande)
    ds = gdal.Open(sentinel_tif)
    arr = ds.GetRasterBand(1).ReadAsArray()
    rows, cols = arr.shape

    # Crée un masque des pixels intérieurs
    mask = np.zeros(arr.shape)
    mask[arr != 0] = 1
    mask2 = mask.copy()
    mask2[0, :] = 0
    mask2[-1, :] = 0
    mask2[:, 0] = 0
    mask2[:, -1] = 0

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            for r in range (i-1, i+2):
                for c in range (j-1, j+2):
                    if mask[r,c] == 0:
                        mask2[i, j] = 0

    # Sauvegarde du masque intérieur au format tif si demandé
    if out_mask_tif is not None:
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_mask_tif, arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte)
        out_ds.GetRasterBand(1).WriteArray(mask2.astype(np.uint8))
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_ds.FlushCache()
        out_ds = None
        print(f"Masque intérieur sauvegardé dans {out_mask_tif}")

    if show:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Masque Sentinel (pixels=0 en noir)")
        plt.imshow(mask, cmap='gray')
        plt.subplot(1,2,2)
        plt.title("Pixels intérieurs retenus (blanc)")
        plt.imshow(mask2, cmap='gray')
        plt.tight_layout()
        plt.show()

    # Charge le CSV et filtre selon le masque
    
    df = pd.read_csv(csv_in)
    print("mask2 shape:", mask2.shape)
    print("row_s min/max:", df["row_s"].min(), df["row_s"].max())
    print("col_s min/max:", df["col_s"].min(), df["col_s"].max())
    mask_df = df.apply(lambda row: bool(mask2[int(row["row_s"]), int(row["col_s"])]), axis=1)
    df_filtered = df[mask_df.values]
    df_filtered.to_csv(csv_out, index=False)
    print(f"{len(df_filtered)} pixels intérieurs sauvegardés dans {csv_out}")


def plot_lichen_proportion_histogram(csv_path, out_png=None, sqrt=False, log=False):
    """
    Affiche et sauvegarde l'histogramme du nombre d'échantillons par tranche de 10% de proportion de lichen.
    - sqrt=True : utilise la colonne 'sqrt_proportion_lichen'
    - log=True : utilise la colonne 'log_proportion_lichen'
    - sinon : utilise 'proportion_lichen'
    """
    df = pd.read_csv(csv_path)
    if log:
        values = df["log_proportion_lichen"].dropna()
        xlabel = "log(% lichen)"
        title = "Histogramme du log(% lichen) par pixel Sentinel"
        bins = np.arange(values.min(), values.max() + 0.5, 0.5)
    elif sqrt:
        values = df["sqrt_proportion_lichen"].dropna()
        xlabel = "sqrt(% lichen)"
        title = "Histogramme de la racine carrée du % lichen par pixel Sentinel"
        bins = np.arange(values.min(), values.max() + 0.5, 0.5)
    else:
        values = df["proportion_lichen"].dropna() * 100
        xlabel = "Proportion de lichen (%)"
        title = "Histogramme des proportions de lichen par pixel Sentinel\n(pas de 10%)"
        bins = np.arange(0, 110, 10)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Nombre d'échantillons")
    plt.title(title)
    if not log and not sqrt:
        plt.xticks(bins)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        print(f"Histogramme sauvegardé dans {out_png}")
    plt.show()

def transform_proportion_to_sqrt(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    # On suppose que proportion_lichen est entre 0 et 1, on convertit en % puis sqrt
    df["sqrt_proportion_lichen"] = np.sqrt(df["proportion_lichen"] * 100)
    # Optionnel : on peut supprimer l'ancienne colonne si tu veux
    # df = df.drop(columns=["proportion_lichen"])
    df.to_csv(csv_out, index=False)
    print(f"CSV transformé et sauvegardé dans {csv_out}")

def transform_proportion_to_log(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    # On suppose que proportion_lichen est entre 0 et 1, on convertit en % puis log
    # On ajoute un petit epsilon pour éviter log(0)
    epsilon = 1e-6
    df["log_proportion_lichen"] = np.log(df["proportion_lichen"] * 100 + epsilon)
    # Optionnel : on peut supprimer l'ancienne colonne si tu veux
    # df = df.drop(columns=["proportion_lichen"])
    df.to_csv(csv_out, index=False)
    print(f"CSV transformé et sauvegardé dans {csv_out}")


def filter_and_balance_sqrt_lichen(csv_in, csv_out, max_high=250, sqrt_col="sqrt_proportion_lichen"):
    df = pd.read_csv(csv_in)
    # On ne garde que les lignes où sqrt_proportion_lichen est défini
    df = df[df[sqrt_col].notnull()]

    lichen = []
    for lichen_proportion in np.arange(4, 10, 0.1):
        new_lichen = df[(df[sqrt_col] > lichen_proportion) & (df[sqrt_col] <= lichen_proportion+0.1)]
        if len(new_lichen) > max_high:  
            new_lichen = new_lichen.sample(n=max_high, random_state=42)
        lichen.append(new_lichen)
        
    balanced = pd.concat([l for l in lichen], ignore_index=True)
    
    # # Prend au maximum max_high tuiles à très forte proportion de lichen
    # if len(high_lichen) > max_high:
    #     high_lichen = high_lichen.sample(n=max_high, random_state=42)
    #     mid_lichen = mid_lichen.sample(n=max_high, random_state=42)

    # # Concatène et sauvegarde
    # balanced = pd.concat([high_lichen, mid_lichen, low_lichen], ignore_index=True)
    balanced.to_csv(csv_out, index=False)
    print(f"CSV filtré et équilibré sauvegardé dans {csv_out} ({len(balanced)} lignes)")


def filter_and_balance_lichen(csv_in, csv_out, max_high=250, col="proportion_lichen"):
    df = pd.read_csv(csv_in)
    # On ne garde que les lignes où sqrt_proportion_lichen est défini
    df = df[df[col].notnull()]

    lichen = []
    for lichen_proportion in np.arange(0, 1, 0.01):
        new_lichen = df[(df[col] > lichen_proportion) & (df[col] <= lichen_proportion+0.1)]
        if len(new_lichen) > max_high:  
            new_lichen = new_lichen.sample(n=max_high, random_state=42)
        lichen.append(new_lichen)
        
    balanced = pd.concat([l for l in lichen], ignore_index=True)
    
    # # Prend au maximum max_high tuiles à très forte proportion de lichen
    # if len(high_lichen) > max_high:
    #     high_lichen = high_lichen.sample(n=max_high, random_state=42)
    #     mid_lichen = mid_lichen.sample(n=max_high, random_state=42)

    # # Concatène et sauvegarde
    # balanced = pd.concat([high_lichen, mid_lichen, low_lichen], ignore_index=True)
    balanced.to_csv(csv_out, index=False)
    print(f"CSV filtré et équilibré sauvegardé dans {csv_out} ({len(balanced)} lignes)")

# # Exemple d'utilisation :
# if __name__ == "__main__":
    # transform_proportion_to_sqrt(
    #     "data/samples/selection5/lichen_proportion.csv",
    #     "data/samples/selection5/lichen_sqrt_proportion.csv"
    # )
#     transform_proportion_to_log(
#         "data/samples/selection5/lichen_proportion.csv",
#         "data/samples/selection5/lichen_log_proportion.csv"
#     )
#     plot_lichen_proportion_histogram(
#         "data/samples/selection5/lichen_log_proportion.csv",
#         "data/samples/selection5/hist_lichen_proportion_log.png",
#     log=True
#     )
#     plot_lichen_proportion_histogram(
#         "data/samples/selection5/lichen_sqrt_proportion.csv",
#         "data/samples/selection5/hist_lichen_proportion_sqrt.png",
#     sqrt=True
#     )

def compute_proportion(classif, sentinel_path):
    """
    Compute the proportion of each class (lichen, chicoutai, crevasses, sphaignes) per Sentinel pixel.
    
    Args:
        classif: Path to the classification mask (values: 1=lichen, 2=chicoutai, 3=crevasse, 4=sphaignes, 0=other, 255=nodata)
        sentinel_path: Path to any Sentinel-2 band to get dimensions and geotransform
        
    Returns:
        Dictionary with (col_s, row_s) as keys and a dictionary of class proportions as values:
        {(col_s, row_s): {'lichen': float, 'chicoutai': float, 'crevasse': float, 'sphaignes': float, 
                          'valid_pixels': int, 'valid_proportion': float, 'total_pixels': int}}
    """
    # Load the classification mask
    ds_mask = gdal.Open(classif)
    mask = ds_mask.GetRasterBand(1).ReadAsArray()
    ds_5m = gdal.Open(sentinel_path)
    cols_5m = ds_5m.RasterXSize
    rows_5m = ds_5m.RasterYSize

    result_dict = {}
    
    # Class mapping
    class_ids = {
        1: 'lichen',
        2: 'chicoutai',
        3: 'crevasse',
        4: 'sphaignes'
    }

    for row_s in tqdm(range(rows_5m), desc="Computing class proportions"):
        for col_s in range(cols_5m):
            xmin, xmax, ymin, ymax = sentinel_to_drone_bounds(col_s, row_s, ds_5m=ds_5m)
            
            # Check if the window is within the bounds of the mask
            if xmin < 0 or ymin < 0 or xmax > mask.shape[1] or ymax > mask.shape[0]:
                result_dict[(col_s, row_s)] = None
                continue
                
            submask = mask[ymin:ymax, xmin:xmax]
            
            # Skip if submask is empty
            if submask.size == 0:
                result_dict[(col_s, row_s)] = None
                continue
            
            # Count valid pixels (non-zero and non-nodata)
            total_pixels = submask.size
            valid_pixels = np.sum((submask != 0) & (submask != 255))
            valid_proportion = valid_pixels / total_pixels if total_pixels > 0 else 0
            
            # If there are no valid pixels, skip
            if valid_pixels == 0:
                result_dict[(col_s, row_s)] = None
                continue
            
            # Initialize proportions dictionary for this sentinel pixel
            proportions = {
                'valid_pixels': valid_pixels,
                'valid_proportion': valid_proportion,
                'total_pixels': total_pixels
            }
            
            # Calculate proportion for each class
            for class_id, class_name in class_ids.items():
                count = np.sum(submask == class_id)
                prop = count / valid_pixels if valid_pixels > 0 else 0
                proportions[class_name] = prop
            
            result_dict[(col_s, row_s)] = proportions

    return result_dict

def save_proportions_to_csv(result_dict, out_csv):
    """
    Save the proportions dictionary to a CSV file.
    
    Args:
        result_dict: Dictionary with class proportions from compute_proportion()
        out_csv: Path to save the CSV file
    """
    rows = []
    
    for (col_s, row_s), props in result_dict.items():
        if props is None:
            # No valid data for this pixel
            row = {
                "col_s": col_s,
                "row_s": row_s,
                "lichen": None,
                "chicoutai": None,
                "crevasse": None,
                "sphaignes": None,
                "valid_pixels": 0,
                "valid_proportion": 0.0,
                "total_pixels": 0
            }
        else:
            # Valid data with proportions
            row = {
                "col_s": col_s,
                "row_s": row_s,
                "lichen": props.get('lichen', 0),
                "chicoutai": props.get('chicoutai', 0),
                "crevasse": props.get('crevasse', 0),
                "sphaignes": props.get('sphaignes', 0),
                "valid_pixels": props.get('valid_pixels', 0),
                "valid_proportion": props.get('valid_proportion', 0.0),
                "total_pixels": props.get('total_pixels', 0)
            }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Proportions saved to {out_csv}")

def plot_class_proportions_histogram(csv_path, out_dir=None):
    """
    Plot histograms of class proportions from CSV file
    
    Args:
        csv_path: Path to the CSV file with class proportions
        out_dir: Directory to save the plots (None to display only)
    """
    df = pd.read_csv(csv_path)
    
    # Classes to plot
    classes = ['lichen', 'chicoutai', 'crevasse', 'sphaignes']
    colors = ['lightgray', 'darkgreen', 'dimgray', 'brown']
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        # Drop NaN values
        values = df[class_name].dropna() * 100  # Convert to percentage
        
        # Create histogram
        axs[i].hist(values, bins=20, color=color, edgecolor='black', alpha=0.7)
        axs[i].set_title(f"Distribution of {class_name} proportion")
        axs[i].set_xlabel("Proportion (%)")
        axs[i].set_ylabel("Number of sentinel pixels")
        axs[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "class_proportions_histogram.png")
        plt.savefig(out_path)
        print(f"Histogram saved to {out_path}")
    else:
        plt.show()


def mask_invalid(csv_in, csv_out, min_valid_proportion=0.5, show=False, out_png=None):
    """
    Filter out Sentinel pixels with less than the specified proportion of valid drone pixels.
    
    Args:
        csv_in: Input CSV file with class proportions
        csv_out: Output CSV file with filtered data
        min_valid_proportion: Minimum proportion of valid pixels required to keep a sample (0-1)
        show: Whether to show a visualization of the mask
        out_png: Path to save the visualization as PNG
    """
    # Load the CSV file
    df = pd.read_csv(csv_in)
    
    # Count rows before filtering
    total_rows = len(df)
    
    # Filter by valid pixel proportion
    df_filtered = df[df['valid_proportion'] >= min_valid_proportion]
    
    # Count rows after filtering
    kept_rows = len(df_filtered)
    
    print(f"Filtered from {total_rows} to {kept_rows} pixels " +
          f"({kept_rows/total_rows*100:.1f}%) having at least {min_valid_proportion*100:.0f}% valid pixels")
    
    # Save the filtered DataFrame
    df_filtered.to_csv(csv_out, index=False)
    print(f"Saved filtered data to {csv_out}")
    
    # Create a visualization if requested
    if show or out_png:
        try:
            # Use a non-interactive backend to avoid display issues
            import matplotlib
            matplotlib.use('Agg')  # Use the 'Agg' backend that doesn't require a GUI
            
            # Get sentinel dimensions from the data
            max_row = int(max(max(df['row_s']), max(df_filtered['row_s']))) + 1
            max_col = int(max(max(df['col_s']), max(df_filtered['col_s']))) + 1
            
            # Create mask arrays
            all_mask = np.zeros((max_row, max_col), dtype=np.uint8)
            valid_mask = np.zeros((max_row, max_col), dtype=np.uint8)
            
            # Fill masks
            for _, row in df.iterrows():
                all_mask[int(row['row_s']), int(row['col_s'])] = 1
            
            for _, row in df_filtered.iterrows():
                valid_mask[int(row['row_s']), int(row['col_s'])] = 1
            
            # Create the figure with two plots side by side
            plt.figure(figsize=(10, 4))
            
            # Plot all pixels with data
            plt.subplot(1, 2, 1)
            plt.imshow(all_mask, cmap='gray')
            plt.title("All pixels with data")
            plt.axis('off')
            
            # Plot the mask of retained pixels
            plt.subplot(1, 2, 2)
            plt.imshow(valid_mask, cmap='gray')
            plt.title(f'Pixels with ≥{min_valid_proportion*100:.0f}% Valid Data')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save the figure if requested
            if out_png:
                plt.savefig(out_png, bbox_inches='tight', dpi=150)
                print(f"Visualization saved to {out_png}")
            
            # Show the figure if requested (might not work in non-interactive environments)
            if show:
                plt.show()
            else:
                plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            print("Continuing without visualization.")