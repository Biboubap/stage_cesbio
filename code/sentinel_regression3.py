import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from osgeo import gdal
import glob
import os

from interaction_sentinel_drone import load_proportion_csv, mask_interior_pixels

def load_all_sentinel_features(indices_dir, bands_dir):
    
    # Liste tous les fichiers *_reshaped.tif (hors databand1)
    bandes_files = sorted(glob.glob(os.path.join(bands_dir, "*.tif")))
    indices_files = sorted(glob.glob(os.path.join(indices_dir, "*.tif")))
    # S'assure que databand1 est en premier


    bands = []
    band_names = []
    for tif in bandes_files:
        ds = gdal.Open(tif)
        arr = ds.GetRasterBand(1).ReadAsArray()
        bands.append(arr)
        band_names.append(os.path.splitext(os.path.basename(tif))[0])
    for tif in indices_files:
        ds = gdal.Open(tif)
        arr = ds.GetRasterBand(1).ReadAsArray()
        bands.append(arr)
        band_names.append(os.path.splitext(os.path.basename(tif))[0])
    features = np.stack(bands, axis=0)  # (n_bands, rows, cols)
    return features, band_names


def random_forest_regression_lichen_multi(csv_path, indices, bandes, out_png, distance_bord=0, show_mask=False, sqrt=False):
    # Charge les données
    df = load_proportion_csv(csv_path)
    features, band_names = load_all_sentinel_features(indices_dir=indices, bands_dir=bandes)

    # Choix de la colonne cible selon sqrt
    target_col = "sqrt_proportion_lichen" if sqrt else "proportion_lichen"
    df = df[df[target_col].notnull()]

    # Filtre les pixels "intérieurs" selon la distance au bord
    if distance_bord > 0:
        mask = mask_interior_pixels(df, distance=distance_bord, show=show_mask)
        df = df[mask]

    # Prépare les features et la cible
    X = []
    y = []
    for _, row in df.iterrows():
        col_s = int(row["col_s"])
        row_s = int(row["row_s"])
        pix_features = features[:, row_s, col_s]
        X.append(pix_features)
        y.append(row[target_col])
    X = np.array(X)
    y = np.array(y)

    # Sépare en train/test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Modèle Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        random_state=42, 
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Si sqrt=True, on remet au carré la prédiction et la vérité terrain pour l'affichage
    if sqrt:
        y_pred_plot = y_pred
        y_test_plot = y_test
        ylabel = "SQRT(Proportion de lichen prédite)"
        xlabel = "SQRT(Proportion de lichen réelle)"
    else:
        y_pred_plot = y_pred
        y_test_plot = y_test
        ylabel = "Proportion de lichen prédite (RF)"
        xlabel = "Proportion de lichen réelle"

    r2 = r2_score(y_test_plot, y_pred_plot)

    # Plot prédiction vs vérité terrain (sur le test uniquement)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test_plot, y_pred_plot, alpha=0.5, s=8, label="Prédictions (test)")
    if sqrt:
        plt.plot([6, 10], [6, 10], 'k--', label="y = x")
        plt.xlim(6, 10)
        plt.ylim(6, 10)
    else:
        plt.plot([0, 1], [0, 1], 'k--', label="y = x")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Random Forest multi-bandes : Prédiction de la proportion de lichen\nR² (test) = {r2:.3f} (distance_bord={distance_bord})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Graphe RF multi-bandes sauvegardé dans {out_png}")
    print("Bandes utilisées :", band_names)

    # Affichage des importances des variables
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Importance des variables (features) :")
    for idx in sorted_idx:
        print(f"{band_names[idx]} : {importances[idx]:.4f}")

    # Optionnel : plot des importances
    plt.figure(figsize=(10, 4))
    plt.bar([band_names[i] for i in sorted_idx], importances[sorted_idx])
    plt.ylabel("Importance")
    plt.title("Importance des variables (Random Forest)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_feature_importance.png"))
    plt.close()
    print("Graphe des importances sauvegardé dans", out_png.replace(".png", "_feature_importance.png"))

# Exemple d'utilisation :
if __name__ == "__main__":
    distance_bord = 1  # ou autre valeur
    random_forest_regression_lichen_multi(
        csv_path="data/samples/selection8/regression/lichen_balanced_22.csv",
        bandes="DataCubeS2/Bandes/median",
        indices="DataCubeS2/Indices/median",
        out_png="data/samples/selection8/regression/multiband_and_indices_1.png",
        distance_bord=distance_bord,
        show_mask=True,
        sqrt=False
    )