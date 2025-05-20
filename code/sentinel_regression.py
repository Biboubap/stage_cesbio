import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from interaction_sentinel_drone import load_proportion_csv, load_sentinel_rgb_bands, mask_interior_pixels
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def plot_regressions_rgb_vs_lichen(csv_path, sentinel_path, out_png, distance_bord=0):
    # Charge les données
    df = load_proportion_csv(csv_path)
    r, g, b = load_sentinel_rgb_bands(sentinel_path)

    # Ne garde que les pixels avec une proportion définie (non None et non NaN)
    df = df[df["proportion_lichen"].notnull()]

    # Filtre les pixels "intérieurs" selon la distance au bord
    if distance_bord > 0:
        mask = mask_interior_pixels(df, distance=distance_bord)
        df = df[mask]

    # Récupère les valeurs RGB et la proportion pour chaque pixel Sentinel
    reds, greens, blues, means, props = [], [], [], [], []
    for _, row in df.iterrows():
        col_s = int(row["col_s"])
        row_s = int(row["row_s"])
        prop = row["proportion_lichen"]
        reds.append(r[row_s, col_s])
        greens.append(g[row_s, col_s])
        blues.append(b[row_s, col_s])
        means.append(np.mean([r[row_s, col_s], g[row_s, col_s], b[row_s, col_s]]))
        props.append(prop)

    X = np.array(props).reshape(-1, 1)
    Y = [np.array(reds), np.array(greens), np.array(blues), np.array(means)]
    colors = ['red', 'green', 'blue', 'gray']
    titles = [
        "Rouge Sentinel vs proportion de lichen",
        "Vert Sentinel vs proportion de lichen",
        "Bleu Sentinel vs proportion de lichen",
        "Moyenne RGB Sentinel vs proportion de lichen"
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, (y, color, title) in enumerate(zip(Y, colors, titles)):
        # Régression linéaire
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        a = model.coef_[0]
        b = model.intercept_
        r2 = r2_score(y, y_pred)

        axs[i].scatter(props, y, color=color, alpha=0.5, s=5, label="Données")
        axs[i].plot(np.sort(props), model.predict(np.sort(X, axis=0)), color='black', lw=2, label="Régression")
        axs[i].set_title(title)
        axs[i].set_xlabel("Proportion de lichen")
        axs[i].set_ylabel(f"Valeur {color.capitalize() if color != 'gray' else 'Moyenne RGB'}")
        axs[i].legend()
        axs[i].text(
            0.05, 0.95,
            f"y = {a:.2f}x + {b:.2f}\n$R^2$ = {r2:.3f}",
            transform=axs[i].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

    plt.suptitle(f"Régressions linéaires RGB Sentinel2 vs proportion de lichen (distance_bord={distance_bord})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png)
    plt.close()
    print(f"Graphe sauvegardé dans {out_png}")

# # Exemple d'utilisation :
# if __name__ == "__main__":
#     distance_bord = 2  # ou 0 pour tout prendre
#     plot_regressions_rgb_vs_lichen(
#         csv_path="data/sentinel2/sentinel_analysis/lichen_proportion_per_sentinel_pixel.csv",
#         sentinel_path="data/sentinel2/rgb/databand1_reshaped.tif",
#         out_png=f"data/sentinel2/sentinel_analysis/rgb_vs_lichen_regression_distance{distance_bord}.png",
#         distance_bord=distance_bord  # ou 0 pour tout prendre
#     )
def random_forest_regression_lichen(csv_path, sentinel_path, out_png, distance_bord=0, show_mask=False):
    # Charge les données
    df = load_proportion_csv(csv_path)
    r, g, b = load_sentinel_rgb_bands(sentinel_path)

    # Ne garde que les pixels avec une proportion définie (non None et non NaN)
    df = df[df["proportion_lichen"].notnull()]

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
        X.append([r[row_s, col_s], g[row_s, col_s], b[row_s, col_s]])
        y.append(row["proportion_lichen"])
    X = np.array(X)
    y = np.array(y)

    # Sépare en train/test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Modèle Random Forest avec les paramètres demandés
    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Plot prédiction vs vérité terrain (sur le test uniquement)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.5, s=8, label="Prédictions (test)")
    plt.plot([0, 1], [0, 1], 'k--', label="y = x")
    plt.xlabel("Proportion de lichen réelle")
    plt.ylabel("Proportion de lichen prédite (RF)")
    plt.title(f"Random Forest : Prédiction de la proportion de lichen\nR² (test) = {r2:.3f} (distance_bord={distance_bord})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Graphe RF sauvegardé dans {out_png}")


# Exemple d'utilisation :
if __name__ == "__main__":
    distance_bord = 0  # ou 0 pour tout prendre
    random_forest_regression_lichen(
        csv_path="data/sentinel2/sentinel_analysis/lichen_proportion_per_sentinel_pixel.csv",
        sentinel_path="data/sentinel2/rgb/databand1_reshaped.tif",
        out_png=f"data/sentinel2/sentinel_analysis/rf_vs_lichen_proportion2_distance{distance_bord}.png",
        distance_bord = distance_bord,  # ou 0 pour tout prendre
        show_mask=False
    )