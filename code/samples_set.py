import numpy as np
import matplotlib.pyplot as plt

from sample import Sample
import pandas as pd

class SamplesSet:
    def __init__(self, n_samples_x, n_samples_y, category=None):
        self.n_samples_x = n_samples_x
        self.n_samples_y = n_samples_y
        self.category = category
        # Liste 2D ordonnée selon x croissant (colonnes) et y croissant (lignes)
        self.samples = []

    def add_sample(self, sample):
        """Ajoute un sample à l'ensemble, puis trie selon x et y croissants."""
        self.samples.append(sample)
        # Trie d'abord par y puis par x
        self.samples.sort(key=lambda s: (s.y, s.x))

    def create_samples_grid(self, x_start, y_start, size_patch=32):
        """Crée une grille régulière de samples ordonnés selon x et y croissants."""
        self.samples = []
        for i_x in range(self.n_samples_x):     
            for i_y in range(self.n_samples_y):  
                sample = Sample(i_x, i_y, x_start, y_start, size_patch)
                self.samples.append(sample)
        # Trie pour garantir l'ordre (y, x)
        self.samples.sort(key=lambda s: (s.y, s.x))

    def get_samples_matrix(self):
        """Retourne les samples sous forme de matrice [n_samples_y][n_samples_x]."""
        matrix = [[None for _ in range(self.n_samples_x)] for _ in range(self.n_samples_y)]
        for sample in self.samples:
            i_x = sample.i_x
            i_y = sample.i_y
            matrix[i_y][i_x] = sample
        return matrix

    def __repr__(self):
        return f"SamplesSet(n_samples_x={self.n_samples_x}, n_samples_y={self.n_samples_y}, category={self.category}, n_total={len(self.samples)})"
    
    def plot_samples(self):
        """
        Affiche tous les samples du set en RGB, ordonnés comme une image :
        - x (rows) croissants de haut en bas
        - y (cols) croissants de gauche à droite
        """
        matrix = self.get_samples_matrix()
        fig, axes = plt.subplots(self.n_samples_y, self.n_samples_x, figsize=(3*self.n_samples_x, 3*self.n_samples_y))
        
        for i_x in range(self.n_samples_x):
            for i_y in range(self.n_samples_y):
                # x = ligne (row), y = colonne (col)
                sample = matrix[i_y][i_x]  # x (row) = i_y, y (col) = i_x
                r, g, b, _ = sample.get_RGBZ()
                rgb = np.dstack((r, g, b))
                ax = axes[i_x, i_y] if self.n_samples_y > 1 and self.n_samples_x > 1 else axes[max(i_y, i_x)]
                ax.imshow(rgb)
                ax.set_title(f"x = {sample.x}, y = {sample.y}")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

    def save_samples_to_csv(self, filename):
        """
        Sauvegarde les caractéristiques du set et les statistiques des samples dans un fichier CSV.
        (Ne sauvegarde pas les patchs d'image, seulement les attributs scalaires)
        """
        import pandas as pd
        # Prépare la première ligne pour les infos du set
        set_info = {
            "i_x": "SET_INFO",
            "i_y": "",
            "x_start": "",
            "y_start": "",
            "size_patch": "",
            "r_mean": "",
            "g_mean": "",
            "b_mean": "",
            "r_n_mean": "",
            "g_n_mean": "",
            "b_n_mean": "",
            "delta_z_x": "",
            "delta_z_y": "",
            "n_samples_x": self.n_samples_x,
            "n_samples_y": self.n_samples_y,
            "category": self.category
        }
        data = [set_info]
        # Puis les samples
        for s in self.samples:
            data.append({
                "i_x": s.i_x,
                "i_y": s.i_y,
                "x_start": s.x_start,
                "y_start": s.y_start,
                "size_patch": s.size_patch,
                "r_mean": s.r_mean,
                "g_mean": s.g_mean,
                "b_mean": s.b_mean,
                "r_n_mean": s.r_n_mean,
                "g_n_mean": s.g_n_mean,
                "b_n_mean": s.b_n_mean,
                "delta_z_x": s.delta_z_x,
                "delta_z_y": s.delta_z_y,
                "n_samples_x": "",
                "n_samples_y": "",
                "category": ""
            })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    @staticmethod
    def load_samples_from_csv(filename):
        """
        Charge un set de samples depuis un fichier CSV (statistiques uniquement, pas les patchs d'image).
        Retourne une instance de SamplesSet.
        """
        import pandas as pd
        df = pd.read_csv(filename)
        # Récupère les infos du set dans la première ligne
        set_info = df.iloc[0]
        n_samples_x = int(set_info.get("n_samples_x", 1))
        n_samples_y = int(set_info.get("n_samples_y", 1))
        category = set_info.get("category", None)
        samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category=category)
        # Parcours les samples (à partir de la 2e ligne)
        for _, row in df.iloc[1:].iterrows():
            s = Sample(
                i_x=int(row["i_x"]),
                i_y=int(row["i_y"]),
                x_start=int(row["x_start"]),
                y_start=int(row["y_start"]),
                size_patch=int(row["size_patch"])
            )
            s.r_mean = row["r_mean"]
            s.g_mean = row["g_mean"]
            s.b_mean = row["b_mean"]
            s.r_n_mean = row["r_n_mean"]
            s.g_n_mean = row["g_n_mean"]
            s.b_n_mean = row["b_n_mean"]
            s.delta_z_x = row["delta_z_x"]
            s.delta_z_y = row["delta_z_y"]
            samples_set.add_sample(s)
        return samples_set
    
if __name__ == "__main__":
    # Créer un ensemble de samples
    x_1 = 12800
    y_1 = 12000
    x_2 = 19650
    y_2 = 11430
    row_start = int(np.round(x_1/32))*32 #row X
    column_start = int(np.round(y_1/32))*32 #column Y
    n_samples_x = 10
    n_samples_y = 10
    
    size_patch= 256
    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="tourbière")
    samples_set.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    samples_set.plot_samples()

    # # Sauvegarder les samples dans un fichier CSV
    # samples_set.save_samples_to_csv("data/samples/samples_set.csv")
    # # Charger les samples depuis le fichier CSV
    # loaded_samples_set = SamplesSet.load_samples_from_csv("data/samples/samples_set.csv")
    # loaded_samples_set.plot_samples()
    
