import numpy as np
import matplotlib.pyplot as plt
import json

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
                x = x_start + i_x * size_patch
                y = y_start + i_y * size_patch
                sample = Sample(i_x, i_y, x, y, size_patch, self.category)
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
        fig, axes = plt.subplots(self.n_samples_x, self.n_samples_y, figsize=(3*self.n_samples_y, 3*self.n_samples_x))
        
        for i_x in range(self.n_samples_x):
            for i_y in range(self.n_samples_y):
                # x = ligne (row), y = colonne (col)
                sample = matrix[i_y][i_x]  # x (row) = i_y, y (col) = i_x
                r, g, b, _ = sample.get_RGBZ()
                rgb = np.dstack((r, g, b))
                ax = axes[i_x, i_y] if self.n_samples_y > 1 and self.n_samples_x > 1 else axes[max(i_y, i_x)]
                ax.imshow(rgb)
                #ax.set_title(f"x = {sample.x}, y = {sample.y}")
                ax.axis("off")
        plt.tight_layout()
        plt.show()


    def save_samples_to_json(self, filename):
        """
        Sauvegarde le set et les samples dans un fichier JSON.
        (Ne sauvegarde pas les patchs d'image, seulement les attributs scalaires)
        """
        data = {
            "n_samples_x": self.n_samples_x,
            "n_samples_y": self.n_samples_y,
            "category": self.category,
            "samples": []
        }
        for s in self.samples:
            sample_dict = {
                "i_x": s.i_x,
                "i_y": s.i_y,
                "size_patch": s.size_patch,
                "x": getattr(s, "x", None),
                "y": getattr(s, "y", None),
                "r_mean": getattr(s, "r_mean", None),
                "g_mean": getattr(s, "g_mean", None),
                "b_mean": getattr(s, "b_mean", None),
                "r_n_mean": getattr(s, "r_n_mean", None),
                "g_n_mean": getattr(s, "g_n_mean", None),
                "b_n_mean": getattr(s, "b_n_mean", None),
                "delta_z_x": getattr(s, "delta_z_x", None),
                "delta_z_y": getattr(s, "delta_z_y", None)
            }
            data["samples"].append(sample_dict)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_samples_from_json(filename):
        """
        Charge un set de samples depuis un fichier JSON (statistiques uniquement, pas les patchs d'image).
        Retourne une instance de SamplesSet.
        """
        from sample import Sample
        with open(filename, "r") as f:
            data = json.load(f)
        samples_set = SamplesSet(
            n_samples_x=data["n_samples_x"],
            n_samples_y=data["n_samples_y"],
            category=data.get("category", None)
        )
        for s in data["samples"]:
            sample = Sample(
                i_x=s["i_x"],
                i_y=s["i_y"],
                x=s["x"],
                y=s["y"],
                size_patch=s["size_patch"]
            )
            sample.x = s.get("x", None)
            sample.y = s.get("y", None)
            sample.r_mean = s.get("r_mean", None)
            sample.g_mean = s.get("g_mean", None)
            sample.b_mean = s.get("b_mean", None)
            sample.r_n_mean = s.get("r_n_mean", None)
            sample.g_n_mean = s.get("g_n_mean", None)
            sample.b_n_mean = s.get("b_n_mean", None)
            sample.delta_z_x = s.get("delta_z_x", None)
            sample.delta_z_y = s.get("delta_z_y", None)
            sample.classification=s.get("classification", None)
            samples_set.add_sample(sample)
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


    # # # Sauvegarder les samples dans un fichier json
    # path = "data/samples/"
    # samples_set.save_samples_to_json(path+"test.json")
    # # # Charger les samples depuis le fichier json
    # loaded_set = SamplesSet.load_samples_from_json(path+"test.json")
    # loaded_set.plot_samples()