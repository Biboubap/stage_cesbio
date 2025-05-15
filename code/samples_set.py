import numpy as np
import matplotlib.pyplot as plt
import json

from sample import Sample
import pandas as pd

class SamplesSet:
    def __init__(self, n_samples_x=None, n_samples_y=None, category=None):
        self.n_samples_x = n_samples_x
        self.n_samples_y = n_samples_y
        self.category = category
        # Dictionnaire indexé par (x, y)
        self.samples = {}  # {(x, y): sample}

    def add_sample(self, sample):
        """Ajoute un sample à l'ensemble, indexé par (x, y)."""
        self.samples[(sample.x, sample.y)] = sample
        self.n_samples_x = None
        self.n_samples_y = None

    def create_samples_grid(self, x_start, y_start, size_patch=32):
        if self.n_samples_x is None or self.n_samples_y is None:
            raise ValueError("n_samples_x and n_samples_y must be defined before creating a grid.")
        self.samples = {}
        for i_x in range(self.n_samples_x):
            for i_y in range(self.n_samples_y):
                x = x_start + i_x * size_patch
                y = y_start + i_y * size_patch
                sample = Sample(i_x, i_y, x, y, size_patch, self.category)
                self.samples[(x, y)] = sample

    def get_samples_matrix(self):
        if self.n_samples_x is None or self.n_samples_y is None:
            raise ValueError("n_samples_x and n_samples_y must be defined before getting the samples matrix.")
        matrix = [[None for _ in range(self.n_samples_x)] for _ in range(self.n_samples_y)]
        for sample in self.samples.values():
            i_x = sample.i_x
            i_y = sample.i_y
            matrix[i_y][i_x] = sample
        return matrix

    def remove_sample(self, i_x, i_y):
        """
        Supprime le sample dont les indices sont i_x et i_y.
        Si un sample est supprimé, n_samples_x et n_samples_y sont mis à None.
        """
        to_remove = None
        for s in self.samples.values():
            if s.i_x == i_x and s.i_y == i_y:
                to_remove = (s.x, s.y)
                break
        if to_remove and to_remove in self.samples:
            del self.samples[to_remove]
            self.n_samples_x = None
            self.n_samples_y = None

    def __repr__(self):
        return f"SamplesSet(n_samples_x={self.n_samples_x}, n_samples_y={self.n_samples_y}, category={self.category}, n_total={len(self.samples)})"

    def plot_samples(self):
        if self.n_samples_x is None or self.n_samples_y is None:
            raise ValueError("n_samples_x and n_samples_y must be defined before plotting samples.")
        matrix = self.get_samples_matrix()
        fig, axes = plt.subplots(self.n_samples_x, self.n_samples_y, figsize=(3*self.n_samples_y, 3*self.n_samples_x))
        for i_x in range(self.n_samples_x):
            for i_y in range(self.n_samples_y):
                sample = matrix[i_y][i_x]
                r, g, b, _ = sample.get_RGBZ()
                rgb = np.dstack((r, g, b))
                ax = axes[i_x, i_y] if self.n_samples_y > 1 and self.n_samples_x > 1 else axes[max(i_y, i_x)]
                ax.imshow(rgb)
                ax.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_samples_as_list(self):
        """
        Affiche tous les samples du set en RGB, sans grille imposée.
        Les samples sont affichés dans une grille carrée aussi compacte que possible.
        """
        samples_list = list(self.samples.values())
        n = len(samples_list)
        if n == 0:
            print("Aucun sample à afficher.")
            return
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = np.array(axes).reshape(n_rows, n_cols)

        for idx, sample in enumerate(samples_list):
            i_row = idx // n_cols
            i_col = idx % n_cols
            ax = axes[i_row, i_col]
            r, g, b, _ = sample.get_RGBZ()
            rgb = np.dstack((r, g, b))
            ax.imshow(rgb)
            ax.set_title(f"i_x={sample.i_x}, i_y={sample.i_y}")
            ax.axis("off")

        for idx in range(n, n_rows * n_cols):
            i_row = idx // n_cols
            i_col = idx % n_cols
            axes[i_row, i_col].axis("off")

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
        for s in self.samples.values():
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
            sample.classification = s.get("classification", None)
            samples_set.add_sample(sample)
        return samples_set

    @staticmethod
    def concatenate_set(set1, set2):
        """
        Concatène deux SamplesSet en une seule liste de samples, tous placés sur la même "ligne" (i_y=0).
        Les indices i_x sont réindexés pour que tous les samples soient à la suite sur la première ligne.
        Le nouveau set n'a pas de n_samples_x/n_samples_y défini.
        """
        if set1.category != set2.category:
            category = "mixte"
        else:
            category = set1.category

        new_set = SamplesSet(n_samples_x=None, n_samples_y=None, category=category)
        samples_list = list(set1.samples.values()) + list(set2.samples.values())
        for new_i_x, s in enumerate(samples_list):
            s_copy = Sample(
                i_x=new_i_x,
                i_y=0,
                x=s.x,
                y=s.y,
                size_patch=s.size_patch,
                classification=getattr(s, "classification", None)
            )
            for attr in ["r_mean", "g_mean", "b_mean", "r_n_mean", "g_n_mean", "b_n_mean", "delta_z_x", "delta_z_y"]:
                setattr(s_copy, attr, getattr(s, attr, None))
            new_set.samples[(s_copy.x, s_copy.y)] = s_copy
        return new_set
    

if __name__ == "__main__":
    # Créer un ensemble de samples
    x_1 = 12800
    y_1 = 12000
    x_2 = 19650
    y_2 = 11430
    row_start = int(np.round(x_1/32))*32 #row X
    column_start = int(np.round(y_1/32))*32 #column Y
    n_samples_x = 3
    n_samples_y = 3
    
    size_patch= 256
    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="tourbière")
    samples_set.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    #samples_set.plot_samples_as_list()
    samples_set.remove_sample(2, 2)
    samples_set.plot_samples_as_list()

    # n_samples_x = 2
    # n_samples_y = 2
    # row_start = int(np.round(x_2/32))*32 #row X
    # column_start = int(np.round(y_2/32))*32 #column Y
    # samples_set2 = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="tourbière")
    # samples_set2.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    # samples_set2.plot_samples_as_list()

    # merged_set = SamplesSet.concatenate_set(samples_set, samples_set2)
    # merged_set.plot_samples_as_list()

    # # # Sauvegarder les samples dans un fichier json
    # path = "data/samples/"
    # samples_set.save_samples_to_json(path+"test.json")
    # # # Charger les samples depuis le fichier json
    # loaded_set = SamplesSet.load_samples_from_json(path+"test.json")
    # loaded_set.plot_samples()