import numpy as np
import matplotlib.pyplot as plt

from sample import Sample

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
        self.samples.sort(key=lambda s: (s.y_start, s.x_start))

    def create_samples_grid(self, x_start, y_start, size_patch=32):
        """Crée une grille régulière de samples ordonnés selon x et y croissants."""
        self.samples = []
        for i_x in range(self.n_samples_x):     
            for i_y in range(self.n_samples_y):  
                sample = Sample(i_x, i_y, x_start, y_start, size_patch)
                self.samples.append(sample)
        # Trie pour garantir l'ordre (y, x)
        self.samples.sort(key=lambda s: (s.y_start, s.x_start))

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
                ax.set_title(f"x = {sample.x_start}, y = {sample.y_start}")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Créer un ensemble de samples
    row_start = int(np.round(19650/32))*32 #row X
    column_start = int(np.round(11430/32))*32 #column Y
    n_samples_x = 10
    n_samples_y = 10
    size_patch= 32
    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="tourbière")
    samples_set.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    samples_set.plot_samples()
