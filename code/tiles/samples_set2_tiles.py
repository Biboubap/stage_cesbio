from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from sample2 import Sample2 
import json
from rasters_manager import RastersManager

class SamplesSet2:
    """
    Gère un ensemble de Sample2, leur création, calcul des voisins, sauvegarde/chargement, etc.
    """
    def __init__(self, ds_path=None, dz_path=None, dt_path=None, n_samples_x=None, n_samples_y=None):
        # Dimensions de la grille
        self.n_samples_x = n_samples_x
        self.n_samples_y = n_samples_y
        self.samples = {}
        
        # Initialize the RastersManager and set paths
        self.rasters = RastersManager()
        self.rasters.set_paths(ds_path, dz_path, dt_path)
        
        # Store paths for later reference
        self.ds_path = ds_path
        self.dz_path = dz_path
        self.dt_path = dt_path

    def create_samples_grid(self, x_start, y_start, size_patch, category=None):
        """
        Crée une grille régulière de samples à partir des rasters chargés.
        """
        if self.n_samples_x is None or self.n_samples_y is None:
            raise ValueError("n_samples_x and n_samples_y must be defined before creating a grid.")
        
        # Ensure rasters are loaded
        self.rasters.load_rasters()
        print("Rasters chargés sur la tuile.")
        self.samples = {}
        for i_x in range(self.n_samples_x):
            for i_y in range(self.n_samples_y):
                x = x_start + i_x * size_patch
                y = y_start + i_y * size_patch
                sample = Sample2(
                    i_x, i_y, x, y, size_patch, category=category
                )
                self.samples[(x, y)] = sample

    def fill_neighbors_all(self, distance_large=3):
        """
        Calcule et remplit les statistiques des voisins pour tous les samples du set.
        
        Args:
            distance_large: Distance pour le voisinage large. Si 0, les statistiques _large_mean
                          ne seront pas calculées.
        """
        for sample in self.samples.values():
            sample.compute_neighbors_all(sample_set=self, distance_large=distance_large)

    def __repr__(self):
        # Affichage synthétique de l'ensemble
        return (f"SamplesSet2(n_samples_x={self.n_samples_x}, n_samples_y={self.n_samples_y}, n_total={len(self.samples)},\n"
                f"  ds_path={self.ds_path}, dz_path={self.dz_path}, dt_path={self.dt_path})")

    def plot_samples(self):
        """
        Affiche la grille de samples sous forme d'images RGB.
        """
        if self.n_samples_x is None or self.n_samples_y is None:
            raise ValueError("n_samples_x and n_samples_y must be defined before plotting samples.")
        matrix = [[None for _ in range(self.n_samples_x)] for _ in range(self.n_samples_y)]
        for sample in self.samples.values():
            matrix[sample.i_y][sample.i_x] = sample
        fig, axes = plt.subplots(self.n_samples_x, self.n_samples_y, figsize=(3*self.n_samples_y, 3*self.n_samples_x))
        for i_x in range(self.n_samples_x):
            for i_y in range(self.n_samples_y):
                sample = matrix[i_y][i_x]
                r, g, b, z, t = sample.get_RGBZT()
                rgb = np.dstack((r, g, b))
                ax = axes[i_x, i_y] if self.n_samples_y > 1 and self.n_samples_x > 1 else axes[max(i_y, i_x)]
                ax.imshow(rgb)
                ax.axis("off")
        plt.tight_layout()

    def get_samples_matrix(self):
        """
        Retourne la matrice 2D (i_y, i_x) des samples.
        """
        if self.n_samples_x is None or self.n_samples_y is None:
            raise ValueError("n_samples_x and n_samples_y must be defined before getting the samples matrix.")
        matrix = [[None for _ in range(self.n_samples_x)] for _ in range(self.n_samples_y)]
        for sample in self.samples.values():
            i_x = sample.i_x
            i_y = sample.i_y
            matrix[i_y][i_x] = sample
        return matrix

    def remove_Sample2(self, i_x, i_y):
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
            r, g, b, z, t = sample.get_RGBZT()
            rgb = np.dstack((r, g, b))
            ax.imshow(rgb)
            ax.set_title(f"i_x={sample.i_x}, i_y={sample.i_y}")
            ax.axis("off")

        for idx in range(n, n_rows * n_cols):
            i_row = idx // n_cols
            i_col = idx % n_cols
            axes[i_row, i_col].axis("off")

        plt.tight_layout()
        #plt.show()

    def save_samples_to_json(self, filename):
        """
        Sauvegarde le set et les samples dans un fichier JSON.
        (Ne sauvegarde pas les patchs d'image, seulement les attributs scalaires et les chemins)
        """
        data = {
            "n_samples_x": self.n_samples_x,
            "n_samples_y": self.n_samples_y,
            "ds_path": self.ds_path if self.ds_path is not None else None,
            "dz_path": self.dz_path if self.dz_path is not None else None,
            "dt_path": self.dt_path if self.dt_path is not None else None,
            "samples": []
        }
        for s in self.samples.values():
            sample_dict = {
                "i_x": s.i_x,
                "i_y": s.i_y,
                "size_patch": s.size_patch,
                "x": getattr(s, "x", None),
                "y": getattr(s, "y", None),
                "category": getattr(s, "category", None),
                "r_mean": getattr(s, "r_mean", None),
                "g_mean": getattr(s, "g_mean", None),
                "b_mean": getattr(s, "b_mean", None),
                "t_mean": getattr(s, "t_mean", None),
                "z_mean": getattr(s, "z_mean", None),
                "r_var": getattr(s, "r_var", None),
                "g_var": getattr(s, "g_var", None),
                "b_var": getattr(s, "b_var", None),
                "t_var": getattr(s, "t_var", None),
                "z_var": getattr(s, "z_var", None),
                "r_n_mean": getattr(s, "r_n_mean", None),
                "g_n_mean": getattr(s, "g_n_mean", None),
                "b_n_mean": getattr(s, "b_n_mean", None),
                "t_n_mean": getattr(s, "t_n_mean", None),
                "z_moins_z_n": getattr(s, "z_moins_z_n", None),
                # Features avec le voisinage large renommées
                "r_large_mean": getattr(s, "r_large_mean", None),
                "g_large_mean": getattr(s, "g_large_mean", None),
                "b_large_mean": getattr(s, "b_large_mean", None),
                "t_large_mean": getattr(s, "t_large_mean", None),
                "z_moins_z_large": getattr(s, "z_moins_z_large", None),
            }
            data["samples"].append(sample_dict)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_samples_from_json(filename):
        """
        Charge un set de samples depuis un fichier JSON (statistiques uniquement, pas les patchs d'image).
        Recharge les rasters à partir des chemins sauvegardés.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        samples_set = SamplesSet2(
            ds_path=data.get("ds_path", None),
            dz_path=data.get("dz_path", None),
            dt_path=data.get("dt_path", None),
            n_samples_x=data.get("n_samples_x", None),
            n_samples_y=data.get("n_samples_y", None)
        )
        
        # Make sure rasters are loaded
        samples_set.rasters.load_rasters()
        
        for s in data["samples"]:
            sample = Sample2(
                i_x=s["i_x"],
                i_y=s["i_y"],
                x=s["x"],
                y=s["y"],
                size_patch=s["size_patch"],
                category=s.get("category", None)
            )
            for attr in [
                "r_var", "g_var", "b_var", "t_var", "z_var",
                "r_mean", "g_mean", "b_mean", "t_mean", "z_mean",  
                "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean", "z_moins_z_n",
                "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean", "z_moins_z_large"
            ]:
                setattr(sample, attr, s.get(attr, None))
            samples_set.add_Sample(sample)
        return samples_set

    @staticmethod
    def concatenate_set(set1, set2):
        """
        Concatène deux SamplesSet en une seule liste de samples, tous placés sur la même "ligne" (i_y=0).
        Les indices i_x sont réindexés pour que tous les samples soient à la suite sur la première ligne.
        Le nouveau set n'a pas de n_samples_x/n_samples_y défini.
        Les chemins des rasters sont hérités de set1 si possible, sinon de set2.
        """
        ds_path = set1.ds_path if set1.ds_path is not None else set2.ds_path
        dz_path = set1.dz_path if set1.dz_path is not None else set2.dz_path
        dt_path = set1.dt_path if set1.dt_path is not None else set2.dt_path

        new_set = SamplesSet2(ds_path=ds_path, dz_path=dz_path, dt_path=dt_path, n_samples_x=None, n_samples_y=None)
        
        samples_list = list(set1.samples.values()) + list(set2.samples.values())
        for new_i_x, s in enumerate(samples_list):
            s_copy = Sample2(
                i_x=new_i_x, i_y=0, x=s.x, y=s.y, size_patch=s.size_patch,
                category=getattr(s, "category", None)
            )
            for attr in [
                "r_mean", "g_mean", "b_mean", "t_mean", 
                "r_var", "g_var", "b_var", "z_var", "t_var",
                "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean", "z_moins_z_n",
                "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean", "z_moins_z_large"
            ]:
                setattr(s_copy, attr, getattr(s, attr, None))
            new_set.samples[(s_copy.x, s_copy.y)] = s_copy
        return new_set

    def clear_rasters(self):
        """Libère la mémoire des rasters chargés"""
        self.rasters.clear_rasters()
    
    def add_Sample(self, sample):
        """
        Ajoute un sample à l'ensemble (clé = (x, y)).
        Fonction pour compatibilité avec le code existant.
        """
        self.samples[(sample.x, sample.y)] = sample

    
if __name__ == "__main__":
    # Initialize with paths
    ds_path = "data/rgb_reshaped.tif"
    dz_path = "data/dsm_reshaped.tif"
    dt_path = "data/thermal_reshaped.tif"
   
    print("Test de la classe SamplesSet")

    # Créer un ensemble de samples
    x_1 = 10000
    y_1 = 10000
    x_2 = 11000
    y_2 = 11000
    row_start = int(np.round(x_1/32))*32 #row X
    column_start = int(np.round(y_1/32))*32 #column Y
    n_samples_x = 3
    n_samples_y = 3
    
    print("Samples_set 1")
    size_patch= 32
    samples_set = SamplesSet2(ds_path, dz_path, dt_path, n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    samples_set.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch, category="tourbiere")
    samples_set.fill_neighbors_all()
    samples_set.plot_samples()
    # plt.show()
    print("Supprimer un sample")
    samples_set.remove_Sample2(2, 2)
    samples_set.plot_samples_as_list()
    # plt.show()
    samples_set.clear_rasters()  

    print("Samples_set 2")
    n_samples_x = 2
    n_samples_y = 2
    row_start = int(np.round(x_2/32))*32 #row X
    column_start = int(np.round(y_2/32))*32 #column Y
    samples_set2 = SamplesSet2(ds_path, dz_path, dt_path, n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    samples_set2.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch, category="tourbière")
    samples_set2.fill_neighbors_all()
    samples_set2.plot_samples_as_list()
    samples_set2.clear_rasters() 

    print("Samples_set 1 + Samples_set 2")
    merged_set = SamplesSet2.concatenate_set(samples_set, samples_set2)
    merged_set.plot_samples_as_list()
    merged_set.clear_rasters()  # Libération de mémoire des rasters
    #plt.show()

   
    print("Sauvegarde et chargement des samples")
    # Sauvegarder les samples dans un fichier json
    path = "data/samples/"
    merged_set.save_samples_to_json(path+"testT.json")
    print(f"Samples sauvegardés dans {path}testT.json")
    

    #  Charger les samples depuis le fichier json
    loaded_set = SamplesSet2.load_samples_from_json(path+"testT.json")
    print(f"Samples chargés depuis {path}testT.json")
    loaded_set.plot_samples_as_list()
    plt.show()