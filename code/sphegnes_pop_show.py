from samples_set import SamplesSet
import glob
import os

def show_samples():
    """
    Affiche les échantillons de sphegnes.
    """
    # Chemin vers les fichiers .json de sphegnes
    path = "data/samples/sphegnes/echantillon1"
    json_files = sorted(glob.glob(os.path.join(path, "*.json")))

    # Charger toutes les populations
    sets = [SamplesSet.load_samples_from_json(f) for f in json_files]

    # Fusionner toutes les populations dans une seule liste de samples
    merged_set = sets[0]
    for s in sets[1:]:
        merged_set = SamplesSet.concatenate_set(merged_set, s)

    # Afficher la liste obtenue
    merged_set.plot_samples_as_list()
    # Sauvegarder la liste obtenue
    merged_set.save_samples_to_json(filename=path + "merged_sphegnes.json")

import json
from sample import Sample

def update_rgbz_variance_in_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    for sample_dict in data["samples"]:
        # On récupère les infos nécessaires pour créer un Sample
        i_x = sample_dict["i_x"]
        i_y = sample_dict["i_y"]
        x = sample_dict["x"]
        y = sample_dict["y"]
        size_patch = sample_dict.get("size_patch", 32)
        category = sample_dict.get("category", None)

        # Création du Sample (cela calcule les stats à partir du raster)
        s = Sample(i_x, i_y, x, y, size_patch=size_patch, category=category)

        # Mise à jour des variances dans le dictionnaire
        sample_dict["r_var"] = float(s.r_var)
        sample_dict["g_var"] = float(s.g_var)
        sample_dict["b_var"] = float(s.b_var)
        sample_dict["z_var"] = float(s.z_var)

    # Sauvegarde dans le même fichier
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

# # Utilisation :
# if __name__ == "__main__":
#     update_rgbz_variance_in_json("data/samples/lichen_sphegnes_selection/lichen_sphegnes_selection.json")