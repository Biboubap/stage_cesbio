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

import json
import random

def balance_lichen_sphegnes(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    lichen_samples = [s for s in data["samples"] if s.get("category") == "lichen"]
    sphegnes_samples = [s for s in data["samples"] if s.get("category") == "sphegnes"]

    n_sphegnes = len(sphegnes_samples)
    n_lichen = len(lichen_samples)

    if n_lichen > n_sphegnes:
        lichen_samples = random.sample(lichen_samples, n_sphegnes)
    elif n_sphegnes > n_lichen:
        sphegnes_samples = random.sample(sphegnes_samples, n_lichen)
    # Sinon, déjà équilibré

    balanced_samples = lichen_samples + sphegnes_samples
    #random.shuffle(balanced_samples)

    balanced_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": balanced_samples
    }

    with open(output_path, "w") as f:
        json.dump(balanced_data, f, indent=2)

# Utilisation :

if __name__ == "__main__":
    balance_lichen_sphegnes(
        "data/samples/lichen_sphegnes_selection/lichen_sphegnes_selection.json",
        "data/samples/lichen_sphegnes_balanced/lichen_sphegnes_balanced.json"
    )

    # Affichage et sauvegarde des populations équilibrées
    import matplotlib.pyplot as plt

    with open("data/samples/lichen_sphegnes_balanced/lichen_sphegnes_balanced.json") as f:
        data = json.load(f)
    lichen_samples = [s for s in data["samples"] if s.get("category") == "lichen"]
    sphegnes_samples = [s for s in data["samples"] if s.get("category") == "sphegnes"]

    # Création des sets pour affichage
    from samples_set import SamplesSet
    from sample import Sample

    lichen_set = SamplesSet()
    for s in lichen_samples:
        lichen_set.add_sample(Sample(
            i_x=s["i_x"], i_y=s["i_y"], x=s["x"], y=s["y"],
            size_patch=s.get("size_patch", 32), category="lichen"
        ))
    sphegnes_set = SamplesSet()
    for s in sphegnes_samples:
        sphegnes_set.add_sample(Sample(
            i_x=s["i_x"], i_y=s["i_y"], x=s["x"], y=s["y"],
            size_patch=s.get("size_patch", 32), category="sphegnes"
        ))

    # Plot et sauvegarde
    plt.figure()
    lichen_set.plot_samples_as_list()
    plt.suptitle(f"Lichen : {len(lichen_samples)}")
    plt.savefig("data/samples/lichen_sphegnes_balanced/lichen_balanced.png")
    plt.close()

    plt.figure()
    sphegnes_set.plot_samples_as_list()
    plt.suptitle(f"Sphegnes : {len(sphegnes_samples)}")
    plt.savefig("data/samples/lichen_sphegnes_balanced/sphegnes_balanced.png")
    plt.close()

    # Sauvegarde du nombre de chaque catégorie
    with open("data/samples/lichen_sphegnes_balanced/pop_counts.txt", "w") as f:
        f.write(f"Lichen: {len(lichen_samples)}\n")
        f.write(f"Sphegnes: {len(sphegnes_samples)}\n")

# # Utilisation :
# if __name__ == "__main__":
#     update_rgbz_variance_in_json("data/samples/lichen_sphegnes_selection/lichen_sphegnes_selection.json")