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

def merge_and_limit_populations(input_dir, output_json, max_per_cat=800):
    # Récupère tous les fichiers json de la sélection
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    # Dictionnaire {catégorie: [samples]}
    cat_samples = {}

    # Parcours des fichiers et collecte par catégorie
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        for s in data["samples"]:
            cat = s.get("category")
            if cat is None:
                continue
            cat_samples.setdefault(cat, []).append(s)

    # Limite à max(max_per_cat, taille de la pop) pour chaque catégorie
    merged_samples = []
    for cat, samples in cat_samples.items():
        n = min(max_per_cat, len(samples))
        print(f"Category '{cat}': {len(samples)} samples, limiting to {n}")
        if len(samples) > n:
            import random
            samples = random.sample(samples, n)
        merged_samples.extend(samples)

    # Création du dictionnaire final
    merged_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": merged_samples
    }

    # Sauvegarde
    with open(output_json, "w") as f:
        json.dump(merged_data, f, indent=2)
    print(f"Merged populations saved to {output_json}")
    print(f"Total samples: {len(merged_samples)}")



def recalcule_neighbors_and_save(input_json, output_json, depth_neighbors=3):
    """
    Charge tous les samples depuis input_json, recalcule les données de voisinage
    avec la distance demandée, et sauvegarde dans output_json.
    """
    print(f"Chargement des samples depuis {input_json} ...")
    samples_set = SamplesSet.load_samples_from_json(input_json)
    print(f"Remplissage des voisins avec depth_neighbors={depth_neighbors} ...")
    samples_set.fill_neighbors_all(depth_neighbors=depth_neighbors)
    print(f"Sauvegarde dans {output_json} ...")
    samples_set.save_samples_to_json(output_json)
    print("Terminé.")

def rename_and_merge_samples():
    # Chemins des fichiers
    base_dir = "data/samples/selection3-2/"
    main_json = base_dir + "full_samples_nei3.json"
    crevasse_json = base_dir + "crevasse_1.json"
    sphaignes_json = base_dir + "sphaignes_1.json"
    output_json = base_dir + "merged_full_nei3.json"

    # Charge le set principal
    main_set = SamplesSet.load_samples_from_json(main_json)
    # Renomme les catégories "sphegnes" en "sphaignes"
    for s in main_set.samples.values():
        if getattr(s, "category", None) == "sphegnes":
            s.category = "sphaignes"

    # Charge les autres sets
    crevasse_set = SamplesSet.load_samples_from_json(crevasse_json)
    sphaignes_set = SamplesSet.load_samples_from_json(sphaignes_json)

    # Fusionne les sets
    merged_set = SamplesSet.concatenate_set(main_set, crevasse_set)
    merged_set = SamplesSet.concatenate_set(merged_set, sphaignes_set)
    
    # Sauvegarde
    merged_set.save_samples_to_json(output_json)
    print(f"Fichier fusionné sauvegardé dans {output_json}")

def remove_forest_and_lake():
    all_samples = []
    
    main_json = "data/samples/selection3/pop3_merged.json"
    output_json = "data/samples/selection4/pop3_noforestlake.json"

    # Charge le set principal
    
    with open(main_json, "r") as f:
        data = json.load(f)
    for s in data.get("samples", []):
            cat = s.get("category", "")
            if cat not in ["foret", "lac"]:
                all_samples.append(s)

    filtered_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": all_samples
    }

    with open(output_json, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"{len(all_samples)} samples sauvegardés dans {output_json}")

if __name__ == "__main__":
    remove_forest_and_lake()

# if __name__ == "__main__":
    # recalcule_neighbors_and_save(
    #     "data/samples/selection3/pop3_merged.json",
    #     "data/samples/selection3-2/full_samples_nei3.json",
    #     depth_neighbors=3
    # )
    # rename_and_merge_samples()
    
    # base_dir = "data/samples/selection3-2/"
    # output_json = base_dir + "merged_full_nei3.json"
    # merged_set = SamplesSet.load_samples_from_json(output_json)
    # nb_sphaignes = len([s for s in merged_set.samples.values() if getattr(s, "category", None) == "sphaignes"])
    # print(f"Nombre de sphaignes : {nb_sphaignes}")


# if __name__ == "__main__":
#     merge_and_limit_populations(
#         input_dir="data/samples/selection3",
#         output_json="data/samples/selection3/pop3_merged.json",
#         max_per_cat=800
#     )

# # Utilisation :
# if __name__ == "__main__":
#     update_rgbz_variance_in_json("data/samples/lichen_sphegnes_selection/lichen_sphegnes_selection.json")