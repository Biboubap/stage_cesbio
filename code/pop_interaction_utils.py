import json
import glob
import os
import random
from collections import Counter

def load_population(json_path):
    """Charge une population depuis un fichier JSON."""
    with open(json_path, "r") as f:
        return json.load(f)

def save_population(data, json_path):
    """Sauvegarde une population dans un fichier JSON."""
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def merge_populations(json_paths):
    """Fusionne plusieurs populations (listes de fichiers JSON)."""
    all_samples = []
    for path in json_paths:
        data = load_population(path)
        all_samples.extend(data.get("samples", []))
    return {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": all_samples
    }

def merge_populations_from_dir(directory):
    """Fusionne toutes les populations JSON d'un dossier."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    return merge_populations(json_files)

def rename_category(samples, old_name, new_name):
    """Renomme une catégorie dans une liste de samples."""
    for s in samples:
        if s.get("category") == old_name:
            s["category"] = new_name
    return samples

def filter_categories(samples, keep_categories):
    """Ne garde que certaines catégories dans une liste de samples."""
    return [s for s in samples if s.get("category") in keep_categories]

def count_categories(samples):
    """Compte le nombre d'échantillons par catégorie."""
    counts = Counter(s.get("category") for s in samples)
    for cat, n in counts.items():
        print(f"{cat}: {n}")
    return counts

def random_sample_category(samples, category, n_max):
    """Garde au maximum n_max échantillons d'une catégorie, choisis au hasard."""
    cat_samples = [s for s in samples if s.get("category") == category]
    other_samples = [s for s in samples if s.get("category") != category]
    if len(cat_samples) > n_max:
        cat_samples = random.sample(cat_samples, n_max)
    return cat_samples + other_samples

def balance_two_categories(samples, cat1, cat2):
    """Équilibre deux catégories en gardant le même nombre d'échantillons pour chaque."""
    samples1 = [s for s in samples if s.get("category") == cat1]
    samples2 = [s for s in samples if s.get("category") == cat2]
    n = min(len(samples1), len(samples2))
    samples1 = random.sample(samples1, n)
    samples2 = random.sample(samples2, n)
    return samples1 + samples2

def remove_categories(samples, remove_cats):
    """Supprime certaines catégories de la liste de samples."""
    return [s for s in samples if s.get("category") not in remove_cats]

# # Exemple d'utilisation :
# if __name__ == "__main__":
#     # Fusionne deux populations, renomme, filtre, limite le nombre de lichen, sauvegarde
#     pop1 = load_population("data/samples/selection4/pop3_noforestlake.json")
#     pop2 = merge_populations_from_dir("data/samples/selection5")
#     all_samples = pop1["samples"] + pop2["samples"]

#     # Renomme "sphegnes" en "sphaignes"
#     all_samples = rename_category(all_samples, "sphegnes", "sphaignes")

#     # Ne garde que sphaignes, lichen, crevasse
#     keep_classes = {"sphaignes", "lichen", "crevasse"}
#     all_samples = filter_categories(all_samples, keep_classes)

#     # Limite à 1600 lichen
#     all_samples = random_sample_category(all_samples, "lichen", 1600)

#     # Compte les classes
#     print("Comptage final :")
#     count_categories(all_samples)

#     # Sauvegarde
#     merged_data = {
#         "n_samples_x": None,
#         "n_samples_y": None,
#         "samples": all_samples
#     }
#     save_population(merged_data, "data/samples/selection5/merged_utilitaires.json")