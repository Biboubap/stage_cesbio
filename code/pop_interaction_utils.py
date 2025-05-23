import json
import glob
import os
import random
from collections import Counter
from sample import Sample
from samples_set import SamplesSet

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
    """Fusionne toutes les populations JSON d'un dossier et retourne un SamplesSet."""
    json_files = glob.glob(os.path.join(directory, "crevasse_*.json"))
    print(f"Fichiers JSON trouvés : {json_files}")
    all_samples = []
    for path in json_files:
        data = load_population(path)
        for s in data.get("samples", []):
            sample = Sample(
                i_x=s.get("i_x"),
                i_y=s.get("i_y"),
                x=s.get("x"),
                y=s.get("y"),
                size_patch=s.get("size_patch"),
                category=s.get("category", None)
            )
            # Remplis ici d'autres attributs si besoin
            all_samples.append(sample)
    # Déduire n_samples_x et n_samples_y si possible, sinon None
    n_samples_x = None
    n_samples_y = None
    if all_samples:
        i_xs = [s.i_x for s in all_samples if s.i_x is not None]
        i_ys = [s.i_y for s in all_samples if s.i_y is not None]
        if i_xs and i_ys:
            n_samples_x = max(i_xs) + 1
            n_samples_y = max(i_ys) + 1
    samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    for s in all_samples:
        samples_set.add_sample(s)
    return samples_set


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
    # Supporte à la fois dicts et objets Sample
    def get_cat(s):
        if isinstance(s, dict):
            return s.get("category")
        else:
            return getattr(s, "category", None)
    counts = Counter(get_cat(s) for s in samples)
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

def split_sampleset(samples_set):
    """
    Découpe chaque sample 32x32 en 4 samples 16x16, complète leurs données,
    et retourne un nouveau SamplesSet.
    """
    new_samples = []
    for s in samples_set.samples.values():
        x0, y0 = s.x, s.y
        for dx in [0, 16]:
            for dy in [0, 16]:
                # Nouveaux indices i_x, i_y (optionnel, ici on les recalcule à la volée)
                new_sample = Sample(
                    i_x=None,  # ou laisse à None si tu ne veux pas d'indice
                    i_y=None,
                    x=x0 + dx,
                    y=y0 + dy,
                    size_patch=16,
                    category=getattr(s, "category", None)
                )
                new_samples.append(new_sample)
    # Création du nouveau SamplesSet (sans grille imposée)
    new_samples_set = SamplesSet()
    for s in new_samples:
        new_samples_set.add_sample(s)
    return new_samples_set

# Exemple d'utilisation :
if __name__ == "__main__":
    # Fusionne deux populations, renomme, filtre, limite le nombre de lichen, sauvegarde
    #pop1 = load_population("data/samples/selection4/pop3_noforestlake.json")
    crevasses = merge_populations_from_dir("data/samples/selection3")
    print("Nombre d'échantillons avant : ", len(crevasses.samples))
    crevasses_split=split_sampleset(crevasses)
    print(crevasses_split)
    crevasses_split.fill_neighbors_all()
    #all_samples = pop1["samples"] + pop2["samples"]

    # Renomme "sphegnes" en "sphaignes"
    #all_samples = rename_category(all_samples, "sphegnes", "sphaignes")

    # Ne garde que sphaignes, lichen, crevasse
    #keep_classes = {"sphaignes", "lichen", "crevasse"}
    #all_samples = filter_categories(all_samples, keep_classes)

    # Limite à 1600 lichen
    #all_samples = random_sample_category(pop2["samples"],"chicoutai", 834)

    # Compte les classes
    print("Comptage final :")
    count_categories(crevasses_split.samples)

    # # Sauvegarde
    # merged_data = {
    #     "n_samples_x": None,
    #     "n_samples_y": None,
    #     "samples": all_samples
    # }
    crevasses_split.save_samples_to_json("data/samples/selection7/crevasses16.json")