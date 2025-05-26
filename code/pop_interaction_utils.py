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
def merge_populations_from_dir(directory, mosaic_to_reshaped=False):
    """Fusionne toutes les populations JSON d'un dossier et retourne un SamplesSet."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Fichiers JSON trouvés : {json_files}")
    samples_set = SamplesSet(n_samples_x=None, n_samples_y=None)
    for path in json_files:
        data = load_population(path)
        for s in data.get("samples", []):
            x=s.get("x")
            y=s.get("y")
            if mosaic_to_reshaped:
                        x -= 2454
                        y -= 4963
            sample = Sample(
                i_x=None,
                i_y=None,
                x = x,
                y = y,
                size_patch=s.get("size_patch"),
                category=s.get("category", None)
            )
            # Remplis ici d'autres attributs si besoin
            samples_set.add_sample(sample)
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

def count_categories(sample_set):
    samples = sample_set.samples.values() 
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

def random_sample_category(sample_set, category, n_max):
    samples = list(sample_set.samples.values())
    """Garde au maximum n_max échantillons d'une catégorie, choisis au hasard."""
    cat_samples = [s for s in samples if getattr(s, "category", None) == category]
    other_samples = [s for s in samples if getattr(s, "category", None) != category]
    if len(cat_samples) > n_max:
        cat_samples = random.sample(cat_samples, n_max)
    # Création d'un nouveau SamplesSet pour rester cohérent avec le reste du code
    new_set = SamplesSet()
    for s in cat_samples + other_samples:
        new_set.add_sample(s)
    return new_set

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
    new_samples_set = SamplesSet()
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
                    size_patch=getattr(s, "size_patch", None),
                    category=getattr(s, "category", None)
                )
                new_samples_set.add_sample(new_sample)
    # Création du nouveau SamplesSet (sans grille imposée)

    return new_samples_set

# Exemple d'utilisation :
if __name__ == "__main__":
   
    # pop6 = SamplesSet.load_samples_from_json("data/samples/selection6/merged_pop6.json")
    # pop7 = merge_populations_from_dir("data/samples/selection7")
    # pop_merged=SamplesSet.concatenate_set(pop6, pop7)
    
    
    
    # #all_samples = pop1["samples"] + pop2["samples"]

    # # Renomme "sphegnes" en "sphaignes"
    # #all_samples = rename_category(all_samples, "sphegnes", "sphaignes")

    # # Ne garde que sphaignes, lichen, crevasse
    # #keep_classes = {"sphaignes", "lichen", "crevasse"}
    # #all_samples = filter_categories(all_samples, keep_classes)

    # # Limite à 1600 lichen
    # for category in ["chicoutai", "crevasse", "sphaignes"]:
    #     pop_merged = random_sample_category(pop_merged,category, 850)
    
    # pop_merged.fill_neighbors_all()
    # print("Comptage final :")
    # count_categories(pop_merged)
    # pop_merged.save_samples_to_json("data/samples/selection7/pop7_merged.json")
    # # # Sauvegarde
    # merged_data = {
    #     "n_samples_x": None,
    #     "n_samples_y": None,
    #     "samples": all_samples
    # }

    #Decoupage des samples 32x32 en 16x16
    crevasses = merge_populations_from_dir("data/samples/selection3/crevasses", mosaic_to_reshaped=True)
    crevasses_split=split_sampleset(crevasses)
    print(crevasses_split)
    crevasses_split.fill_neighbors_all()
 
    print("Comptage final :")
    count_categories(crevasses_split)

    crevasses_split.save_samples_to_json("data/samples/selection7/crevasses.json")