import json
import glob
import os
import random
from collections import Counter
from sample2 import Sample2
from samples_set2 import SamplesSet2
from rasters_manager import RastersManager

def load_population(json_path):
    """
    Charge une population depuis un fichier JSON.
    Retourne le contenu JSON brut.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def save_population(data, json_path):
    """
    Sauvegarde une population dans un fichier JSON.
    """
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def merge_populations_to_json(json_paths, output_path):
    """
    Fusionne plusieurs populations de fichiers JSON et sauvegarde le résultat.
    
    Args:
        json_paths: Liste de chemins vers des fichiers JSON contenant des populations
        output_path: Chemin où sauvegarder le résultat fusionné
    """
    all_samples = []
    for path in json_paths:
        data = load_population(path)
        all_samples.extend(data.get("samples", []))
    
    merged_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": all_samples
    }
    
    save_population(merged_data, output_path)
    return merged_data

def merge_populations_to_samples_set(json_paths, ds_path=None, dz_path=None, dt_path=None):
    """
    Fusionne plusieurs populations de fichiers JSON en un SamplesSet2.
    
    Args:
        json_paths: Liste de chemins vers des fichiers JSON contenant des populations
        ds_path, dz_path, dt_path: Chemins des rasters à utiliser
        
    Returns:
        SamplesSet2: Un ensemble de samples fusionnés
    """
    # Initialize raster manager once
    rasters = RastersManager()
    rasters.set_paths(ds_path, dz_path, dt_path)
    
    # Create a new SamplesSet2
    samples_set = SamplesSet2(ds_path, dz_path, dt_path, n_samples_x=None, n_samples_y=None)
    offset = 0
    for path in json_paths:
        data = load_population(path)
        print(f"Chargement de {len(data.get('samples', []))} samples depuis {path}")
        for s in data.get("samples", []):
            # Create Sample2 objects from json data
            sample = Sample2(
                i_x=s.get("i_x")+ offset,
                i_y=s.get("i_y")+ offset, 
                x=s.get("x")+ offset*s.get("size_patch"),
                y=s.get("y")+ offset*s.get("size_patch"),
                size_patch=s.get("size_patch"),
                category=s.get("category")
            )
            # Copy available feature values
            for attr in [
                "r_mean", "g_mean", "b_mean", "t_mean", "z_mean",
                "r_var", "g_var", "b_var", "t_var", "z_var",
                "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean",
                "z_moins_z_n",
                "r_large_mean", "g_large_mean", "b_large_mean", 
                "t_large_mean", "z_moins_z_large"
            ]:
                if attr in s:
                    setattr(sample, attr, s.get(attr))
            
            samples_set.add_Sample(sample)
        offset += 80
        
    print(f"Total samples fusionnés: {len(samples_set.samples)}")
    return samples_set

def merge_populations_from_dir(directory, ds_path=None, dz_path=None, dt_path=None, mosaic_to_reshaped=False):
    """
    Fusionne toutes les populations JSON d'un dossier et retourne un SamplesSet2.
    
    Args:
        directory: Dossier contenant les fichiers JSON
        ds_path, dz_path, dt_path: Chemins des rasters à utiliser
        mosaic_to_reshaped: Si True, ajuste les coordonnées pour le changement de système
        
    Returns:
        SamplesSet2: Un ensemble de samples fusionnés
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))
    print(f"Fichiers JSON trouvés : {json_files}")
    
    # Initialize raster manager
    rasters = RastersManager()
    rasters.set_paths(ds_path, dz_path, dt_path)
    
    # Create a new SamplesSet2
    samples_set = SamplesSet2(ds_path, dz_path, dt_path, n_samples_x=None, n_samples_y=None)
    
    for path in json_files:
        data = load_population(path)
        for s in data.get("samples", []):
            x = s.get("x")
            y = s.get("y")
            
            # Ajustement des coordonnées si nécessaire
            if mosaic_to_reshaped:
                x -= 2454
                y -= 4963
                
            # Create Sample2 objects from json data
            sample = Sample2(
                i_x=s.get("i_x"),
                i_y=s.get("i_y"), 
                x=x,
                y=y,
                size_patch=s.get("size_patch"),
                category=s.get("category")
            )
            
            # Copy available feature values
            for attr in [
                "r_mean", "g_mean", "b_mean", "t_mean", "z_mean",
                "r_var", "g_var", "b_var", "t_var", "z_var",
                "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean",
                "z_moins_z_n",
                "r_large_mean", "g_large_mean", "b_large_mean", 
                "t_large_mean", "z_moins_z_large"
            ]:
                if attr in s:
                    setattr(sample, attr, s.get(attr))
            
            samples_set.add_Sample(sample)
    
    return samples_set

def convert_legacy_samples(json_path, ds_path=None, dz_path=None, dt_path=None, distance_large=3):
    """
    Charge une population d'anciens Samples, les convertit en Sample2 et calcule les nouvelles features.
    
    Args:
        json_path: Chemin vers le fichier JSON contenant les anciens samples
        ds_path, dz_path, dt_path: Chemins des rasters à utiliser
        distance_large: Distance à utiliser pour les features "large"
        
    Returns:
        SamplesSet2: Un ensemble de samples avec toutes les nouvelles features calculées
    """
    # Initialize raster manager
    rasters = RastersManager()
    rasters.set_paths(ds_path, dz_path, dt_path)
    rasters.load_rasters()
    
    # Create SamplesSet2
    samples_set = SamplesSet2(ds_path, dz_path, dt_path, n_samples_x=None, n_samples_y=None)
    
    # Load the legacy samples
    with open(json_path, "r") as f:
        data = json.load(f)
    
    print(f"Chargement de {len(data.get('samples', []))} samples depuis {json_path}")
    
    # First pass: create basic samples from existing data
    for s in data.get("samples", []):
        sample = Sample2(
            i_x=s.get("i_x"),
            i_y=s.get("i_y"), 
            x=s.get("x"),
            y=s.get("y"),
            size_patch=s.get("size_patch"),
            category=s.get("category")
        )
        
        # Copy existing feature values
        for attr in [
            "r_mean", "g_mean", "b_mean", "t_mean", "z_mean",
            "r_var", "g_var", "b_var", "t_var", "z_var",
            "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean",
            "z_moins_z_n"
        ]:
            if attr in s:
                setattr(sample, attr, s.get(attr))
                
        # Add sample to set
        samples_set.add_Sample(sample)
    
    print("Calcul des nouvelles features de voisinage...")
    # Second pass: compute neighbor features 
    # This will recalculate neighborhood features and add the new 'large' features
    samples_set.fill_neighbors_all(distance_large=distance_large)
    
    print(f"Conversion terminée. Nouvelles features ajoutées.")
    return samples_set

def filter_samples_by_category(samples_set, categories):
    """
    Filtre un SamplesSet2 pour ne garder que les samples des catégories spécifiées.
    
    Args:
        samples_set: SamplesSet2 à filtrer
        categories: Liste des catégories à conserver
        
    Returns:
        SamplesSet2: Un nouveau SamplesSet2 avec uniquement les samples des catégories spécifiées
    """
    filtered_set = SamplesSet2(
        ds_path=samples_set.ds_path,
        dz_path=samples_set.dz_path,
        dt_path=samples_set.dt_path,
        n_samples_x=None,
        n_samples_y=None
    )
    
    for sample in samples_set.samples.values():
        if sample.category in categories:
            filtered_set.add_Sample(sample)
            
    return filtered_set

def balance_samples_by_category(samples_set, max_per_category=None):
    """
    Équilibre un SamplesSet2 pour avoir le même nombre d'échantillons par catégorie.
    
    Args:
        samples_set: SamplesSet2 à équilibrer
        max_per_category: Nombre maximum d'échantillons par catégorie (None = min count)
        
    Returns:
        SamplesSet2: Un nouveau SamplesSet2 équilibré
    """
    # Count samples per category
    category_counts = {}
    for sample in samples_set.samples.values():
        category = sample.category
        if category not in category_counts:
            category_counts[category] = []
        category_counts[category].append(sample)
    
    # Determine sample count per category
    if max_per_category is None:
        min_count = min([len(samples) for samples in category_counts.values()])
        max_per_category = min_count
    
    # Create balanced set
    balanced_set = SamplesSet2(
        ds_path=samples_set.ds_path,
        dz_path=samples_set.dz_path,
        dt_path=samples_set.dt_path,
        n_samples_x=None,
        n_samples_y=None
    )
    
    for category, samples in category_counts.items():
        # Sample randomly if we have more than needed
        selected_samples = samples
        if len(samples) > max_per_category:
            selected_samples = random.sample(samples, max_per_category)
        
        # Add samples to the new set
        for sample in selected_samples:
            balanced_set.add_Sample(sample)
    
    print(f"Équilibrage terminé: {len(balanced_set.samples)} samples au total")
    category_summary = Counter([s.category for s in balanced_set.samples.values()])
    print(f"Répartition par catégorie: {category_summary}")
    
    return balanced_set

# Exemple d'utilisation :
def merge_Wap_samples():
    
    import glob
    import os

    # Paths
    base_dir = "data/samples/selection11"
    out_dir = os.path.join(base_dir, "pop_merged")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "pop_merged.json")
    temp_json = os.path.join(out_dir, "all_samples_combined.json")

    # ds_path = "data/rgb_reshaped.tif"
    # dz_path = "data/dsm_reshaped.tif"
    # dt_path = "data/thermal_reshaped.tif"

    # Chercher tous les fichiers forest_*, large_depression_*, peat_plateau_*
    patterns = [
        os.path.join(base_dir, "forest_*.json"),
        os.path.join(base_dir, "large_depression_*.json"),
        os.path.join(base_dir, "peat_plateau_*.json"),
    ]
    json_files = []
    for pat in patterns:
        json_files.extend(glob.glob(pat))
    print(f"Fichiers trouvés ({len(json_files)}): {json_files}")

    # ÉTAPE 1: Fusionner tous les samples au niveau JSON (préserve tous les samples)
    print("Fusion des échantillons au format JSON...")
    merged_json = merge_populations_to_json(json_files, temp_json)
    
    # Compter les samples par catégorie avant équilibrage
    category_counts = Counter([s["category"] for s in merged_json["samples"]])
    print(f"Samples avant équilibrage: {len(merged_json['samples'])}")
    print(f"Répartition initiale: {category_counts}")
    
    # ÉTAPE 2: Équilibrer les catégories dans le JSON
    balanced_samples = {}
    for s in merged_json["samples"]:
        category = s["category"]
        if category not in balanced_samples:
            balanced_samples[category] = []
        balanced_samples[category].append(s)
    
    # Limiter chaque catégorie à 3300 samples
    all_balanced_samples = []
    for category, samples in balanced_samples.items():
        if len(samples) > 3300:
            selected = random.sample(samples, 3300)
        else:
            selected = samples
        all_balanced_samples.extend(selected)
    
    # ÉTAPE 3: Sauvegarder le résultat équilibré
    final_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": all_balanced_samples
    }
    save_population(final_data, out_json)
    
    # Statistiques finales
    final_counts = Counter([s["category"] for s in all_balanced_samples])
    print(f"Équilibrage terminé: {len(all_balanced_samples)} samples au total")
    print(f"Répartition par catégorie: {final_counts}")
    print(f"Population fusionnée et équilibrée sauvegardée dans {out_json}")

# def merge_selection12_samples():
#     """
#     Fusionne toutes les populations de selection12 avec des modifications spécifiques:
#     - Reclassifie les 75 derniers samples de watered_depression_3 en "green_depression"
#     - Retire les 35 premiers samples de sphaignes_2
#     - Retire les 57 derniers samples de green_depression_2
#     """
#     import os
#     import glob
    
#     # Paths
#     base_dir = "data/samples/selection12"
#     out_dir = "data/samples/pop_merged"
#     os.makedirs(out_dir, exist_ok=True)
#     out_json = os.path.join(out_dir, "pop_12.json")
    
#     # Find all JSON files in selection12
#     json_files = glob.glob(os.path.join(base_dir, "*.json"))
#     print(f"Fichiers trouvés ({len(json_files)}): {json_files}")
    
#     # Load and modify all samples
#     all_samples = []
    
#     for path in json_files:
#         data = load_population(path)
#         samples = data.get("samples", [])
#         filename = os.path.basename(path)
        
#         # Apply specific modifications
#         if filename == "watered_depression_3.json":
#             # Reclassify the last 75 samples as "green_depression"
#             if len(samples) >= 75:
#                 for i in range(len(samples) - 75, len(samples)):
#                     samples[i]["category"] = "green_depression"
#             print(f"Reclassifiés {min(75, len(samples))} derniers samples de {filename} en 'green_depression'")
                
#         elif filename == "sphaignes_2.json":
#             # Remove the first 35 samples
#             if len(samples) >= 35:
#                 samples = samples[35:]
#             else:
#                 samples = []
#             print(f"Retirés {min(35, len(data.get('samples', [])))} premiers samples de {filename}")
                
#         elif filename == "green_depression_2.json":
#             # Remove the last 57 samples
#             if len(samples) >= 57:
#                 samples = samples[:-57]
#             print(f"Retirés {min(57, len(data.get('samples', [])))} derniers samples de {filename}")
        
#         all_samples.extend(samples)
    
#     # Create the merged data
#     merged_data = {
#         "n_samples_x": None,
#         "n_samples_y": None,
#         "samples": all_samples
#     }
    
#     # Save the merged data
#     save_population(merged_data, out_json)
    
#     # Count samples by category
#     categories_count = Counter([s["category"] for s in all_samples])
#     print(f"\nPopulation fusionnée: {len(all_samples)} samples au total")
#     print(f"Répartition par catégorie:")
#     for category, count in sorted(categories_count.items(), key=lambda x: x[0]):
#         print(f"  - {category}: {count} samples")
#     print(f"Population sauvegardée dans {out_json}")
    
#     return merged_data

# def duplicate_and_balance_selection12():
#     """
#     Load pop12.json from selection12/pop_merged/, duplicate samples for chicoutai and green_depression,
#     balance all categories to 2000 samples each, and save back to the same directory.
#     """
#     import os
    
#     # Paths
#     input_path = "data/samples/selection12/pop_merged/pop_12.json"
#     output_path = "data/samples/selection12/pop_merged/pop_12_balanced.json"
    
#     print(f"Loading samples from {input_path}...")
#     data = load_population(input_path)
#     samples = data.get("samples", [])
    
#     # Count initial samples by category
#     category_counts = Counter([s["category"] for s in samples])
#     print(f"Initial samples: {len(samples)} total")
#     print(f"Initial distribution by category: {category_counts}")
    
#     # Organize samples by category
#     samples_by_category = {}
#     for s in samples:
#         category = s["category"]
#         if category not in samples_by_category:
#             samples_by_category[category] = []
#         samples_by_category[category].append(s)
    
#     # Duplicate samples for chicoutai and green_depression if needed
#     for category in ["chicoutai", "green_depression"]:
#         if category in samples_by_category:
#             original_samples = samples_by_category[category].copy()
#             while len(samples_by_category[category]) < 2000:
#                 # Select a random sample to duplicate
#                 sample_to_duplicate = random.choice(original_samples)
#                 duplicate = sample_to_duplicate.copy()  # Create a deep copy
                
#                 # Add small variation to x, y coordinates to avoid exact duplicates
#                 variation = random.randint(1, 5)
#                 duplicate["x"] += variation
#                 duplicate["y"] += variation
                
#                 samples_by_category[category].append(duplicate)
    
#     # Balance all categories to 2000 samples each
#     balanced_samples = []
#     for category, category_samples in samples_by_category.items():
#         if len(category_samples) > 2000:
#             # Randomly select 2000 samples
#             selected_samples = random.sample(category_samples, 2000)
#         else:
#             selected_samples = category_samples
        
#         balanced_samples.extend(selected_samples)
#         print(f"{category}: {len(selected_samples)} samples after balancing")
    
#     # Create final data structure
#     balanced_data = {
#         "n_samples_x": None,
#         "n_samples_y": None,
#         "samples": balanced_samples
#     }
    
#     # Save balanced data
#     save_population(balanced_data, output_path)
    
#     # Print final stats
#     final_counts = Counter([s["category"] for s in balanced_samples])
#     print(f"\nBalanced population: {len(balanced_samples)} samples total")
#     print(f"Final distribution by category: {final_counts}")
#     print(f"Balanced population saved to {output_path}")
    
#     return balanced_data

def merge_selection12_and_13(balance_max=None):
    """
    Merge pop_12.json from selection12/pop_merged with all JSON files from selection13
    and count samples by category.
    
    Args:
        balance_max: If provided, limits each category to this many samples (random selection)
    """
    import os
    import glob
    from collections import Counter
    
    # Paths
    selection12_path = "data/samples/selection12/pop_merged/pop_12.json"
    selection13_dir = "data/samples/selection13"
    output_dir = "data/samples/selection13/merged"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_12_13.json")
    
    # Load samples from selection12
    print(f"Loading samples from {selection12_path}...")
    data12 = load_population(selection12_path)
    samples12 = data12.get("samples", [])
    
    # Count samples by category in selection12
    categories12 = Counter([s["category"] for s in samples12])
    print(f"Selection12 samples: {len(samples12)} total")
    print(f"Selection12 distribution by category: {categories12}")
    
    # Find all JSON files in selection13
    json_files = glob.glob(os.path.join(selection13_dir, "*.json"))
    print(f"\nFound {len(json_files)} JSON files in selection13.")
    
    # Load samples from selection13
    all_samples13 = []
    for path in json_files:
        data = load_population(path)
        samples = data.get("samples", [])
        all_samples13.extend(samples)
        
        # Print details for each file
        filename = os.path.basename(path)
        file_categories = Counter([s["category"] for s in samples])
        print(f"{filename}: {len(samples)} samples - {dict(file_categories)}")
    
    # Count samples by category in selection13
    categories13 = Counter([s["category"] for s in all_samples13])
    print(f"\nSelection13 samples: {len(all_samples13)} total")
    print(f"Selection13 distribution by category: {categories13}")
    
    # Merge samples from both selections
    merged_samples = samples12 + all_samples13
    
    # Count merged samples by category
    merged_categories = Counter([s["category"] for s in merged_samples])
    print(f"\nMerged samples before balancing: {len(merged_samples)} total")
    print(f"Distribution by category before balancing:")
    for category, count in sorted(merged_categories.items()):
        print(f"  - {category}: {count} samples")
    
    # Balance categories if requested
    if balance_max is not None:
        # Group samples by category
        samples_by_category = {}
        for s in merged_samples:
            category = s["category"]
            if category not in samples_by_category:
                samples_by_category[category] = []
            samples_by_category[category].append(s)
        
        # Limit each category to balance_max samples
        balanced_samples = []
        for category, samples in samples_by_category.items():
            if len(samples) > balance_max:
                # Random selection to limit to balance_max
                selected = random.sample(samples, balance_max)
                print(f"Category '{category}': {len(samples)} → {len(selected)} samples (randomly selected)")
            else:
                selected = samples
                print(f"Category '{category}': {len(samples)} samples (unchanged)")
            balanced_samples.extend(selected)
        
        merged_samples = balanced_samples
        
        # Count after balancing
        balanced_categories = Counter([s["category"] for s in merged_samples])
        print(f"\nAfter balancing: {len(merged_samples)} total")
        print(f"Distribution by category after balancing:")
        for category, count in sorted(balanced_categories.items()):
            print(f"  - {category}: {count} samples")
    
    # Create merged data structure
    merged_data = {
        "n_samples_x": None,
        "n_samples_y": None,
        "samples": merged_samples
    }
    
    # Save merged data
    balance_suffix = f"_balanced_{balance_max}" if balance_max is not None else ""
    output_path_final = output_path.replace(".json", f"{balance_suffix}.json")
    save_population(merged_data, output_path_final)
    print(f"\nMerged samples saved to {output_path_final}")
    
    return merged_data

# Exemple d'utilisation :
if __name__ == "__main__":
    # Merge selection12 and selection13 samples, balanced to max 2500 samples per category
    merge_selection12_and_13(balance_max=2500)

