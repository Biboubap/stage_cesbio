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
    
    for path in json_paths:
        data = load_population(path)
        for s in data.get("samples", []):
            # Create Sample2 objects from json data
            sample = Sample2(
                i_x=s.get("i_x"),
                i_y=s.get("i_y"), 
                x=s.get("x"),
                y=s.get("y"),
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
if __name__ == "__main__":
    # Initialize paths
    ds_path = "data/rgb_reshaped.tif"
    dz_path = "data/dsm_reshaped.tif" 
    dt_path = "data/thermal_reshaped.tif"
    
    # Convert legacy samples
    samples_set = convert_legacy_samples(
        "data/samples/selection8/merged_pop/merged_pop8_2.json",
        ds_path, dz_path, dt_path
    )
    
    # Save the result with the new features
    samples_set.save_samples_to_json("data/samples/selection10/pop8_converted.json")
    
    print(f"Samples converted and saved. Total: {len(samples_set.samples)}")