from samples_set import SamplesSet
import glob
import os

if __name__ == "__main__":
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