from samples_set import SamplesSet
import numpy as np
import matplotlib.pyplot as plt
import os
from osgeo import gdal, ogr

def plot_rgb(xmin, ymin, xmax, ymax, path):
    
    ds = gdal.Open(r'data/twin_lake_mosaïc.tif')
    
    # Read the three bands

    r = ds.GetRasterBand(1).ReadAsArray()[xmin:xmax, ymin:ymax] 
    g = ds.GetRasterBand(2).ReadAsArray()[xmin:xmax, ymin:ymax]
    b = ds.GetRasterBand(3).ReadAsArray()[xmin:xmax, ymin:ymax]

    rgb = np.dstack((r, g, b))
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title("Fênetre samples")
    plt.savefig(path + "fenetre_selection.png")
    plt.show()



def pop_selection(x_start, y_start, n_samples_x, n_samples_y, size_patch, file_path, default=0):
    print("Sélection de la population à prendre")
    x_start = int(np.round(x_start/32))*32
    y_start = int(np.round(y_start/32))*32
    x_max = x_start + n_samples_x * size_patch
    y_max = y_start + n_samples_y * size_patch
    print(f"Grille globale : {x_start}, {y_start} à {x_max}, {y_max}")
    os.makedirs(file_path, exist_ok=True)
    plot_rgb(x_start, y_start, x_max, y_max, file_path)
    

    # Création du set global
    row_start = x_start
    column_start = y_start
    all_samples_set = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y)
    all_samples_set.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    samples_matrix = all_samples_set.get_samples_matrix()  # [i_y][i_x]
    #n_total = n_samples_x * n_samples_y

   # Initialisation de l'état de sélection selon le paramètre default
    selection_state = {(i_x, i_y): default for i_x in range(n_samples_x) for i_y in range(n_samples_y)}

    # Demande la classe à sélectionner
    class_dict = {"l": "lichen", "s": "sphegnes", "c": "crevasse", "w": "lac", "f": "foret", "q" : "flaque"}
    while True:
        class_key = input("Classe à sélectionner (l, s, c, w, f, q) : ").strip().lower()
        if class_key in class_dict:
            break
        print("Classe invalide.")
    class_name = class_dict[class_key]
    print(f"Classe sélectionnée : {class_name}")

    def plot_block(block_x, block_y):
        plt.close('all')
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        axes = np.array(axes).reshape(10, 10)
        for dy in range(10):
            for dx in range(10):
                i_x = block_x * 10 + dx
                i_y = block_y * 10 + dy
                ax = axes[dx, dy]
                sample = samples_matrix[i_y][i_x]
                r, g, b, _ = sample.get_RGBZ()
                rgb = np.dstack((r, g, b)).astype(np.uint8)
                ax.imshow(rgb)
                ax.img_idx = (i_x, i_y)
                ax.axis("off")
                state = selection_state.get((i_x, i_y), default)
                if state == 1:
                    ax.set_title("✔")
                else:
                    ax.set_title("")
        fig.suptitle(f"Bloc ({block_x},{block_y}) - Sélection")
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    def on_click(event):
        ax = event.inaxes
        if ax is not None:
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                state = selection_state.get(idx, default)
                state = (state + 1) % 2
                selection_state[idx] = state
                if state == 1:
                    ax.set_title("✔")
                else:
                    ax.set_title("")
                plt.draw()

    # Parcours des blocs 10x10 dans la matrice globale
    n_blocks_x = n_samples_x // 10
    n_blocks_y = n_samples_y // 10
    for block_y in range(n_blocks_y):
        for block_x in range(n_blocks_x):
            plot_block(block_x, block_y)
            print(f"Bloc ({block_x},{block_y}) affiché. Ferme la fenêtre pour passer au suivant.")

    # Création des deux populations
    sample_set = SamplesSet()
    for (i_x, i_y), state in selection_state.items():
        sample = samples_matrix[i_y][i_x]
        if state == 1:
            sample.category = class_name
            sample_set.add_sample(sample)

    # Calcul des grandeurs
    sample_set.fill_neighbors_all(depth_neighbors=1)
    
    # Incrémentation du nom de fichier
    import glob
    idx = 1
    while os.path.exists(os.path.join(file_path, f"{class_name}_{idx}.json")):
        idx += 1
    json_path = os.path.join(file_path, f"{class_name}_{idx}.json")
    png_path = os.path.join(file_path, f"{class_name}_{idx}.png")

    # Affichage et sauvegarde
    if len(sample_set.samples) > 0:
        plt.figure()
        sample_set.plot_samples_as_list()
        plt.suptitle(f"{class_name} : {len(sample_set.samples)} samples")
        plt.savefig(png_path)
        #plt.show()
        #plt.close()

    # Sauvegarde du set global avec catégories
    all_selected_set = SamplesSet()
    for (i_x, i_y), state in selection_state.items():
        sample = samples_matrix[i_y][i_x]
        if state in [1, 2]:
            all_selected_set.add_sample(sample)
    all_selected_set.save_samples_to_json(filename=json_path)

     # Sauvegarde/MAJ du fichier texte des comptes
    txt_path = os.path.join(file_path, "pop_counts.txt")
    counts = {}
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    counts[k.strip()] = int(v.strip())
    counts[class_name] = counts.get(class_name, 0) + len(sample_set.samples)
    with open(txt_path, "w") as f:
        for k, v in counts.items():
            f.write(f"{k}: {v}\n")
    print(f"{len(sample_set.samples)} samples '{class_name}' sauvegardés dans {json_path}")


if __name__ == "__main__":
    # Paramètres de la grille globale
    x_start = 15000
    y_start = 8500
    size_patch = 32

    n_samples_x = 20
    n_samples_y = 20
    file_path = "data/samples/selection3/"
    
    pop_selection(x_start, y_start, n_samples_x, n_samples_y, size_patch, file_path, default=0)