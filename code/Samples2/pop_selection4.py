from samples_set2 import SamplesSet2
import numpy as np
import matplotlib.pyplot as plt
import os
from osgeo import gdal
from rasters_manager import RastersManager

def plot_rgb(xmin, ymin, xmax, ymax, ds_path, path):
    """Plot RGB image with the given boundaries and save it to the path."""
    ds = gdal.Open(ds_path)

    # Read the three bands - flip indices to match expected orientation
    r = ds.GetRasterBand(1).ReadAsArray()[xmin:xmax, ymin:ymax]
    g = ds.GetRasterBand(2).ReadAsArray()[xmin:xmax, ymin:ymax]
    b = ds.GetRasterBand(3).ReadAsArray()[xmin:xmax, ymin:ymax]

    # Normalize values for proper display
    r_norm = np.clip(r / 255.0, 0, 1)
    g_norm = np.clip(g / 255.0, 0, 1)
    b_norm = np.clip(b / 255.0, 0, 1)

    rgb = np.dstack((r_norm, g_norm, b_norm))
    plt.figure(figsize=(samples_plot_nb, samples_plot_nb))
    plt.imshow(rgb)
    plt.title("Fenêtre samples")
    plt.savefig(os.path.join(path, "fenetre_selection.png"))
    plt.close()

def pop_selection(x_start, y_start, n_samples_x, n_samples_y, size_patch, file_path, 
                 ds_path=None, dz_path=None, dt_path=None, default=0, distance_large=3, class_dict=None, samples_plot_nb = 10):
    """
    Interactive tool to select samples for training and testing models, using Sample2 and SamplesSet2.
    
    Args:
        x_start: Starting x coordinate (pixel)
        y_start: Starting y coordinate (pixel)
        n_samples_x: Number of samples in x direction
        n_samples_y: Number of samples in y direction
        size_patch: Size of each patch in pixels
        file_path: Directory where to save results
        ds_path: Path to RGB image
        dz_path: Path to DSM image
        dt_path: Path to thermal image
        default: Default selection state (0=not selected, 1=selected)
        distance_large: Distance for large neighborhood calculation
        class_dict: Dictionary mapping shortcut keys to class names
    """
    print("Sélection de la population à prendre")
    x_start = int(np.round(x_start/size_patch))*size_patch
    y_start = int(np.round(y_start/size_patch))*size_patch
    x_max = x_start + n_samples_x * size_patch
    y_max = y_start + n_samples_y * size_patch
    print(f"Grille globale : {x_start}, {y_start} à {x_max}, {y_max}")
    os.makedirs(file_path, exist_ok=True)
    
    if ds_path:
        plot_rgb(x_start, y_start, x_max, y_max, ds_path, file_path)

    # Initialize RastersManager with paths
    rasters_mgr = RastersManager()
    rasters_mgr.set_paths(ds_path, dz_path, dt_path)
    
    # Création du set global
    row_start = x_start
    column_start = y_start
    all_samples_set = SamplesSet2(
        ds_path=ds_path,
        dz_path=dz_path,
        dt_path=dt_path,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y
    )
    all_samples_set.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    samples_matrix = all_samples_set.get_samples_matrix()  # [i_y][i_x]

     # Demande la classe à sélectionner
    if class_dict is None: 
        class_dict = {"l": "lichen", "s": "sphaignes", "c": "crevasse", "w": "lac", "f": "foret", "q": "flaque"}

    while True:
        class_key = input(f"Classe à sélectionner : {class_dict.keys()}").strip().lower()
        if class_key in class_dict:
            break
        print("Classe invalide.")
    class_name = class_dict[class_key]
    print(f"Classe sélectionnée : {class_name}")

    # Initialisation de l'état de sélection avec default=0 (non sélectionné)
    selection_state = {(i_x, i_y): 0 for i_x in range(n_samples_x) for i_y in range(n_samples_y)}

    def plot_block(block_x, block_y):
        plt.close('all')
        fig, axes = plt.subplots(samples_plot_nb, samples_plot_nb, figsize=(20, 20))
        axes = np.array(axes).reshape(samples_plot_nb, samples_plot_nb)

        def on_key_press(event):
            # Sélection/désélection tout le bloc
            if event.key == '1':
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            selection_state[(i_x, i_y)] = 1
                update_plot_titles(fig)
            elif event.key == '0':
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            selection_state[(i_x, i_y)] = 0
                update_plot_titles(fig)
            # Flèche haut : toggle moitié supérieure
            elif event.key == 'left':
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb // 2):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = 0 if selection_state.get(key, 0) == 1 else 1
                update_plot_titles(fig)
            # Flèche bas : toggle moitié inférieure
            elif event.key == 'right':
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb // 2, samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = 0 if selection_state.get(key, 0) == 1 else 1
                update_plot_titles(fig)
            # Flèche gauche : toggle moitié gauche
            elif event.key == 'up':
                for dx in range(samples_plot_nb // 2):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = 0 if selection_state.get(key, 0) == 1 else 1
                update_plot_titles(fig)
            # Flèche droite : toggle moitié droite
            elif event.key == 'down':
                for dx in range(samples_plot_nb // 2, samples_plot_nb):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = 0 if selection_state.get(key, 0) == 1 else 1
                update_plot_titles(fig)
            # Entrée : ferme la fenêtre
            elif event.key == 'enter':
                plt.close(fig)

        for dx in range(samples_plot_nb):
            for dy in range(samples_plot_nb):
                i_x = block_x * samples_plot_nb + dx
                i_y = block_y * samples_plot_nb + dy
                if i_x < n_samples_x and i_y < n_samples_y:  # Check if within bounds
                    ax = axes[dx, dy]
                    sample = samples_matrix[i_y][i_x]
                    r, g, b, _, _ = sample.get_RGBZT()  # Note the change from RGBZ to RGBZT
                    if r is not None and g is not None and b is not None:
                        rgb = np.dstack((r, g, b)).astype(np.uint8)
                        ax.imshow(rgb)
                    ax.img_idx = (i_x, i_y)
                    ax.axis("off")
                    state = selection_state.get((i_x, i_y), 0)
                    if state == 1:
                        ax.set_title("✔")
                    else:
                        ax.set_title("")
                else:
                    axes[dx, dy].axis("off")  # Hide axes for out-of-bounds cells
        fig.suptitle(f"Bloc ({block_x},{block_y}) - Sélection\nAppuyer sur '1' pour tout sélectionner, '0' pour tout désélectionner")
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        plt.show()

    def on_click(event):
        ax = event.inaxes
        if ax is not None:
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                state = selection_state.get(idx, 0)
                state = (state + 1) % 2
                selection_state[idx] = state
                if state == 1:
                    ax.set_title("✔")
                else:
                    ax.set_title("")
                plt.draw()

   

    def update_plot_titles(fig):
        for ax in fig.get_axes():
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                state = selection_state.get(idx, 0)
                if state == 1:
                    ax.set_title("✔")
                else:
                    ax.set_title("")
        fig.canvas.draw_idle()

    # Parcours des blocs samples_plot_nbxsamples_plot_nb dans la matrice globale
    n_blocks_x = (n_samples_x + samples_plot_nb - 1) // samples_plot_nb  # Ceiling division
    n_blocks_y = (n_samples_y + samples_plot_nb - 1) // samples_plot_nb  # Ceiling division
    for block_y in range(n_blocks_y):
        for block_x in range(n_blocks_x):
            plot_block(block_x, block_y)
            print(f"Bloc ({block_x},{block_y}) affiché. Ferme la fenêtre pour passer au suivant.")
            print("Appuyer sur '1' pour tout sélectionner ou '0' pour tout désélectionner")

    # Création des deux populations
    sample_set = SamplesSet2(ds_path=ds_path, dz_path=dz_path, dt_path=dt_path)
    for (i_x, i_y), state in selection_state.items():
        if state == 1 and i_y < n_samples_y and i_x < n_samples_x:
            sample = samples_matrix[i_y][i_x]
            sample.category = class_name
            sample_set.add_Sample(sample)  # Note the change from add_sample to add_Sample

    # Calcul des grandeurs
    if len(sample_set.samples) > 0:
        sample_set.fill_neighbors_all(distance_large=distance_large)
    
    # Incrémentation du nom de fichier
    import glob
    idx = 1
    while os.path.exists(os.path.join(file_path, f"{class_name}_{idx}.json")):
        idx += 1
    json_path = os.path.join(file_path, f"{class_name}_{idx}.json")
    png_path = os.path.join(file_path, f"{class_name}_{idx}.png")

    # Affichage et sauvegarde
    if len(sample_set.samples) > 0:
        plt.figure(figsize=(20, 20))
        sample_set.plot_samples_as_list()
        plt.suptitle(f"{class_name} : {len(sample_set.samples)} samples")
        plt.savefig(png_path)
        plt.close()

    # Sauvegarde du set global avec catégories
    all_selected_set = SamplesSet2(ds_path=ds_path, dz_path=dz_path, dt_path=dt_path)
    for (i_x, i_y), state in selection_state.items():
        if state in [1, 2] and i_y < n_samples_y and i_x < n_samples_x:
            sample = samples_matrix[i_y][i_x]
            all_selected_set.add_Sample(sample)  # Note the change from add_sample to add_Sample
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

    # Renommage de la fenêtre de sélection
    import shutil
    src_img = os.path.join(file_path, "fenetre_selection.png")
    dst_img = png_path
    if os.path.exists(src_img) and src_img != dst_img:
        shutil.copy(src_img, dst_img)
        print(f"fenetre_selection.png copié vers {os.path.basename(dst_img)}")
    
    # Free raster resources
    all_samples_set.clear_rasters()
    sample_set.clear_rasters()
    all_selected_set.clear_rasters()
    rasters_mgr.clear_rasters()


if __name__ == "__main__":
    size_patch = 64
    distance_large = 3

    n_samples_x = 38
    n_samples_y = 76
    samples_plot_nb = 16
    default = None
        
    x_start = 2495
    y_start = 0

    # Chemins des rasters
    nb = "10_05"
    ds_path = f"drone_treated/WAP32_tiles/rgb/WAP32_full_transparent_mosaic_group1_{nb}.tif"
    dz_path = f"drone_treated/WAP32_tiles/dsm/WAP32_full_dsm_{nb}.tif"
    dt_path = None
    
    save_dir = "data/samples/selection11/"

    class_dict = {
        "pp": "peat_plateau",
        "ld" : "large_depression",
        "fo": "forest",
        "la": "lake"
    }
    
    
    
    pop_selection(
        x_start=x_start, 
        y_start=y_start, 
        n_samples_x=n_samples_x, 
        n_samples_y=n_samples_y, 
        size_patch=size_patch, 
        file_path=save_dir,
        ds_path=ds_path,
        dz_path=dz_path,
        dt_path=dt_path,
        default=default,
        distance_large=distance_large,
        class_dict=class_dict,
        samples_plot_nb = samples_plot_nb
    )
