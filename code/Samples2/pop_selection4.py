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
    plt.show()
    

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
        class_dict = {"l": "lichen", "s": "sphaignes", "c": "chicoutai", "w": "lac", "f": "foret", "q": "flaque"}
    
    # Keep track of current active category
    current_category_key = None
    early_exit = False  # Flag for early exit from selection
    
    while True:
        class_key = input(f"Classe initiale à sélectionner : {class_dict.keys()}").strip().lower()
        if class_key in class_dict:
            current_category_key = class_key
            break
        print("Classe invalide.")
    
    print(f"Classe sélectionnée : {class_dict[current_category_key]} (key: {current_category_key})")

    # Initialisation de l'état de sélection avec None (non sélectionné)
    # Now the state will store the category key instead of just 0 or 1
    selection_state = {(i_x, i_y): None for i_x in range(n_samples_x) for i_y in range(n_samples_y)}

    def plot_block(block_x, block_y):
        nonlocal current_category_key
        plt.close('all')
        fig, axes = plt.subplots(samples_plot_nb, samples_plot_nb, figsize=(20, 20))
        axes = np.array(axes).reshape(samples_plot_nb, samples_plot_nb)

        def on_key_press(event):
            nonlocal current_category_key, early_exit
            
            # Early exit with 'q' key
            if event.key == 'q':
                print("Early exit requested. Saving current selections...")
                early_exit = True
                plt.close(fig)
                return
            
            # Switch category when space is pressed
            if event.key == ' ':
                # Create a separate category selection window instead of using input()
                # to avoid event loop conflicts
                plt.figure(figsize=(4, 3))
                plt.axis('off')
                cat_buttons = {}
                
                plt.text(0.1, 0.9, "Select category:", fontsize=12)
                
                # Create "buttons" for each category
                y_pos = 0.8
                for i, (key, cat_name) in enumerate(class_dict.items()):
                    y_pos -= 0.1
                    plt.text(0.2, y_pos, f"{key}: {cat_name}", fontsize=10)
                    # Create invisible rectangle for click detection
                    rect = plt.Rectangle((0.1, y_pos-0.05), 0.8, 0.08, 
                                        alpha=0.2, facecolor='gray', edgecolor='black')
                    plt.gca().add_patch(rect)
                    cat_buttons[key] = rect
                
                plt.tight_layout()
                
                def on_cat_click(event):
                    nonlocal current_category_key
                    if event.inaxes:
                        for key, rect in cat_buttons.items():
                            contains, _ = rect.contains(event)
                            if contains:
                                current_category_key = key
                                print(f"Classe active changée pour : {class_dict[current_category_key]} (key: {current_category_key})")
                                plt.close()
                                
                                # Update the main figure title to reflect the new active category
                                fig.suptitle(f"Bloc ({block_x},{block_y}) - Sélection\n"
                                            f"Classe active: {class_dict[current_category_key]} (key: {current_category_key})\n"
                                            f"'0' pour tout sélectionner/désélectionner, Espace pour changer de classe\n"
                                            f"Pavé numérique et flèches pour sélectionner des régions")
                                fig.canvas.draw_idle()
                                return
                
                plt.gcf().canvas.mpl_connect('button_press_event', on_cat_click)
                plt.show(block=True)  # This is OK to block since it's a separate window
                
                # Remove this duplicate update since we now do it in on_cat_click
                # fig.suptitle(f"Bloc ({block_x},{block_y}) - Sélection\n"
                #            f"Classe active: {class_dict[current_category_key]} (key: {current_category_key})\n"
                #            f"'1' pour tout sélectionner, '0' pour tout désélectionner, Espace pour changer de classe")
                # fig.canvas.draw_idle()
                return
            
            # Number pad grid selection (dividing plot into 9 equal regions)
            if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                key_num = int(event.key)
            
                
                # Adjust numpad mapping to match arrow key directions:
                # Arrow keys: left=top half, right=bottom half, up=left half, down=right half
                #
                # So numpad should follow:
                # 7 8 9  ->  top-left, top-center, top-right
                # 4 5 6  ->  middle-left, middle-center, middle-right
                # 1 2 3  ->  bottom-left, bottom-center, bottom-right

                # Custom key mapping to exchange specific keys:
                # Exchange: 4 and 8, 1 and 9, 2 and 6
                if key_num == 4:
                    key_num = 8
                elif key_num == 8:
                    key_num = 4
                elif key_num == 1:
                    key_num = 9
                elif key_num == 9:
                    key_num = 1
                elif key_num == 2:
                    key_num = 6
                elif key_num == 6:
                    key_num = 2
                
                # Adjust numpad mapping to match arrow key directions after key exchanges
                
                # For numpad: calculate row and column
                # Row mapping: 7,8,9 = row 0 (top), 4,5,6 = row 1 (middle), 1,2,3 = row 2 (bottom)
                # Col mapping: 7,4,1 = col 0 (left), 8,5,2 = col 1 (center), 9,6,3 = col 2 (right)
                row = (key_num - 1) // 3  # 0 = top row, 2 = bottom row
                row = 2 - row  # Flip rows: 0 = bottom row, 2 = top row - to match "left"=top
                
                col = (key_num - 1) % 3  # 0 = left, 1 = center, 2 = right 
                
                # Calculate the segment size (each region is 1/3 of total size)
                segment_size_x = samples_plot_nb // 3
                segment_size_y = samples_plot_nb // 3
                
                # Calculate start and end indices for the selected region
                # Apply the same orientation as arrow keys:
                # x corresponds to columns (left-right) in the displayed grid
                # y corresponds to rows (top-bottom) in the displayed grid
                start_x = block_x * samples_plot_nb + col * segment_size_x
                end_x = start_x + segment_size_x
                start_y = block_y * samples_plot_nb + row * segment_size_y
                end_y = start_y + segment_size_y
                
                # Toggle all samples in this region
                all_selected = True
                all_unselected = True
                
                # First pass: check if all are selected or all are unselected
                for i_x in range(start_x, min(end_x, block_x * samples_plot_nb + samples_plot_nb)):
                    for i_y in range(start_y, min(end_y, block_y * samples_plot_nb + samples_plot_nb)):
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            if selection_state.get(key) == current_category_key:
                                all_unselected = False
                            else:
                                all_selected = False
                
                # Second pass: toggle based on state
                for i_x in range(start_x, min(end_x, block_x * samples_plot_nb + samples_plot_nb)):
                    for i_y in range(start_y, min(end_y, block_y * samples_plot_nb + samples_plot_nb)):
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            if all_selected:
                                selection_state[key] = None  # Deselect all if all were selected
                            else:
                                selection_state[key] = current_category_key  # Select all otherwise
                
                update_plot_titles(fig)
                return
            
            # 0 key: toggle selection for entire block
            elif event.key == '0':
                # Check if all cells are already selected
                all_selected = True
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            if selection_state.get((i_x, i_y)) != current_category_key:
                                all_selected = False
                                break
                    if not all_selected:
                        break
                
                # Toggle selection for all cells
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            if all_selected:
                                selection_state[(i_x, i_y)] = None  # Deselect all
                            else:
                                selection_state[(i_x, i_y)] = current_category_key  # Select all
                
                update_plot_titles(fig)
            # Flèche haut : toggle moitié supérieure
            elif event.key == 'left':
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb // 2):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            # Flèche bas : toggle moitié inférieure
            elif event.key == 'right':
                for dx in range(samples_plot_nb):
                    for dy in range(samples_plot_nb // 2, samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            # Flèche gauche : toggle moitié gauche
            elif event.key == 'up':
                for dx in range(samples_plot_nb // 2):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            # Flèche droite : toggle moitié droite
            elif event.key == 'down':
                for dx in range(samples_plot_nb // 2, samples_plot_nb):
                    for dy in range(samples_plot_nb):
                        i_x = block_x * samples_plot_nb + dx
                        i_y = block_y * samples_plot_nb + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
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
                    r, g, b, _, _ = sample.get_RGBZT()
                    if r is not None and g is not None and b is not None:
                        rgb = np.dstack((r, g, b)).astype(np.uint8)
                        ax.imshow(rgb)
                    ax.img_idx = (i_x, i_y)
                    ax.axis("off")
                    cat_key = selection_state.get((i_x, i_y), None)
                    if cat_key is not None:
                        ax.set_title(f"{cat_key}")
                    else:
                        ax.set_title("")
                else:
                    axes[dx, dy].axis("off")  # Hide axes for out-of-bounds cells
        fig.suptitle(f"Bloc ({block_x},{block_y}) - Sélection\n"
                    f"Classe active: {class_dict[current_category_key]} (key: {current_category_key})\n"
                    f"'0' pour tout sélectionner/désélectionner, Espace pour changer de classe\n"
                    f"Pavé numérique et flèches pour sélectionner des régions")
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Use non-blocking display to prevent multiple event loops
        plt.ion()  # Turn on interactive mode
        plt.show()
        plt.pause(0.001)  # Small pause to update the display
        
        # Wait for the figure to be closed
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
        
        plt.ioff()  # Turn off interactive mode

    def on_click(event):
        nonlocal current_category_key
        ax = event.inaxes
        if ax is not None:
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                current_state = selection_state.get(idx, None)
                # Toggle between current category and None
                if current_state is None:
                    selection_state[idx] = current_category_key
                else:
                    selection_state[idx] = None
                
                # Update title to show category key or empty
                if selection_state[idx] is not None:
                    ax.set_title(f"{selection_state[idx]}")
                else:
                    ax.set_title("")
                plt.draw()

    def update_plot_titles(fig):
        for ax in fig.get_axes():
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                cat_key = selection_state.get(idx, None)
                if cat_key is not None:
                    ax.set_title(f"{cat_key}")
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
            print(f"Classe active: {class_dict[current_category_key]} (key: {current_category_key})")
            print("Appuyer sur 'q' pour arrêter la sélection et sauvegarder")
            # print("Appuyer sur '0' pour tout sélectionner/désélectionner")
            # print("Appuyer sur Espace pour changer de classe")
            # print("Appuyer sur 1-9 pour sélectionner les régions:")
            # print("  7=haut-gauche, 8=haut-centre, 9=haut-droite")
            # print("  4=milieu-gauche, 5=centre, 6=milieu-droite")
            # print("  1=bas-gauche, 2=bas-centre, 3=bas-droite")

            # Check for early exit after each block
            if early_exit:
                print("Arrêt anticipé de la sélection. Sauvegarde des échantillons sélectionnés...")
                break
        if early_exit:
            break

    # Create a dictionary to organize samples by category
    samples_by_category = {}
    
    # Process selected samples
    for (i_x, i_y), cat_key in selection_state.items():
        if cat_key is not None and i_y < n_samples_y and i_x < n_samples_x:
            category_name = class_dict[cat_key]
            if category_name not in samples_by_category:
                samples_by_category[category_name] = []
                
            sample = samples_matrix[i_y][i_x]
            sample.category = category_name
            samples_by_category[category_name].append(sample)
    
    # Create a combined set with all samples
    all_selected_set = SamplesSet2(ds_path=ds_path, dz_path=dz_path, dt_path=dt_path)
    
    # Create individual sets for each category and save them
    for category_name, samples in samples_by_category.items():
        # Create a sample set for this category
        sample_set = SamplesSet2(ds_path=ds_path, dz_path=dz_path, dt_path=dt_path)
        
        # Add samples to this category's set
        for sample in samples:
            sample_set.add_Sample(sample)
            all_selected_set.add_Sample(sample)
        
        # Calculate features if there are samples
        if len(sample_set.samples) > 0:
            sample_set.fill_neighbors_all(distance_large=distance_large)
        
        # Create unique filename for this category
        idx = 1
        while os.path.exists(os.path.join(file_path, f"{category_name}_{idx}.json")):
            idx += 1
        json_path = os.path.join(file_path, f"{category_name}_{idx}.json")
        png_path = os.path.join(file_path, f"{category_name}_{idx}.png")
        
        # Visualize and save this category's samples
        if len(sample_set.samples) > 0:
            plt.figure(figsize=(20, 20))
            sample_set.plot_samples_as_list()
            plt.suptitle(f"{category_name} : {len(sample_set.samples)} samples")
            plt.savefig(png_path)
            plt.close()
            
            # Save the sample set for this category
            sample_set.save_samples_to_json(filename=json_path)
            print(f"{len(sample_set.samples)} samples '{category_name}' sauvegardés dans {json_path}")
    
    # Save/update the count file
    txt_path = os.path.join(file_path, "pop_counts.txt")
    counts = {}
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    counts[k.strip()] = int(v.strip())
    
    # Update counts with new samples
    for category_name, samples in samples_by_category.items():
        counts[category_name] = counts.get(category_name, 0) + len(samples)
    
    # Write updated counts
    with open(txt_path, "w") as f:
        for k, v in counts.items():
            f.write(f"{k}: {v}\n")
    
    # Renommage de la fenêtre de sélection si nécessaire
    import shutil
    src_img = os.path.join(file_path, "fenetre_selection.png")
    dst_img = os.path.join(file_path, "last_selection_window.png")
    if os.path.exists(src_img) and src_img != dst_img:
        shutil.copy(src_img, dst_img)
        print(f"fenetre_selection.png copié vers last_selection_window.png")
    
    # Summary of all selections
    total_samples = sum(len(samples) for samples in samples_by_category.values())
    print(f"\nRésumé des sélections:")
    for category_name, samples in samples_by_category.items():
        print(f"- {category_name}: {len(samples)} samples")
    print(f"Total: {total_samples} samples")
    
    # Free raster resources
    all_samples_set.clear_rasters()
    all_selected_set.clear_rasters()
    for category_name, samples in samples_by_category.items():
        if category_name in locals():
            locals()[f"sample_set_{category_name}"].clear_rasters()
    rasters_mgr.clear_rasters()


if __name__ == "__main__":
    wap = 23
    size_patch = 16
    distance_large = 3

    n_samples_x = 30
    n_samples_y = 30
    samples_plot_nb = 15
    default = None
        
    x_start = 1000
    y_start = 1000

    # Chemins des rasters
    nb = "08_06"
    
    ds_path = f"drone_treated/WAP{wap}_tiles/rgb/Wap{wap}_main_transparent_mosaic_group1_{nb}.tif"
    dz_path = f"drone_treated/WAP{wap}_tiles/dsm/Wap{wap}_main_dsm_{nb}.tif"
    dt_path = None
    
    save_dir = "data/samples/selection13/"

    class_dict = {
        # "pp": "peat_plateau",
        # "ld" : "large_depression",
        # "fo": "forest",
        # "la": "lake"
        "l": "lichen",
        "s": "sphaignes",
        "c" : "chicoutai",
        "gd": "green_depression",
        "wd": "watered_depression",
        "dd" : "dry_depression",
        "bd": "black_depression",
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
