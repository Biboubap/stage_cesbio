from samples_set import SamplesSet
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Création du set initial
    x_s1 = 16134
    y_s1 = 8910
    n_samples_x = 10
    n_samples_y = 10
    size_patch = 32

    category= "sphegnes"
    nb_pop = 6
    x_s1 += n_samples_x * 1 * size_patch
    y_s1 += n_samples_y * 1 * size_patch
    row_start = int(np.round(x_s1 / 32)) * 32  # row X
    column_start = int(np.round(y_s1 / 32)) * 32  # column Y
    pop_sphegnes_1 = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category=category)
    pop_sphegnes_1.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)

    # Récupère la liste des samples dans l'ordre de plot_samples_as_list
    samples = pop_sphegnes_1.samples

    # Affichage interactif pour sélection
    selected = set()

    def on_click(event):
        ax = event.inaxes
        if ax is not None:
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                if idx in selected:
                    selected.remove(idx)
                    ax.set_title("")
                else:
                    selected.add(idx)
                    ax.set_title("✔")
                plt.draw()

    n = len(samples)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, sample in enumerate(samples):
        i_row = idx // n_cols
        i_col = idx % n_cols
        ax = axes[i_row, i_col]
        r, g, b, _ = sample.get_RGBZ()
        rgb = np.dstack((r, g, b))
        ax.imshow(rgb)
        ax.img_idx = idx
        ax.axis("off")

    # Masquer les axes vides
    for idx in range(n, n_rows * n_cols):
        i_row = idx // n_cols
        i_col = idx % n_cols
        axes[i_row, i_col].axis("off")

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    # Après fermeture de la fenêtre, créer le set sélectionné
    selected_samples = [samples[idx] for idx in sorted(selected)]
    pop_sphegnes1_selected = SamplesSet(n_samples_x=None, n_samples_y=None, category="sphegnes_selected")
    for s in selected_samples:
        pop_sphegnes1_selected.add_sample(s)

    # Affiche la sélection
    pop_sphegnes1_selected.plot_samples_as_list()

    # Sauvegarde
    path = "data/samples/sphegnes/"
    pop_sphegnes1_selected.save_samples_to_json(filename=path + f"pop_sphegnes{nb_pop}_selected.json")