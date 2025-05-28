import matplotlib.pyplot as plt
import numpy as np


class Sample2:
    def __init__(self, i_x, i_y, x, y, size_patch, rast_r, rast_g, rast_b, rast_z=None, rast_t=None, category=None):
        self.size_patch = size_patch
        self.i_x = i_x
        self.i_y = i_y
        self.x = x
        self.y = y
        self.category = category

        # Patch extraction
        r = rast_r[x:x + size_patch, y:y + size_patch]
        g = rast_g[x:x + size_patch, y:y + size_patch]
        b = rast_b[x:x + size_patch, y:y + size_patch]
        self.r_mean = float(np.mean(r))
        self.g_mean = float(np.mean(g))
        self.b_mean = float(np.mean(b))
        self.r_var = float(np.var(r))
        self.g_var = float(np.var(g))
        self.b_var = float(np.var(b))

        # Altitude (z)
        if rast_z is not None:
            z = rast_z[x:x + size_patch, y:y + size_patch]
            self.z_mean = float(np.mean(z))
            self.z_var = float(np.var(z))
        else:
            self.z_mean = None
            self.z_var = None

        # Température (t)
        if rast_t is not None:
            t = rast_t[x:x + size_patch, y:y + size_patch]
            self.t_mean = float(np.mean(t))
            self.t_var = float(np.var(t))
        else:
            self.t_mean = None
            self.t_var = None

        # Voisinages et gradients
        self.r_n_mean = self.g_n_mean = self.b_n_mean = None
        self.t_n_mean = None
        self.r_n_var = self.g_n_var = self.b_n_var = None
        self.t_n_var = self.z_n_var = None
        self.z_moins_z_n = None
        
        # Stockage des rasters pour méthodes ultérieures
        self._rast_r = rast_r
        self._rast_g = rast_g
        self._rast_b = rast_b
        self._rast_z = rast_z
        self._rast_t = rast_t

    def get_RGBZT(self):
        r = self._rast_r[self.x:self.x + self.size_patch, self.y:self.y + self.size_patch]
        g = self._rast_g[self.x:self.x + self.size_patch, self.y:self.y + self.size_patch]
        b = self._rast_b[self.x:self.x + self.size_patch, self.y:self.y + self.size_patch]
        z = self._rast_z[self.x:self.x + self.size_patch, self.y:self.y + self.size_patch] if self._rast_z is not None else None
        t = self._rast_t[self.x:self.x + self.size_patch, self.y:self.y + self.size_patch] if self._rast_t is not None else None
        return r, g, b, z, t

    def get_z_mean(self, x, y, sample_set=None, size_patch=None):
        global rast_z
        if sample_set is not None and (x, y) in sample_set.samples:
            return sample_set.samples[(x, y)].z_mean
        elif 0 <= x < rast_z.shape[0] - size_patch and 0 <= y < rast_z.shape[1] - size_patch:
            z_patch = rast_z[x:x + size_patch, y:y + size_patch] #en mm
            return float(np.mean(z_patch))
        else:
            return None
        
    def compute_neighbors_all(self, sample_set, depth_neighbors=1, depth_neighbors_z=1):
        # RGB/T
        r_sum = g_sum = b_sum = t_sum = 0
        r_var_sum = g_var_sum = b_var_sum = t_var_sum = 0
        nb_neighbors = 0
        for i in range(-depth_neighbors, depth_neighbors + 1):
            for j in range(-depth_neighbors, depth_neighbors + 1):
                if i == 0 and j == 0:
                    continue
                x_n = self.x + i * self.size_patch
                y_n = self.y + j * self.size_patch
                neighbor = sample_set.samples.get((x_n, y_n))
                if neighbor:
                    r_sum += neighbor.r_mean
                    g_sum += neighbor.g_mean
                    b_sum += neighbor.b_mean
                    r_var_sum += neighbor.r_var
                    g_var_sum += neighbor.g_var
                    b_var_sum += neighbor.b_var
                    if self._rast_t is not None:
                        t_sum += neighbor.t_mean
                        t_var_sum += neighbor.t_var
                    nb_neighbors += 1
        if nb_neighbors > 0:
            self.r_n_mean = r_sum / nb_neighbors
            self.g_n_mean = g_sum / nb_neighbors
            self.b_n_mean = b_sum / nb_neighbors
            self.r_n_var = r_var_sum / nb_neighbors
            self.g_n_var = g_var_sum / nb_neighbors
            self.b_n_var = b_var_sum / nb_neighbors
            if self._rast_t is not None:
                self.t_n_mean = t_sum / nb_neighbors
                self.t_n_var = t_var_sum / nb_neighbors
            else:
                self.t_n_mean = None
                self.t_n_var = None

        # Z
        if self._rast_z is not None:
            z_sum = z_var_sum = 0
            nb_neighbors_z = 0
            for i in range(-depth_neighbors_z, depth_neighbors_z + 1):
                for j in range(-depth_neighbors_z, depth_neighbors_z + 1):
                    if i == 0 and j == 0:
                        continue
                    x_n = self.x + i * self.size_patch
                    y_n = self.y + j * self.size_patch
                    neighbor = sample_set.samples.get((x_n, y_n))
                    if neighbor and neighbor.z_mean is not None:
                        z_sum += neighbor.z_mean
                        z_var_sum += neighbor.z_var
                        nb_neighbors_z += 1
            if nb_neighbors_z > 0:
                self.z_moins_z_n = self.z_mean - z_sum / nb_neighbors_z
                self.z_n_var = z_var_sum / nb_neighbors_z
            else:
                self.z_moins_z_n = None
                self.z_n_var = None
        else:
            self.z_moins_z_n = None
            self.z_n_var = None

    def __repr__(self):
        def fmt(val):
            return f"{val:.2f}" if val is not None else "None"
        return (f"Sample2(x={self.x}, y={self.y}, "
                f"r_mean={fmt(self.r_mean)}, g_mean={fmt(self.g_mean)}, b_mean={fmt(self.b_mean)}, "
                f"z_mean={fmt(self.z_mean)}, z_var={fmt(self.z_var)}, t_mean={fmt(self.t_mean)}, "
                f"r_n_mean={fmt(self.r_n_mean)}, g_n_mean={fmt(self.g_n_mean)}, b_n_mean={fmt(self.b_n_mean)}, t_n_mean={fmt(self.t_n_mean)}, "
                f"z_moins_z_n={fmt(self.z_moins_z_n)}, "
                f"r_n_var={fmt(self.r_n_var)}, g_n_var={fmt(self.g_n_var)}, b_n_var={fmt(self.b_n_var)}, t_n_var={fmt(self.t_n_var)}, "
                f"z_n_var={fmt(self.z_n_var)}"
        )
    
    def plot_sample(self):
        def fmt2(val):
            return f"{val:.2f}" if val is not None else "None"
        def fmt5(val):
            return f"{val:.5f}" if val is not None else "None"
        
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))  # 1 row, 3 columns

        r,g,b,z, t = self.get_RGBZT()

        rgb = np.dstack((r, g, b))
        axes[0].imshow(rgb)
        axes[0].set_title(f"x = {self.x}, y = {self.y}")
        axes[0].axis("off")

        # Statistiques de la cellule
        cell_text = (
            f"Sample(x={self.x}, y={self.y}, "
            f"r_mean={fmt2(self.r_mean)}, g_mean={fmt2(self.g_mean)}, b_mean={fmt2(self.b_mean)}, "
            #f"r_var={fmt2(self.r_var)}, g_var={fmt2(self.g_var)}, b_var={fmt2(self.b_var)}, "
            f"z_mean={fmt2(self.z_mean)}, z_var={fmt5(self.z_var)}, t_mean={fmt2(self.t_mean)}, "
        )
        axes[1].text(0, 0.5, cell_text, fontsize=10, ha="left", va="center", wrap=True)
        axes[1].axis("off")

        # Statistiques des voisins et gradients
            
        neighbor_text = (
            f"R (neighbors): mean={fmt2(self.r_n_mean)}\n"
            f"G (neighbors): mean={fmt2(self.g_n_mean)}\n"
            f"B (neighbors): mean={fmt2(self.b_n_mean)}\n"
            f"T (neighbors): mean={fmt2(self.t_n_mean)}\n"
            f"z_moins_z_n: {fmt5(self.z_moins_z_n)}"
            f"r_n_var={fmt2(self.r_n_var)}, g_n_var={fmt2(self.g_n_var)}, b_n_var={fmt2(self.b_n_var)}, t_n_var={fmt2(self.t_n_var)}\n"
            f"z_n_var={fmt2(self.z_n_var)}\n"

        )
        axes[2].text(0, 0.5, neighbor_text, fontsize=10, ha="left", va="center", wrap=True)
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    

if __name__ == "__main__":
    from osgeo import gdal
    ds = gdal.Open(r'data/rgb_reshaped.tif')
    dz = gdal.Open(r'data/dsm_reshaped.tif')
    dt = gdal.Open(r'data/thermal_reshaped.tif')

    rast_r = ds.GetRasterBand(1).ReadAsArray()
    rast_g = ds.GetRasterBand(2).ReadAsArray()
    rast_b = ds.GetRasterBand(3).ReadAsArray()
    rast_z = dz.GetRasterBand(1).ReadAsArray()*1000
    rast_t = dt.GetRasterBand(1).ReadAsArray()

    sample = Sample2(i_x=0, i_y=0, x=1000, y=1000, size_patch=128,
                     rast_r=rast_r, rast_g=rast_g, rast_b=rast_b,
                     rast_z= rast_z, rast_t=None)

    print(sample)
    sample.plot_sample()
