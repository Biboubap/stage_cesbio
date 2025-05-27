import sys
import numpy as np
import rasterio

input_path = sys.argv[1]
output_path = sys.argv[2]

with rasterio.open(input_path) as src:
    data = src.read()  # shape: (13, rows, cols)
    median = np.median(data, axis=0).astype(src.dtypes[0])  # shape: (rows, cols)

    profile = src.profile
    profile.update(count=1)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(median, 1)