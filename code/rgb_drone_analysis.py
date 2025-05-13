from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
#https://www.youtube.com/watch?v=p_BsFdV_LUk&list=PL4aUQR9L9RFp7kuu38hInDE-9ByueEMES

gdal.UseExceptions()


ds = gdal.Open(r'data/twin lake mosaïc.tif')
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

# Read the three bands

r = ds.GetRasterBand(1).ReadAsArray()[:15000] #SHAPE (49674, 23408)
g = ds.GetRasterBand(2).ReadAsArray()[:15000]
b = ds.GetRasterBand(3).ReadAsArray()[:15000]

def plot_rgb(r, g, b, nmin = 10000, nmax = 15000):
    # Stack the bands into an RGB image
    rgb = np.dstack((r, g, b))

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb[nmin:nmax, nmin:nmax])
    plt.title("RGB Image")
    plt.axis("off")
    plt.show()


def to_white(r, g, b):
    """
    Convert RGB array to white array
    """
    # Stack the bands into an RGB image
    avg = (r + g + b) / 3
    return avg

def plot_value(v_array, nmin = 10000, nmax = 15000):
    """
    Plot white array
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(v_array[nmin:nmax, nmin:nmax], cmap='gray')
    plt.title("White Image")
    plt.axis("off")
    # plt.show()

def plot_combined(r, g, b, nmin=10000, nmax=15000):
    """
    Plot RGB and grayscale images side by side in a single figure
    """
    # Convert RGB to grayscale
    white_array = to_white(r, g, b)
    # Create a single figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # 1 row, 2 columns
    # Plot RGB image
    rgb = np.dstack((r, g, b))
    axes[0].imshow(rgb[nmin:nmax, nmin:nmax])
    axes[0].set_title("RGB Image")
    axes[0].axis("off")
    # Plot grayscale image
    axes[1].imshow(white_array[nmin:nmax, nmin:nmax], cmap='gray')
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")
    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

def plot_rgb_and_bands(r, g, b, nmin=10000, nmax=15000):
    """
    Plot a large RGB image and three smaller plots for R, G, and B bands
    """
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 15))
    
    # Add the large RGB plot (spanning multiple grid cells)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)  # Large plot
    rgb = np.dstack((r, g, b))
    ax1.imshow(rgb[nmin:nmax, nmin:nmax])
    ax1.set_title("RGB Image")
    ax1.axis("off")
    
    # Add the smaller R, G, and B plots
    ax2 = plt.subplot2grid((3, 3), (2, 0))  # Small R plot
    ax2.imshow(r[nmin:nmax, nmin:nmax], cmap='Reds')
    ax2.set_title("Red Band")
    ax2.axis("off")
    
    ax3 = plt.subplot2grid((3, 3), (2, 1))  # Small G plot
    ax3.imshow(g[nmin:nmax, nmin:nmax], cmap='Greens')
    ax3.set_title("Green Band")
    ax3.axis("off")
    
    ax4 = plt.subplot2grid((3, 3), (2, 2))  # Small B plot
    ax4.imshow(b[nmin:nmax, nmin:nmax], cmap='Blues')
    ax4.set_title("Blue Band")
    ax4.axis("off")
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Plot the RGB image
    plot_rgb(r, g, b, nmin=5000, nmax=10000)

    # Plot the white image
    # plot_value(to_white(r, g, b))  

    # Plot RGB and grayscale images in a single figure
    # plot_combined(r, g, b)

    # Plot the RGB image and individual bands
    # plot_rgb_and_bands(r, g, b)