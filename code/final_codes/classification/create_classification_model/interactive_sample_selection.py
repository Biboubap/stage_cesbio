#!/usr/bin/env python3
"""
Interactive Sample Selection Tool

Direct adaptation of pop_selection4.py that uses block-based utility classes
and accepts input through command-line arguments.

Usage:
  python interactive_sample_selection.py --rgb path/to/rgb.tif --dsm path/to/dsm.tif 
                                        --classes "Class1,Class2,Class3" 
                                        --output path/to/output_dir
                                        --sample-size 32
                                        --distance-start-horizontal 0 --distance-start-vertical 0
                                        --blocks-x 4 --blocks-y 4
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import shutil
import json
from collections import Counter
from osgeo import gdal

# Add parent directory to path to import utility modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sample_set import SampleSet

# Global constants
SAMPLES_PER_WINDOW = 12  # Fixed number of samples in each dimension per window

def meters_to_pixels(raster_path, distance_horizontal, distance_vertical):
    """
    Convert distances in meters to pixel coordinates based on the raster's resolution.
    
    Args:
        raster_path: Path to the raster file
        distance_horizontal: Distance in meters from left edge
        distance_vertical: Distance in meters from top edge
        
    Returns:
        Tuple of (pixel_x, pixel_y) coordinates
    """
    ds = gdal.Open(raster_path)
    if ds is None:
        raise ValueError(f"Could not open raster file: {raster_path}")
    
    # Get geotransform: (originX, pixelWidth, 0, originY, 0, pixelHeight)
    geotransform = ds.GetGeoTransform()
    
    # Extract origin and pixel dimensions
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = abs(geotransform[5])  # Usually negative, need absolute value
    
    # Calculate pixel coordinates
    pixel_x = int(distance_horizontal / pixel_width)
    pixel_y = int(distance_vertical / pixel_height)
    
    print(f"Converting: {distance_horizontal}m horizontal, {distance_vertical}m vertical")
    print(f"Raster resolution: {pixel_width}m × {pixel_height}m per pixel")
    print(f"Converted to pixel coordinates: ({pixel_x}, {pixel_y})")
    
    # Close the dataset
    ds = None
    
    return pixel_x, pixel_y

def plot_rgb(xmin, ymin, xmax, ymax, ds_path, output_path, show=False):
    """Plot RGB image with the given boundaries and save it to the path."""
    from osgeo import gdal
    
    ds = gdal.Open(ds_path)
    
    # Read the three bands
    r = ds.GetRasterBand(1).ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin)
    g = ds.GetRasterBand(2).ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin)
    b = ds.GetRasterBand(3).ReadAsArray(xmin, ymin, xmax - xmin, ymax - ymin)
    
    # Normalize values for proper display
    r_norm = np.clip(r / 255.0, 0, 1)
    g_norm = np.clip(g / 255.0, 0, 1)
    b_norm = np.clip(b / 255.0, 0, 1)
    
    rgb = np.dstack((r_norm, g_norm, b_norm))
    plt.figure(figsize=(SAMPLES_PER_WINDOW, SAMPLES_PER_WINDOW))
    plt.imshow(rgb)
    plt.title("Fenêtre samples")
    plt.savefig(os.path.join(output_path, "fenetre_selection.png"))
    if show:
        plt.show()  
    plt.close()

def parse_args():
    """Parse command line arguments for the interactive sample selection tool."""
    parser = argparse.ArgumentParser(description="Interactive tool for selecting and categorizing sample patches from drone imagery")
    
    # Required arguments
    parser.add_argument("--rgb", required=True, help="Path to RGB drone image (GeoTIFF)")
    parser.add_argument("--output", required=True, help="Directory to save selected samples")
    parser.add_argument("--classes", required=True, help="Comma-separated list of class names")
    
    # Optional arguments
    parser.add_argument("--dsm", help="Path to Digital Surface Model (GeoTIFF)")
    parser.add_argument("--thermal", help="Path to thermal image (GeoTIFF)")
    parser.add_argument("--sample-size", type=int, default=32, help="Size of each sample patch in pixels")
    parser.add_argument("--distance-start-horizontal", type=float, default=0, 
                        help="Distance in meters from the left edge of the image")
    parser.add_argument("--distance-start-vertical", type=float, default=0, 
                        help="Distance in meters from the top edge of the image")
    parser.add_argument("--blocks-x", type=int, default=4, help="Number of blocks to process in X direction")
    parser.add_argument("--blocks-y", type=int, default=4, help="Number of blocks to process in Y direction")
    
    return parser.parse_args()

def pop_selection_adapted(x_start, y_start, n_samples_x, n_samples_y, size_patch, 
                        output_path, rgb_path=None, dsm_path=None, thermal_path=None, 
                        distance_large=3, class_dict=None):
    """
    Interactive tool to select samples for training and testing models.
    
    Args:
        x_start: Starting x coordinate (pixel)
        y_start: Starting y coordinate (pixel)
        n_samples_x: Number of samples in x direction
        n_samples_y: Number of samples in y direction
        size_patch: Size of each patch in pixels
        output_path: Directory where to save results
        rgb_path: Path to RGB image
        dsm_path: Path to DSM image
        thermal_path: Path to thermal image
        distance_large: Distance for large neighborhood calculation
        class_dict: Dictionary mapping shortcut keys to class names
    """
    print("Starting sample selection...")
    # Round start coordinates to match patch size
    x_start = int(np.round(x_start/size_patch))*size_patch
    y_start = int(np.round(y_start/size_patch))*size_patch
    x_max = x_start + n_samples_x * size_patch
    y_max = y_start + n_samples_y * size_patch
    print(f"Grid coordinates: ({x_start}, {y_start}) to ({x_max}, {y_max})")
    os.makedirs(output_path, exist_ok=True)
    
    # Show selection window preview
    if rgb_path:
        plot_rgb(x_start, y_start, x_max, y_max, rgb_path, output_path, show=True)
        
    
    # Create samples set
    all_samples_set = SampleSet(
        rgb_path=rgb_path,
        dsm_path=dsm_path,
        thermal_path=thermal_path,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y
    )
    all_samples_set.create_samples_grid(x_start=x_start, y_start=y_start, size_patch=size_patch)
    samples_matrix = all_samples_set.get_samples_matrix()
    
    # Default class dictionary if not provided
    if class_dict is None:
        class_dict = {}
        # Use first letters of class names
        for i, class_name in enumerate(["Class1", "Class2", "Class3", "Class4", "Class5"]):
            key = class_name[0].lower()
            class_dict[key] = class_name
    
    # Track current active category
    current_category_key = None
    early_exit = False
    
    # Get initial category
    while True:
        print(f"Available classes: {', '.join([f'{k}: {v}' for k, v in class_dict.items()])}")
        class_key = input(f"Select initial class (key): ").strip().lower()
        if class_key in class_dict:
            current_category_key = class_key
            break
        print("Invalid class key. Try again.")
    
    print(f"Selected class: {class_dict[current_category_key]} (key: {current_category_key})")
    
    # Initialize selection state
    selection_state = {(i_x, i_y): None for i_x in range(n_samples_x) for i_y in range(n_samples_y)}
    
    def show_category_selector(current_key, block_x, block_y, main_fig):
        """Show category selector window."""
        nonlocal current_category_key
        
        # Calculate how many categories we can fit per page
        categories_per_page = 12
        num_categories = len(class_dict)
        num_pages = (num_categories + categories_per_page - 1) // categories_per_page
        current_page = 0

        def show_page(page_num):
            plt.figure(figsize=(5, 6))
            plt.axis('off')
            cat_buttons = {}
            
            plt.text(0.1, 0.95, f"Select category (Page {page_num+1}/{num_pages}):", fontsize=12)
            
            # Add navigation arrows if multiple pages
            if num_pages > 1:
                if page_num > 0:
                    prev_btn = plt.Rectangle((0.2, 0.02), 0.2, 0.05, alpha=0.2, facecolor='lightblue', edgecolor='black')
                    plt.gca().add_patch(prev_btn)
                    plt.text(0.25, 0.04, "Prev", ha='center', va='center')
                    cat_buttons['prev'] = prev_btn
                
                if page_num < num_pages - 1:
                    next_btn = plt.Rectangle((0.6, 0.02), 0.2, 0.05, alpha=0.2, facecolor='lightblue', edgecolor='black')
                    plt.gca().add_patch(next_btn)
                    plt.text(0.65, 0.04, "Next", ha='center', va='center')
                    cat_buttons['next'] = next_btn
            
            # Get categories for current page
            start_idx = page_num * categories_per_page
            end_idx = min(start_idx + categories_per_page, num_categories)
            page_categories = list(class_dict.items())[start_idx:end_idx]
            
            # Create "buttons" for each category
            y_pos = 0.9
            for key, cat_name in page_categories:
                y_pos -= 0.06
                # Highlight the currently selected category
                if key == current_key:
                    rect = plt.Rectangle((0.1, y_pos-0.025), 0.8, 0.05, 
                                       alpha=0.4, facecolor='yellow', edgecolor='black')
                else:
                    rect = plt.Rectangle((0.1, y_pos-0.025), 0.8, 0.05, 
                                       alpha=0.2, facecolor='gray', edgecolor='black')
                plt.gca().add_patch(rect)
                plt.text(0.2, y_pos, f"{key}: {cat_name}", fontsize=10)
                cat_buttons[key] = rect
            
            plt.tight_layout()
            
            def on_cat_click(event):
                nonlocal current_page, current_category_key
                if event.inaxes:
                    # Navigation buttons
                    if 'prev' in cat_buttons and cat_buttons['prev'].contains(event)[0]:
                        plt.close()
                        show_page(current_page - 1)
                        return
                    elif 'next' in cat_buttons and cat_buttons['next'].contains(event)[0]:
                        plt.close()
                        show_page(current_page + 1)
                        return
                    
                    # Category buttons
                    for key, rect in cat_buttons.items():
                        if key not in ['prev', 'next'] and rect.contains(event)[0]:
                            current_category_key = key
                            print(f"Active class changed to: {class_dict[current_category_key]} (key: {current_category_key})")
                            plt.close()
                            
                            # Update the main figure title
                            main_fig.suptitle(f"Block ({block_x},{block_y}) - Selection\n"
                                           f"Active class: {class_dict[current_category_key]} (key: {current_category_key})\n"
                                           f"'0' to select/deselect all, Space to change class\n"
                                           f"Number pad and arrow keys to select regions")
                            main_fig.canvas.draw_idle()
                            return
            
            plt.gcf().canvas.mpl_connect('button_press_event', on_cat_click)
            plt.show(block=True)
        
        # Show the first page
        show_page(current_page)
    
    def plot_block(block_x, block_y):
        """Plot a block of samples for interactive selection."""
        nonlocal current_category_key, early_exit
        
        plt.close('all')
        fig, axes = plt.subplots(SAMPLES_PER_WINDOW, SAMPLES_PER_WINDOW, figsize=(20, 20))
        axes = np.array(axes).reshape(SAMPLES_PER_WINDOW, SAMPLES_PER_WINDOW)
        
        def on_key_press(event):
            nonlocal current_category_key, early_exit
            
            # Early exit with 'q'
            if event.key == 'q':
                print("Early exit requested. Saving current selections...")
                early_exit = True
                plt.close(fig)
                return
            
            # Switch category with space
            if event.key == ' ':
                show_category_selector(current_category_key, block_x, block_y, fig)
                return
            
            # Number pad grid selection
            if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                key_num = int(event.key)
                
                # Custom key mapping
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
                
                # Calculate row and column
                row = (key_num - 1) // 3
                row = 2 - row  # Flip rows
                col = (key_num - 1) % 3
                
                # Calculate segment size
                segment_size_x = SAMPLES_PER_WINDOW // 3
                segment_size_y = SAMPLES_PER_WINDOW // 3
                
                # Calculate start and end indices
                start_x = block_x * SAMPLES_PER_WINDOW + col * segment_size_x
                end_x = start_x + segment_size_x
                start_y = block_y * SAMPLES_PER_WINDOW + row * segment_size_y
                end_y = start_y + segment_size_y
                
                # Check if all are selected or all are unselected
                all_selected = True
                all_unselected = True
                
                for i_x in range(start_x, min(end_x, block_x * SAMPLES_PER_WINDOW + SAMPLES_PER_WINDOW)):
                    for i_y in range(start_y, min(end_y, block_y * SAMPLES_PER_WINDOW + SAMPLES_PER_WINDOW)):
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            if selection_state.get(key) == current_category_key:
                                all_unselected = False
                            else:
                                all_selected = False
                
                # Toggle based on state
                for i_x in range(start_x, min(end_x, block_x * SAMPLES_PER_WINDOW + SAMPLES_PER_WINDOW)):
                    for i_y in range(start_y, min(end_y, block_y * SAMPLES_PER_WINDOW + SAMPLES_PER_WINDOW)):
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            if all_selected:
                                selection_state[key] = None
                            else:
                                selection_state[key] = current_category_key
                
                update_plot_titles(fig)
                return
            
            # Toggle selection for entire block with '0'
            elif event.key == '0':
                all_selected = True
                for dx in range(SAMPLES_PER_WINDOW):
                    for dy in range(SAMPLES_PER_WINDOW):
                        i_x = block_x * SAMPLES_PER_WINDOW + dx
                        i_y = block_y * SAMPLES_PER_WINDOW + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            if selection_state.get((i_x, i_y)) != current_category_key:
                                all_selected = False
                                break
                    if not all_selected:
                        break
                
                # Toggle selection
                for dx in range(SAMPLES_PER_WINDOW):
                    for dy in range(SAMPLES_PER_WINDOW):
                        i_x = block_x * SAMPLES_PER_WINDOW + dx
                        i_y = block_y * SAMPLES_PER_WINDOW + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            if all_selected:
                                selection_state[(i_x, i_y)] = None
                            else:
                                selection_state[(i_x, i_y)] = current_category_key
                
                update_plot_titles(fig)
                return
            
            # Arrow keys for region selection
            elif event.key == 'left':  # Toggle top half
                for dx in range(SAMPLES_PER_WINDOW):
                    for dy in range(SAMPLES_PER_WINDOW // 2):
                        i_x = block_x * SAMPLES_PER_WINDOW + dx
                        i_y = block_y * SAMPLES_PER_WINDOW + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            
            elif event.key == 'right':  # Toggle bottom half
                for dx in range(SAMPLES_PER_WINDOW):
                    for dy in range(SAMPLES_PER_WINDOW // 2, SAMPLES_PER_WINDOW):
                        i_x = block_x * SAMPLES_PER_WINDOW + dx
                        i_y = block_y * SAMPLES_PER_WINDOW + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            
            elif event.key == 'up':  # Toggle left half
                for dx in range(SAMPLES_PER_WINDOW // 2):
                    for dy in range(SAMPLES_PER_WINDOW):
                        i_x = block_x * SAMPLES_PER_WINDOW + dx
                        i_y = block_y * SAMPLES_PER_WINDOW + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            
            elif event.key == 'down':  # Toggle right half
                for dx in range(SAMPLES_PER_WINDOW // 2, SAMPLES_PER_WINDOW):
                    for dy in range(SAMPLES_PER_WINDOW):
                        i_x = block_x * SAMPLES_PER_WINDOW + dx
                        i_y = block_y * SAMPLES_PER_WINDOW + dy
                        if i_x < n_samples_x and i_y < n_samples_y:
                            key = (i_x, i_y)
                            selection_state[key] = current_category_key if selection_state[key] is None else None
                update_plot_titles(fig)
            
            # Close window with Enter
            elif event.key == 'enter':
                plt.close(fig)
        
        # Display samples in the grid
        for dx in range(SAMPLES_PER_WINDOW):
            for dy in range(SAMPLES_PER_WINDOW):
                i_x = block_x * SAMPLES_PER_WINDOW + dx
                i_y = block_y * SAMPLES_PER_WINDOW + dy
                if i_x < n_samples_x and i_y < n_samples_y:
                    ax = axes[dx, dy]
                    sample = samples_matrix[i_y][i_x]
                    r, g, b, _, _ = sample.get_RGBZT()
                    r = r.T
                    g = g.T
                    b = b.T

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
                    axes[dx, dy].axis("off")
        
        # Set figure title
        fig.suptitle(f"Block ({block_x},{block_y}) - Selection\n"
                   f"Active class: {class_dict[current_category_key]} (key: {current_category_key})\n"
                   f"'0' to select/deselect all, Space to change class\n"
                   f"Number pad and arrow keys to select regions, 'q' to quit")
        
        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Use interactive mode
        plt.ion()
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        plt.pause(0.001)
        
        # Wait for figure to be closed
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
        
        plt.ioff()
    
    def on_click(event):
        """Handle mouse click events to toggle sample selection."""
        ax = event.inaxes
        if ax is not None:
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                current_state = selection_state.get(idx, None)
                # Toggle between current category and None
                selection_state[idx] = None if current_state == current_category_key else current_category_key
                
                # Update title
                if selection_state[idx] is not None:
                    ax.set_title(f"{selection_state[idx]}")
                else:
                    ax.set_title("")
                plt.draw()
    
    def update_plot_titles(fig):
        """Update all plot titles based on selection state."""
        for ax in fig.get_axes():
            idx = getattr(ax, 'img_idx', None)
            if idx is not None:
                cat_key = selection_state.get(idx, None)
                if cat_key is not None:
                    ax.set_title(f"{cat_key}")
                else:
                    ax.set_title("")
        fig.canvas.draw_idle()
    
    # Process blocks
    n_blocks_x = (n_samples_x + SAMPLES_PER_WINDOW - 1) // SAMPLES_PER_WINDOW
    n_blocks_y = (n_samples_y + SAMPLES_PER_WINDOW - 1) // SAMPLES_PER_WINDOW
    for block_y in range(n_blocks_y):
        for block_x in range(n_blocks_x):
            plot_block(block_x, block_y)
            print(f"Block ({block_x},{block_y}) displayed. Close window to continue.")
            print(f"Active class: {class_dict[current_category_key]} (key: {current_category_key})")
            print("Press 'q' to stop selection and save")
            
            if early_exit:
                print("Early exit requested. Saving selected samples...")
                break
        if early_exit:
            break
    
    # Organize samples by category
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
    all_selected_set = SampleSet(rgb_path=rgb_path, dsm_path=dsm_path, thermal_path=thermal_path)
    
    # Create individual sets for each category and save them
    for category_name, samples in samples_by_category.items():
        # Create sample set for this category
        sample_set = SampleSet(rgb_path=rgb_path, dsm_path=dsm_path, thermal_path=thermal_path)
        
        # Add samples
        for sample in samples:
            sample_set.add_Sample(sample)
            all_selected_set.add_Sample(sample)
        
        # Calculate features
        if len(sample_set.samples) > 0:
            sample_set.fill_neighbors_all(distance_large=distance_large)
        
        # Create unique filename
        idx = 1
        while os.path.exists(os.path.join(output_path, f"{category_name}_{idx}.json")):
            idx += 1
        json_path = os.path.join(output_path, f"{category_name}_{idx}.json")
        png_path = os.path.join(output_path, f"{category_name}_{idx}.png")
        
        # Visualize and save
        if len(sample_set.samples) > 0:
            plt.figure(figsize=(20, 20))
            sample_set.plot_samples_as_list()
            plt.suptitle(f"{category_name} : {len(sample_set.samples)} samples")
            plt.savefig(png_path)
            plt.close()
            
            # Save JSON
            sample_set.save_samples_to_json(json_path)
            print(f"{len(sample_set.samples)} '{category_name}' samples saved to {json_path}")
    
    # Save/update count file
    txt_path = os.path.join(output_path, "pop_counts.txt")
    counts = {}
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    counts[k.strip()] = int(v.strip())
    
    # Update counts
    for category_name, samples in samples_by_category.items():
        counts[category_name] = counts.get(category_name, 0) + len(samples)
    
    # Write counts
    with open(txt_path, "w") as f:
        for k, v in counts.items():
            f.write(f"{k}: {v}\n")
    
    # Rename selection window
    src_img = os.path.join(output_path, "fenetre_selection.png")
    dst_img = os.path.join(output_path, "last_selection_window.png")
    if os.path.exists(src_img) and src_img != dst_img:
        shutil.copy(src_img, dst_img)
    
    # Summary
    total_samples = sum(len(samples) for samples in samples_by_category.values())
    print(f"\nSelection summary:")
    for category_name, samples in samples_by_category.items():
        print(f"- {category_name}: {len(samples)} samples")
    print(f"Total: {total_samples} samples")
    
    # Clean up
    all_samples_set.clear_rasters()
    all_selected_set.clear_rasters()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Parse class names and create class dictionary
    class_names = [name.strip() for name in args.classes.split(',')]
    
    # Check if we have at least one class
    if len(class_names) < 1:
        print("Error: At least one class name must be provided")
        sys.exit(1)
    
    # Create class dictionary (using first letter of each class name as key)
    class_dict = {}
    for name in class_names:
        # Use first letter as key
        key = name[0].lower()
        # If key already exists, try to find another letter
        if key in class_dict:
            # Look for another letter
            for i in range(1, len(name)):
                alt_key = name[i].lower()
                if alt_key not in class_dict:
                    key = alt_key
                    break
        class_dict[key] = name
    
    print(f"Starting interactive sample selection with {len(class_names)} classes:")
    for key, name in class_dict.items():
        print(f"  - {key}: {name}")
    
    # Calculate total number of samples
    n_samples_x = args.blocks_x * SAMPLES_PER_WINDOW
    n_samples_y = args.blocks_y * SAMPLES_PER_WINDOW
    
    # Convert distances in meters to pixel coordinates
    start_column, start_row = meters_to_pixels(
        args.rgb, 
        args.distance_start_horizontal, 
        args.distance_start_vertical
    )
    
    # Run interactive selection
    pop_selection_adapted(
        x_start=start_column,
        y_start=start_row,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y,
        size_patch=args.sample_size,
        output_path=args.output,
        rgb_path=args.rgb,
        dsm_path=args.dsm,
        thermal_path=args.thermal,
        distance_large=3,
        class_dict=class_dict
    )

if __name__ == "__main__":
    main()

# python interactive_sample_selection.py --rgb path/to/rgb.tif --dsm path/to/dsm.tif 
#                                        --classes "Class1,Class2,Class3" 
#                                        --output path/to/output_dir
#                                        --sample-size 32
#                                        --distance-start-horizontal 0 --distance-start-vertical 0
#                                        --blocks-x 4 --blocks-y 4
"""
python code/final_codes/classification/create_classification_model/interactive_sample_selection.py \
    --rgb drone_treated/WAP32_full_transparent_mosaic_group1.tif \
    --dsm drone_treated/WAP32_full_dsm.tif \
    --classes "Lichen, Green, Trough" \
    --output data/selection_test \
    --sample-size 64 \
    --distance-start-horizontal 214 --distance-start-vertical 418 \
    --blocks-x 2 --blocks-y 2
"""