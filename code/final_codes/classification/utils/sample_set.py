import os
import json
import numpy as np
import matplotlib.pyplot as plt
from .sample import Sample
from .block_rasters_manager import BlockRastersManager

class SampleSet:
    """
    Simplified version of SamplesSet to work with the interactive selection tool.
    """
    def __init__(self, rgb_path=None, dsm_path=None, thermal_path=None, n_samples_x=None, n_samples_y=None):
        self.rgb_path = rgb_path
        self.dsm_path = dsm_path
        self.thermal_path = thermal_path
        self.n_samples_x = n_samples_x
        self.n_samples_y = n_samples_y
        self.samples = {}
        self.block_rasters = None
        
        if rgb_path:
            
            self.block_rasters = BlockRastersManager(
                rgb_path=rgb_path, 
                dsm_path=dsm_path,
                thermal_path=thermal_path
            )
    
    def create_samples_grid(self, x_start=0, y_start=0, size_patch=32):
        """Create a grid of samples."""
        if not self.block_rasters:
            raise ValueError("Rasters not initialized.")
        
        # Create a block covering the entire area
        block = {
            'x_start': x_start,
            'y_start': y_start,
            'x_end': x_start + (self.n_samples_x * size_patch),
            'y_end': y_start + (self.n_samples_y * size_patch)
        }
        
        # Load the block
        self.block_rasters.load_block(block)
        
        # Create samples for each position in the grid
        for i_y in range(self.n_samples_y):
            for i_x in range(self.n_samples_x):
                # Calculate pixel coordinates
                x = i_x * size_patch
                y = i_y * size_patch
                
                # Create a sample
                sample = Sample(
                    i_x=i_x,
                    i_y=i_y,
                    x=x,
                    y=y,
                    size_patch=size_patch,
                    block_rasters=self.block_rasters
                )
                
                # Store the sample
                self.samples[(i_x, i_y)] = sample
    
    def get_samples_matrix(self):
        """Get samples organized in a 2D matrix."""
        if not self.samples:
            return []
        
        # Create empty matrix
        matrix = [[None for _ in range(self.n_samples_x)] for _ in range(self.n_samples_y)]
        
        # Fill matrix with samples
        for (i_x, i_y), sample in self.samples.items():
            if 0 <= i_x < self.n_samples_x and 0 <= i_y < self.n_samples_y:
                matrix[i_y][i_x] = sample
                
        return matrix
    
    def add_Sample(self, sample):
        """Add a sample to the collection."""
        self.samples[(sample.i_x, sample.i_y)] = sample
    
    def fill_neighbors_all(self, distance_large=3):
        """Calculate neighborhood statistics for all samples."""
        for sample in self.samples.values():
            sample.compute_neighbors(self.samples, self.block_rasters, distance_large)
    
    def clear_rasters(self):
        """Clear raster data to free memory."""
        if self.block_rasters:
            self.block_rasters.clear_rasters()
    
    def save_samples_to_json(self, filename):
        """Save samples to a JSON file."""
        data = {
            "n_samples_x": self.n_samples_x,
            "n_samples_y": self.n_samples_y,
            "samples": []
        }
        
        for sample in self.samples.values():
            # Skip samples without a category
            if not hasattr(sample, 'category'):
                continue
                
            # Create sample dictionary
            sample_dict = {
                "i_x": sample.i_x,
                "i_y": sample.i_y,
                "x": sample.x,
                "y": sample.y,
                "size_patch": sample.size_patch,
                "category": sample.category
            }
            
            # Add feature values
            for attr in [
                "r_mean", "g_mean", "b_mean", "r_var", "g_var", "b_var",
                "z_mean", "z_var", "t_mean", "t_var",
                "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean", "z_moins_z_n",
                "r_large_mean", "g_large_mean", "b_large_mean", 
                "t_large_mean", "z_moins_z_large"
            ]:
                if hasattr(sample, attr) and getattr(sample, attr) is not None:
                    sample_dict[attr] = getattr(sample, attr)
            
            data["samples"].append(sample_dict)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        return len(data["samples"])
    
    def plot_samples_as_list(self, max_samples=400, cols=20):
        """Plot samples as a grid of small images."""
        # Filter samples with category
        samples_with_category = [s for s in self.samples.values() if hasattr(s, 'category')]
        
        # Limit samples if needed
        if max_samples and len(samples_with_category) > max_samples:
            samples_with_category = samples_with_category[:max_samples]
        
        # Calculate grid dimensions
        n_samples = len(samples_with_category)
        if n_samples == 0:
            print("No samples with categories to plot")
            return
            
        rows = (n_samples + cols - 1) // cols
        
        # Create the figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each sample
        for i, sample in enumerate(samples_with_category):
            if i >= len(axes):
                break
                
            # Get RGB data for the sample
            r, g, b, _, _ = sample.get_RGBZT()
            if r is not None and g is not None and b is not None:
                rgb = np.dstack((r, g, b))
                axes[i].imshow(rgb)
            
            # Turn off axis
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
    
    @staticmethod
    def load_samples_from_json(filename):
        """
        Load samples from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            SampleSet with samples from the JSON file
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create a new SampleSet without raster paths
        sample_set = SampleSet(
            n_samples_x=data.get("n_samples_x"),
            n_samples_y=data.get("n_samples_y")
        )
        
        # Add each sample from the data
        for s in data.get("samples", []):
            # Create a sample without block_rasters
            sample = Sample(
                i_x=s.get("i_x"),
                i_y=s.get("i_y"),
                x=s.get("x"),
                y=s.get("y"),
                size_patch=s.get("size_patch"),
                block_rasters=None  # No block rasters when loading from JSON
            )
            
            # Add category attribute
            if "category" in s:
                sample.category = s.get("category")
            
            # Add all available attributes directly (bypassing calculation)
            for attr in [
                "r_mean", "g_mean", "b_mean", "r_var", "g_var", "b_var",
                "z_mean", "z_var", "t_mean", "t_var",
                "r_n_mean", "g_n_mean", "b_n_mean", "t_n_mean", "z_moins_z_n",
                "r_large_mean", "g_large_mean", "b_large_mean", 
                "t_large_mean", "z_moins_z_large"
            ]:
                if attr in s:
                    setattr(sample, attr, s.get(attr))
            
            # Add the sample to the set
            sample_set.add_Sample(sample)
        
        return sample_set
