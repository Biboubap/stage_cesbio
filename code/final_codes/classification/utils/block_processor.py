import numpy as np
import logging
from collections import defaultdict

# Use relative imports for sibling modules in the same directory
from .block_rasters_manager import BlockRastersManager
from .block_sample import BlockSample

logger = logging.getLogger(__name__)

def extract_features(samples_dict, feature_names_ref=None):
    """
    Extract features from samples for model prediction.
    
    Args:
        samples_dict: Dictionary of samples indexed by position
        feature_names_ref: Optional list of feature names to include
        
    Returns:
        Tuple of (features array, positions list, feature names list)
    """
    # Extract positions and prepare feature array
    positions = list(samples_dict.keys())
    n_samples = len(positions)
    
    if n_samples == 0:
        return np.array([]), [], []
    
    # Get first sample to determine available features
    first_sample = next(iter(samples_dict.values()))
    
    # Get available feature names
    feature_names = []
    for attr in [
        'r_mean', 'g_mean', 'b_mean', 't_mean', 'z_mean',
        'r_var', 'g_var', 'b_var', 't_var', 'z_var',
        'r_n_mean', 'g_n_mean', 'b_n_mean', 't_n_mean', 'z_moins_z_n',
        'r_large_mean', 'g_large_mean', 'b_large_mean', 't_large_mean', 'z_moins_z_large'
    ]:
        if hasattr(first_sample, attr) and getattr(first_sample, attr) is not None:
            feature_names.append(attr)
    
    # Filter feature names if reference list is provided
    if feature_names_ref is not None:
        feature_names = [name for name in feature_names if name in feature_names_ref]
    
    # Create feature array
    n_features = len(feature_names)
    features = np.zeros((n_samples, n_features))
    
    # Fill feature array
    for i, (pos, sample) in enumerate(samples_dict.items()):
        for j, feat_name in enumerate(feature_names):
            features[i, j] = getattr(sample, feat_name, 0)
    
    return features, positions, feature_names

def filter_features(features, all_feature_names, required_feature_names):
    """
    Filter features to include only those required by the model.
    
    Args:
        features: Feature array
        all_feature_names: List of all feature names
        required_feature_names: List of feature names required by the model
        
    Returns:
        Filtered feature array
    """
    indices = [all_feature_names.index(name) for name in required_feature_names 
               if name in all_feature_names]
    
    if len(indices) != len(required_feature_names):
        missing = [name for name in required_feature_names if name not in all_feature_names]
        logger.warning(f"Missing features: {missing}")
    
    return features[:, indices]

def merge_predictions(pred_map1, pred_map2):
    """
    Merge predictions from the two models according to specified rules.
    
    Args:
        pred_map1: Predictions from the no_chicoutai model
        pred_map2: Predictions from the 16_7 model
        
    Returns:
        Merged prediction map
    """
    # Create the merged prediction map
    n_samples_y, n_samples_x = pred_map1.shape
    merged_map = np.zeros((n_samples_y, n_samples_x), dtype=np.uint8)
    
    # Process all samples
    for i_y in range(n_samples_y):
        for i_x in range(n_samples_x):
            val1 = pred_map1[i_y, i_x]
            val2 = pred_map2[i_y, i_x]
            
            # Special cases for 0 and 255 (No Data)
            if val1 == 0 or val1 == 255:
                merged_map[i_y, i_x] = val1
                continue
                
            # For classes 1 (Chicoutai) and 4 (Lichen) from no_chicoutai model,
            # use the predictions from the 16_7 model
            if val1 == 1 or val1 == 4:  # Chicoutai or Lichen
                # Map 16_7 model classes
                if val2 == 8:  # peat_pure_lichen
                    merged_map[i_y, i_x] = 1  # Pure_Lichen
                elif val2 == 6:  # peat_degraded_lichen
                    merged_map[i_y, i_x] = 2  # Degraded_Lichen
                elif val2 in [2, 7, 9]:  # depression_green, peat_green, through_green
                    merged_map[i_y, i_x] = 3  # Green
                elif val2 in [4, 10]:  # depression_sphagnum, through_sphagnum
                    merged_map[i_y, i_x] = 4  # Sphagnum
                elif val2 in [1, 3]:  # depression_fen, depression_peat
                    merged_map[i_y, i_x] = 5  # Depression
                elif val2 == 5:  # depression_water
                    merged_map[i_y, i_x] = 6  # Water
                else:
                    merged_map[i_y, i_x] = 0  # Default to No Data
            else:
                # For other classes from no_chicoutai model, map them directly
                if val1 == 3:  # green_depression
                    merged_map[i_y, i_x] = 3  # Green
                elif val1 == 5:  # sphaignes
                    merged_map[i_y, i_x] = 4  # Sphagnum
                elif val1 in [2, 7]:  # dry_depression, black_depression
                    merged_map[i_y, i_x] = 5  # Depression
                elif val1 == 6:  # watered_depression
                    merged_map[i_y, i_x] = 6  # Water
                else:
                    merged_map[i_y, i_x] = 0  # Default to No Data
    
    return merged_map

def filter_isolated_samples(pred_map, min_group_size=3):
    """
    Filter out isolated samples to reduce noise in the classification.
    
    Args:
        pred_map: Prediction map
        min_group_size: Minimum size of a group to keep
        
    Returns:
        Filtered prediction map
    """
    pred_map_filtered = pred_map.copy()
    height, width = pred_map.shape
    visited = np.zeros_like(pred_map, dtype=bool)
    
    # Perform connected component labeling and filtering
    for y in range(height):
        for x in range(width):
            if visited[y, x] or pred_map[y, x] == 0:
                continue
                
            # BFS to find connected components
            class_value = pred_map[y, x]
            group = [(y, x)]
            group_pixels = [(y, x)]
            visited[y, x] = True
            
            i = 0
            while i < len(group):
                cy, cx = group[i]
                i += 1
                
                # Check neighbors (4-connectivity)
                for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
                    if (0 <= ny < height and 0 <= nx < width and
                        not visited[ny, nx] and pred_map[ny, nx] == class_value):
                        group.append((ny, nx))
                        group_pixels.append((ny, nx))
                        visited[ny, nx] = True
            
            # If group is too small, set its pixels to the majority class of neighbors
            if len(group_pixels) < min_group_size:
                for py, px in group_pixels:
                    # Collect neighbor classes (including diagonals)
                    neighbor_classes = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = py + dy, px + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if pred_map[ny, nx] != class_value and pred_map[ny, nx] != 0:
                                    neighbor_classes.append(pred_map[ny, nx])
                    
                    # Set to most common neighbor class if any exist, otherwise keep
                    if neighbor_classes:
                        class_counts = defaultdict(int)
                        for cls in neighbor_classes:
                            class_counts[cls] += 1
                        most_common = max(class_counts.items(), key=lambda x: x[1])[0]
                        pred_map_filtered[py, px] = most_common
    
    return pred_map_filtered

def predict_samples(model, features, positions, shape_y, shape_x):
    """
    Predict classes using the provided model and features.
    
    Args:
        model: Trained model
        features: Feature array
        positions: List of (i_x, i_y) positions
        shape_y, shape_x: Shape of the output prediction map
        
    Returns:
        Prediction map as numpy array
    """
    # Predict classes
    predictions = model.predict(features)
    
    # Create prediction map
    pred_map = np.zeros((shape_y, shape_x), dtype=np.uint8)
    
    # Check if predictions are strings and map them to integers if necessary
    if len(predictions) > 0 and isinstance(predictions[0], str):
        # Define mapping based on expected values in merge_predictions
        label_map = {
            'chicoutai': 1,           # Chicoutai
            'dry_depression': 2,      # dry_depression
            'green_depression': 3,    # green_depression
            'lichen': 4,              # Lichen
            'sphaignes': 5,           # sphaignes
            'watered_depression': 6,  # watered_depression
            'black_depression': 7,    # black_depression
            # Add other string labels that might be encountered
            'peat_pure_lichen': 8,    # peat_pure_lichen
            'peat_degraded_lichen': 6, # peat_degraded_lichen
            'depression_green': 2,    # depression_green
            'peat_green': 7,          # peat_green
            'through_green': 9,       # through_green
            'depression_sphagnum': 4, # depression_sphagnum
            'through_sphagnum': 10,   # through_sphagnum
            'depression_fen': 1,      # depression_fen
            'depression_peat': 3,     # depression_peat
            'depression_water': 5,    # depression_water
        }
        
        # Fill prediction map using the mapping
        for i, (i_x, i_y) in enumerate(positions):
            if predictions[i] in label_map:
                pred_map[i_y, i_x] = label_map[predictions[i]]
            else:
                # Use a default value for unknown labels
                logger.warning(f"Unknown class label: {predictions[i]}, using default value 0")
                pred_map[i_y, i_x] = 0
    else:
        # Fill prediction map directly with numeric predictions
        for i, (i_x, i_y) in enumerate(positions):
            pred_map[i_y, i_x] = predictions[i]
    
    return pred_map

def process_block_with_overlap(rgb_path, dsm_path, model1, model2, 
                             feature_names_model1, feature_names_model2,
                             block, patch_size, block_index=0):
    """
    Process a single block of raster data with overlap handling.
    
    Args:
        rgb_path: Path to RGB raster
        dsm_path: Path to DSM raster
        model1: First classification model
        model2: Second classification model
        feature_names_model1: Features required by model1
        feature_names_model2: Features required by model2
        block: Dictionary with block coordinates and valid region
        patch_size: Size of patches for classification
        block_index: Index of the current block for logging
        
    Returns:
        Tuple of (classification array, block info)
    """
    try:
        logger.info(f"Processing block {block_index}: {block['x_start']}:{block['x_end']}, {block['y_start']}:{block['y_end']}")
        
        # Initialize BlockRastersManager for this block
        block_rasters = BlockRastersManager(rgb_path=rgb_path, dsm_path=dsm_path)
        block_rasters.load_block(block)
        
        # Get block dimensions
        block_height, block_width = block_rasters.get_block_shape()
        
        # Calculate number of samples in each dimension
        n_samples_y = block_height // patch_size
        n_samples_x = block_width // patch_size
        
        logger.info(f"Block has dimensions {block_height}x{block_width}, creating {n_samples_y}x{n_samples_x} samples")
        
        # Create samples for this block
        samples = {}
        nan_positions = []
        
        for i_y in range(n_samples_y):
            for i_x in range(n_samples_x):
                # Calculate coordinates relative to the block
                y = i_y * patch_size
                x = i_x * patch_size
                
                # Skip if patch extends beyond block
                if y + patch_size > block_height or x + patch_size > block_width:
                    continue
                
                # Create sample
                sample = BlockSample(
                    i_x=i_x, 
                    i_y=i_y, 
                    x=x, 
                    y=y, 
                    size_patch=patch_size,
                    block_rasters=block_rasters
                )
                
                # Check for NaN values
                if (sample.r_mean is None or np.isnan(sample.r_mean) or
                    sample.g_mean is None or np.isnan(sample.g_mean) or
                    sample.b_mean is None or np.isnan(sample.b_mean)):
                    nan_positions.append((i_x, i_y))
                    # Fix NaN values
                    sample.r_mean = 0.0 if sample.r_mean is None or np.isnan(sample.r_mean) else sample.r_mean
                    sample.g_mean = 0.0 if sample.g_mean is None or np.isnan(sample.g_mean) else sample.g_mean
                    sample.b_mean = 0.0 if sample.b_mean is None or np.isnan(sample.b_mean) else sample.b_mean
                    sample.r_var = 0.0 if sample.r_var is None or np.isnan(sample.r_var) else sample.r_var
                    sample.g_var = 0.0 if sample.g_var is None or np.isnan(sample.g_var) else sample.g_var
                    sample.b_var = 0.0 if sample.b_var is None or np.isnan(sample.b_var) else sample.b_var
                    sample.z_mean = 0.0 if sample.z_mean is None or np.isnan(sample.z_mean) else sample.z_mean
                    sample.z_var = 0.0 if sample.z_var is None or np.isnan(sample.z_var) else sample.z_var
                
                # Store sample
                samples[(i_x, i_y)] = sample
        
        # Compute neighborhood statistics
        logger.info(f"Computing neighborhood statistics for {len(samples)} samples")
        for sample in samples.values():
            sample.compute_neighbors(samples, block_rasters, distance_large=3)
        
        # Extract features
        logger.info("Extracting features")
        features, positions, all_feature_names = extract_features(samples)
        
        if len(features) == 0:
            logger.warning("No valid samples found in this block")
            return None, block
        
        # Filter features for each model
        features_model1 = filter_features(features, all_feature_names, feature_names_model1)
        features_model2 = filter_features(features, all_feature_names, feature_names_model2)
        
        # Predict with both models
        logger.info("Running predictions with both models")
        pred_map1 = predict_samples(model1, features_model1, positions, n_samples_y, n_samples_x)
        pred_map2 = predict_samples(model2, features_model2, positions, n_samples_y, n_samples_x)
        
        # Mark transparent (NaN) areas with class 0
        for i_x, i_y in nan_positions:
            pred_map1[i_y, i_x] = 0
            pred_map2[i_y, i_x] = 0
        
        # Merge predictions
        logger.info("Merging predictions")
        merged_map = merge_predictions(pred_map1, pred_map2)
        
        # Apply filter to remove isolated samples
        logger.info("Filtering isolated samples")
        merged_map_filtered = filter_isolated_samples(merged_map)
        
        # Upsample to full resolution
        logger.info("Upsampling to full resolution")
        full_res_map = np.zeros((block_height, block_width), dtype=np.uint8)
        
        for i_y in range(n_samples_y):
            for i_x in range(n_samples_x):
                if i_y < merged_map_filtered.shape[0] and i_x < merged_map_filtered.shape[1]:
                    y_start = i_y * patch_size
                    x_start = i_x * patch_size
                    y_end = min(y_start + patch_size, block_height)
                    x_end = min(x_start + patch_size, block_width)
                    
                    full_res_map[y_start:y_end, x_start:x_end] = merged_map_filtered[i_y, i_x]
        
        # Clean up resources
        block_rasters.clear_rasters()
        
        return full_res_map, block
        
    except Exception as e:
        logger.error(f"Error processing block {block_index}: {e}")
        import traceback
        traceback.print_exc()
        return None, block
        return None, block
