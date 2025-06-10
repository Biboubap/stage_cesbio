import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from osgeo import gdal
import glob
import os
import joblib
from scipy.stats import pearsonr

def load_all_sentinel_features(indices_dir, bands_dir):
    """
    Load all Sentinel bands and indices as features
    
    Args:
        indices_dir: Directory containing indices TIF files
        bands_dir: Directory containing bands TIF files
    
    Returns:
        features: NumPy array of shape (n_features, rows, cols)
        band_names: List of feature names
    """
    # List all tif files
    band_files = sorted(glob.glob(os.path.join(bands_dir, "*.tif")))
    index_files = sorted(glob.glob(os.path.join(indices_dir, "*.tif")))
    
    print(f"Found {len(band_files)} band files and {len(index_files)} index files")
    
    bands = []
    band_names = []
    
    # Load band files
    for tif in band_files:
        ds = gdal.Open(tif)
        if ds is None:
            print(f"Warning: Could not open {tif}")
            continue
        arr = ds.GetRasterBand(1).ReadAsArray()
        bands.append(arr)
        file_name = os.path.splitext(os.path.basename(tif))[0]
        band_name = file_name.replace('mediane_clipped_STACK_2023_Band', '').replace('_WAP32', '')
        band_names.append(f"band_{band_name}")
    
    # Load index files
    for tif in index_files:
        ds = gdal.Open(tif)
        if ds is None:
            print(f"Warning: Could not open {tif}")
            continue
        arr = ds.GetRasterBand(1).ReadAsArray()
        bands.append(arr)
        file_name = os.path.splitext(os.path.basename(tif))[0]
        index_name = file_name.replace('mediane_clipped_', '').replace('_WAP32', '')
        band_names.append(f"index_{index_name}")
    
    features = np.stack(bands, axis=0)  # (n_features, rows, cols)
    return features, band_names

def prepare_data_for_regression(csv_path, sentinel_features, feature_names, use_sqrt=True):
    """
    Prepare data for regression by extracting features and targets from the CSV
    
    Args:
        csv_path: Path to the CSV file with class proportions
        sentinel_features: NumPy array of shape (n_features, rows, cols)
        feature_names: List of feature names
        use_sqrt: Whether to use sqrt-transformed data for certain classes
    
    Returns:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of class names
    """
    # Load CSV with class proportions
    df = pd.read_csv(csv_path)
    
    # Find class columns (exclude metadata columns and the 'none' class)
    metadata_cols = ['col_s', 'row_s', 'valid_pixels', 'total_pixels', 'valid_proportion']
    
    # Get all regular class columns (not starting with sqrt_, band_, index_)
    regular_class_cols = [col for col in df.columns if col not in metadata_cols 
                 and not col.startswith('band_') and not col.startswith('index_')
                 and not col.startswith('sqrt_') and col != 'none']
    
    # Only use sqrt-transformed classes if requested
    sqrt_classes = ["dry_depression", "sphaignes", "black_depression"]
    
    # Determine which classes to use
    class_cols = []
    for col in regular_class_cols:
        if use_sqrt and col in sqrt_classes and f"sqrt_{col}" in df.columns:
            class_cols.append(f"sqrt_{col}")
        else:
            class_cols.append(col)
    
    # Add through_proportion or sqrt_through_proportion based on use_sqrt flag
    if "through_proportion" in df.columns:
        if use_sqrt and "sqrt_through_proportion" in df.columns:
            class_cols.append("sqrt_through_proportion")
        else:
            class_cols.append("through_proportion")
    
    print(f"Found {len(class_cols)} classes: {class_cols}")
    
    # Make sure we have consistent data lengths (needed for both sqrt and non-sqrt cases)
    # Get the first row to determine the expected length
    first_col = class_cols[0]
    expected_length = len(df[first_col])
    
    # Prepare feature and target arrays
    X = []
    y_dict = {class_name: [] for class_name in class_cols}
    
    # For each pixel in the CSV
    for _, row in df.iterrows():
        col_s = int(row["col_s"])
        row_s = int(row["row_s"])
        
        # Extract features for this pixel
        if (0 <= row_s < sentinel_features.shape[1] and 
            0 <= col_s < sentinel_features.shape[2]):
            pixel_features = sentinel_features[:, row_s, col_s]
            X.append(pixel_features)
            
            # Extract target values (class proportions)
            for class_name in class_cols:
                y_dict[class_name].append(row[class_name])
    
    X = np.array(X)
    for class_name in class_cols:
        y_dict[class_name] = np.array(y_dict[class_name])
        
        # Verify that all arrays have the same length
        if len(y_dict[class_name]) != len(X):
            print(f"Warning: Length mismatch for {class_name}: expected {len(X)}, got {len(y_dict[class_name])}")
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y_dict, class_cols

def group_classes(y_dict, class_names):
    """
    Group classes into 3 categories:
    1. [Lichen]
    2. [Chicoutai, Green Depression]
    3. [Through proportion - using sqrt_through_proportion or through_proportion directly]
    
    Args:
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of class names
    
    Returns:
        y_grouped: Dictionary with arrays of grouped targets
        group_names: List of group names
    """
    # Define groups (excluding 'none')
    group1 = ["lichen"]
    group2 = ["chicoutai", "green_depression"]
    
    # Initialize arrays for groups
    n_samples = len(y_dict[list(y_dict.keys())[0]])
    y_group1 = np.zeros(n_samples)
    y_group2 = np.zeros(n_samples)
    
    # Sum proportions for each group
    for class_name in class_names:
        if class_name in group1:
            y_group1 += y_dict[class_name]
        elif class_name in group2:
            y_group2 += y_dict[class_name]
    
    # For group 3, check what's available in y_dict and use the appropriate column
    if "sqrt_through_proportion" in y_dict:
        y_group3 = y_dict["sqrt_through_proportion"]
        group3_name = "group3_sqrt_through_proportion"
    elif "through_proportion" in y_dict:
        y_group3 = y_dict["through_proportion"]
        group3_name = "group3_through_proportion"
    else:
        # Fallback to calculating from individual components
        y_group3 = np.zeros(n_samples)
        sqrt_used = False
        
        # Check if we're using sqrt-transformed classes
        sqrt_classes = [name for name in class_names if name.startswith('sqrt_') and 
                      name.replace('sqrt_', '') in ["dry_depression", "sphaignes", "black_depression"]]
        
        if sqrt_classes:
            # Using sqrt-transformed classes
            for class_name in sqrt_classes:
                y_group3 += y_dict[class_name]
            group3_name = "group3_sqrt_through_proportion"
        else:
            # Using regular classes
            regular_classes = ["dry_depression", "sphaignes", "black_depression"]
            for class_name in regular_classes:
                if class_name in class_names:
                    y_group3 += y_dict[class_name]
            group3_name = "group3_through_proportion"
    
    # Create dictionary for grouped targets
    y_grouped = {
        "group1_lichen": y_group1,
        "group2_chicoutai_green": y_group2,
        group3_name: y_group3
    }
    
    group_names = list(y_grouped.keys())
    
    return y_grouped, group_names

def train_multivariate_rf(X, y_dict, class_names, test_size=0.3, random_state=42):
    """
    Train a RandomForestRegressor for each class
    
    Args:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of target class names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        models: Dictionary of trained models for each class
        X_test: Test features
        y_test_dict: Dictionary with test targets for each class
    """
    # Split data into train and test sets
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    
    # Use the same train/test split for all classes
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    X_train, X_test = X[train_idx], X[test_idx]
    for class_name in class_names:
        y_train_dict[class_name] = y_dict[class_name][train_idx]
        y_test_dict[class_name] = y_dict[class_name][test_idx]
    
    # Train a model for each class
    models = {}
    for class_name in class_names:
        print(f"Training model for {class_name}...")
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=random_state,
            max_depth=None,
            max_features="sqrt"
        )
        rf.fit(X_train, y_train_dict[class_name])
        models[class_name] = rf
    
    return models, X_test, y_test_dict

def train_histgb_rf(X, y_dict, class_names, test_size=0.3, random_state=42):
    """
    Train a HistGradientBoostingRegressor for each class
    
    Args:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of target class names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        models: Dictionary of trained models for each class
        X_test: Test features
        y_test_dict: Dictionary with test targets for each class
    """
    # Split data into train and test sets
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    
    # Use the same train/test split for all classes
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    X_train, X_test = X[train_idx], X[test_idx]
    for class_name in class_names:
        y_train_dict[class_name] = y_dict[class_name][train_idx]
        y_test_dict[class_name] = y_dict[class_name][test_idx]
    
    # Train a model for each class
    models = {}
    for class_name in class_names:
        print(f"Training HistGradientBoostingRegressor for {class_name}...")
        gb = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.1,
            max_depth=None,  # Auto-determined
            min_samples_leaf=20,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        gb.fit(X_train, y_train_dict[class_name])
        models[class_name] = gb
    
    return models, X_test, y_test_dict

def train_multioutput_rf(X, y_dict, class_names, test_size=0.3, random_state=42):
    """
    Train a multi-output RandomForest model that predicts all classes at once
    
    Args:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of target class names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        model: The trained multi-output model
        X_test: Test features
        y_test: Test targets matrix
        y_test_dict: Dictionary of test targets by class name
    """
    # Combine all target variables into a single matrix
    y_matrix = np.column_stack([y_dict[class_name] for class_name in class_names])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_matrix, test_size=test_size, random_state=random_state
    )
    
    # Create and train multi-output model directly using RandomForestRegressor
    # which can handle multivariate outputs natively
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        random_state=random_state,
        max_depth=30,
        max_features="sqrt",
        n_jobs=-1  # Use all available cores for faster training
    )
    
    print("Training multi-output Random Forest model (direct method)...")
    model.fit(X_train, y_train)
    
    # Create y_test_dict for evaluation
    y_test_dict = {}
    for i, class_name in enumerate(class_names):
        y_test_dict[class_name] = y_test[:, i]
    
    return model, X_test, y_test, y_test_dict

def train_multioutput_histgb(X, y_dict, class_names, test_size=0.3, random_state=42):
    """
    Train a multi-output HistGradientBoostingRegressor model that predicts all classes at once
    
    Args:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of target class names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        model: The trained multi-output model
        X_test: Test features
        y_test: Test targets matrix
        y_test_dict: Dictionary of test targets by class name
    """
    from sklearn.multioutput import MultiOutputRegressor
    
    # Combine all target variables into a single matrix
    y_matrix = np.column_stack([y_dict[class_name] for class_name in class_names])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_matrix, test_size=test_size, random_state=random_state
    )
    
    # HistGradientBoostingRegressor doesn't natively support multi-output
    # So we use MultiOutputRegressor as a wrapper
    base_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.1,
        max_depth=None,  # Auto-determined
        min_samples_leaf=20,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    
    model = MultiOutputRegressor(base_model, n_jobs=-1)  # Parallelize training
    print("Training multi-output HistGradientBoostingRegressor model...")
    model.fit(X_train, y_train)
    
    # Create y_test_dict for evaluation
    y_test_dict = {}
    for i, class_name in enumerate(class_names):
        y_test_dict[class_name] = y_test[:, i]
    
    return model, X_test, y_test, y_test_dict

def evaluate_multivariate_rf(models, X_test, y_test_dict, class_names, output_path, feature_names=None):
    """
    Evaluate the performance of the multivariate RF models and create plots
    
    Args:
        models: Dictionary of trained models for each class
        X_test: Test features
        y_test_dict: Dictionary with test targets for each class
        class_names: List of target class names
        output_path: Path to save the evaluation plots
        feature_names: List of feature names used for training
    
    Returns:
        metrics: Dictionary with performance metrics for each class
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Create a figure for the scatter plots
    n_cols = min(3, len(class_names))
    n_rows = (len(class_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    
    # If only one subplot, axes needs to be in a 2D array
    if len(class_names) == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Evaluate each model
    for i, class_name in enumerate(class_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Make predictions
        model = models[class_name]
        y_pred = model.predict(X_test)
        y_true = y_test_dict[class_name]
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pearson_coef, _ = pearsonr(y_true, y_pred)
        
        metrics[class_name] = {
            "r2": r2,
            "rmse": rmse,
            "pearson": pearson_coef
        }
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Customize labels based on whether this is a sqrt-transformed class
        if class_name.startswith('sqrt_') or 'sqrt' in class_name:
            ax.set_xlabel("Actual sqrt(proportion)")
            ax.set_ylabel("Predicted sqrt(proportion)")
            ax.set_title(f"{class_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        else:
            ax.set_xlabel("Actual proportion")
            ax.set_ylabel("Predicted proportion")
            ax.set_title(f"{class_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
    
    # Hide empty subplots
    for i in range(len(class_names), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Evaluation plot saved to {output_path}")
    
    # Also create a feature importance plot for all models
    if feature_names is not None:
        plot_feature_importance(models, class_names, output_path.replace('.png', '_feature_importance.png'), feature_names)
    
    return metrics

def plot_feature_importance(models, class_names, output_path, feature_names, top_n=20):
    """
    Plot feature importance for all models
    
    Args:
        models: Dictionary of trained models
        class_names: List of class names
        output_path: Path to save the plot
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    # Compute average importance across all models
    avg_importance = np.zeros(len(feature_names))
    
    for class_name in class_names:
        model = models[class_name]
        # Handle different model types that store feature importance differently
        if hasattr(model, 'feature_importances_'):
            # RandomForestRegressor
            avg_importance += model.feature_importances_
        elif hasattr(model, '_final_estimator') and hasattr(model._final_estimator, 'feature_importances_'):
            # MultiOutputRegressor with RandomForestRegressor
            avg_importance += model._final_estimator.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            # HistGradientBoostingRegressor
            importance = model.get_feature_importance()
            avg_importance += importance / importance.sum()  # Normalize to sum to 1 like feature_importances_
        else:
            print(f"Warning: Model for {class_name} doesn't have recognized feature importance attribute")
            continue
    
    avg_importance /= len(class_names)
    
    # Sort by average importance
    indices = np.argsort(avg_importance)[::-1]
    
    # Limit to top_n features
    if len(indices) > top_n:
        indices = indices[:top_n]
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create bars for each class + average
    bar_width = 0.8 / (len(class_names) + 1)
    x = np.arange(len(indices))
    
    # Plot bars for each class
    for i, class_name in enumerate(class_names):
        model = models[class_name]
        
        # Get importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, '_final_estimator') and hasattr(model._final_estimator, 'feature_importances_'):
            importance = model._final_estimator.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            raw_importance = model.get_feature_importance()
            importance = raw_importance / raw_importance.sum()  # Normalize
        else:
            continue  # Skip this model
            
        plt.bar(x + i * bar_width, importance[indices], bar_width, alpha=0.7, label=class_name)
    
    # Plot average importance
    plt.bar(x + len(class_names) * bar_width, avg_importance[indices], bar_width, 
            color='black', alpha=0.7, label='Average')
    
    # Add labels and legend
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance by Class')
    plt.xticks(x + bar_width * (len(class_names) / 2), [feature_names[i] for i in indices], rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")

def save_rf_models(models, feature_names, class_names, output_path):
    """
    Save the trained RF models
    
    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
        class_names: List of class names
        output_path: Path to save the models
    """
    model_data = {
        "models": models,
        "feature_names": feature_names,
        "class_names": class_names
    }
    joblib.dump(model_data, output_path)
    print(f"Models saved to {output_path}")

def predict_proportions_from_rasters(sentinel_bands_dir, sentinel_indices_dir, model_path, output_dir):
    """
    Apply the trained models to new rasters and save the predicted proportions
    
    Args:
        sentinel_bands_dir: Directory containing Sentinel-2 band rasters
        sentinel_indices_dir: Directory containing Sentinel-2 indices rasters
        model_path: Path to the saved models
        output_dir: Directory to save the predicted proportion rasters
    """
    # Load models
    model_data = joblib.load(model_path)
    models = model_data["models"]
    class_names = model_data["class_names"]
    
    # Load features
    features, feature_names = load_all_sentinel_features(sentinel_indices_dir, sentinel_bands_dir)
    
    # Get raster dimensions and geo information from one of the band files
    band_files = glob.glob(os.path.join(sentinel_bands_dir, "*.tif"))
    if not band_files:
        print("No band files found")
        return
    
    reference_ds = gdal.Open(band_files[0])
    if reference_ds is None:
        print(f"Could not open reference file: {band_files[0]}")
        return
    
    width = reference_ds.RasterXSize
    height = reference_ds.RasterYSize
    geo_transform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For each class, create a prediction raster
    for class_name in class_names:
        # Create output raster
        output_path = os.path.join(output_dir, f"proportion_{class_name}.tif")
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)
        
        # Allocate output array
        output_array = np.zeros((height, width), dtype=np.float32)
        
        # Reshape features for prediction
        n_features = features.shape[0]
        X_flat = features.reshape(n_features, -1).T  # Shape: (n_pixels, n_features)
        
        # Predict in batches to avoid memory issues
        batch_size = 10000
        n_batches = (X_flat.shape[0] + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_flat.shape[0])
            X_batch = X_flat[start_idx:end_idx]
            
            # Predict
            y_pred_batch = models[class_name].predict(X_batch)
            
            # Reshape to 2D
            row_indices = (start_idx // width, end_idx // width)
            for j, pred in enumerate(y_pred_batch):
                pixel_idx = start_idx + j
                row = pixel_idx // width
                col = pixel_idx % width
                output_array[row, col] = pred
        
        # Write to raster
        out_ds.GetRasterBand(1).WriteArray(output_array)
        out_ds.FlushCache()
        out_ds = None
        
        print(f"Proportion prediction for {class_name} saved to {output_path}")

def run_multivariate_regression(data_dir, output_dir, wap_number=32, use_sqrt=True, use_peat=False):
    """
    Run the full multivariate regression workflow
    
    Args:
        data_dir: Directory containing the input data
        output_dir: Directory to save the output
        wap_number: WAP site number
        use_sqrt: Whether to use sqrt-transformed values for certain classes
        use_peat: Whether to include peat data in the analysis
    """
    print("Starting multivariate regression analysis...")
    print(f"Using sqrt transformation for certain classes: {use_sqrt}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    peat_suffix = "_peat" if use_peat else ""
    
    sentinel_bands_dir = f"DataCubeS2/BandsS22023_WAP{wap_number}{peat_suffix}/mediane"
    sentinel_indices_dir = f"DataCubeS2/IndicesS22023_WAP{wap_number}{peat_suffix}/mediane"
    csv_path = os.path.join(data_dir, f"class_proportions_WAP{wap_number}_filtered.csv")
    
    # If not using sqrt but comparing, make output filenames reflect this
    suffix = "_sqrt" if use_sqrt else "_raw"
    
    # 1. Load Sentinel features
    print("\n1. Loading Sentinel features...")
    sentinel_features, feature_names = load_all_sentinel_features(sentinel_indices_dir, sentinel_bands_dir)
    
    # 2. Prepare data for regression
    print("\n2. Preparing data for regression...")
    X, y_dict, class_names = prepare_data_for_regression(csv_path, sentinel_features, feature_names, use_sqrt=use_sqrt)
    
    # 3. Train and evaluate individual RF models for all classes
    print("\n3. Training and evaluating individual RandomForest models...")
    rf_models, X_test_rf, y_test_dict_rf = train_multivariate_rf(X, y_dict, class_names)
    rf_metrics = evaluate_multivariate_rf(
        rf_models, X_test_rf, y_test_dict_rf, class_names,
        output_path=os.path.join(output_dir, f"all_classes_rf_regression{suffix}.png"),
        feature_names=feature_names
    )
    save_rf_models(
        rf_models, feature_names, class_names,
        output_path=os.path.join(output_dir, f"all_classes_rf_models{suffix}.joblib")
    )
    
    # 4. Train and evaluate multi-output RF model
    print("\n4. Training and evaluating multi-output RandomForest model...")
    rf_multioutput_model, X_test_mo_rf, y_test_mo_rf, y_test_dict_mo_rf = train_multioutput_rf(X, y_dict, class_names)
    rf_mo_metrics = evaluate_multioutput_rf(
        rf_multioutput_model, X_test_mo_rf, y_test_mo_rf, y_test_dict_mo_rf, class_names,
        output_path=os.path.join(output_dir, f"all_classes_rf_multioutput_regression{suffix}.png"),
        feature_names=feature_names
    )
    
    # Save multi-output RF model
    joblib.dump({
        "model": rf_multioutput_model,
        "feature_names": feature_names,
        "class_names": class_names
    }, os.path.join(output_dir, "rf_multioutput_model.joblib"))
    
    # 5. Train and evaluate individual HistGradientBoostingRegressor models
    print("\n5. Training and evaluating individual HistGradientBoostingRegressor models...")
    histgb_models, X_test_histgb, y_test_dict_histgb = train_histgb_rf(X, y_dict, class_names)
    histgb_metrics = evaluate_multivariate_rf(  # Reuse the same evaluation function
        histgb_models, X_test_histgb, y_test_dict_histgb, class_names,
        output_path=os.path.join(output_dir, f"all_classes_histgb_regression{suffix}.png"),
        feature_names=feature_names
    )
    save_rf_models(  # Reuse the same saving function
        histgb_models, feature_names, class_names,
        output_path=os.path.join(output_dir, f"all_classes_histgb_models{suffix}.joblib")
    )
    
    # 6. Train and evaluate multi-output HistGradientBoostingRegressor model
    print("\n6. Training and evaluating multi-output HistGradientBoostingRegressor model...")
    histgb_multioutput_model, X_test_mo_histgb, y_test_mo_histgb, y_test_dict_mo_histgb = train_multioutput_histgb(X, y_dict, class_names)
    histgb_mo_metrics = evaluate_multioutput_rf(  # Reuse the same evaluation function
        histgb_multioutput_model, X_test_mo_histgb, y_test_mo_histgb, y_test_dict_mo_histgb, class_names,
        output_path=os.path.join(output_dir, f"all_classes_histgb_multioutput_regression{suffix}.png"),
        feature_names=feature_names
    )
    
    # Save multi-output HistGB model
    joblib.dump({
        "model": histgb_multioutput_model,
        "feature_names": feature_names,
        "class_names": class_names
    }, os.path.join(output_dir, "histgb_multioutput_model.joblib"))
    
    # 7. Group classes
    print("\n7. Processing grouped classes...")
    y_grouped, group_names = group_classes(y_dict, class_names)
    
    # 8. Train and evaluate grouped models for RF
    print("\n8. Training and evaluating grouped RandomForest models...")
    grouped_rf_models, X_test_grouped_rf, y_test_grouped_rf = train_multivariate_rf(X, y_grouped, group_names)
    grouped_rf_metrics = evaluate_multivariate_rf(
        grouped_rf_models, X_test_grouped_rf, y_test_grouped_rf, group_names,
        output_path=os.path.join(output_dir, f"grouped_classes_rf_regression{suffix}.png"),
        feature_names=feature_names
    )
    save_rf_models(
        grouped_rf_models, feature_names, group_names,
        output_path=os.path.join(output_dir, f"grouped_classes_rf_models{suffix}.joblib")
    )
    
    # 9. Train multi-output model for grouped classes with RF
    print("\n9. Training and evaluating multi-output grouped RandomForest model...")
    rf_multioutput_grouped_model, X_test_mo_g_rf, y_test_mo_g_rf, y_test_dict_mo_g_rf = train_multioutput_rf(X, y_grouped, group_names)
    rf_grouped_mo_metrics = evaluate_multioutput_rf(
        rf_multioutput_grouped_model, X_test_mo_g_rf, y_test_mo_g_rf, y_test_dict_mo_g_rf, group_names,
        output_path=os.path.join(output_dir, f"grouped_classes_rf_multioutput_regression{suffix}.png"),
        feature_names=feature_names
    )
    
    # Save multi-output grouped RF model
    joblib.dump({
        "model": rf_multioutput_grouped_model,
        "feature_names": feature_names,
        "class_names": group_names
    }, os.path.join(output_dir, "rf_multioutput_grouped_model.joblib"))
    
    # 10. Train and evaluate grouped models for HistGB
    print("\n10. Training and evaluating grouped HistGradientBoostingRegressor models...")
    grouped_histgb_models, X_test_grouped_histgb, y_test_grouped_histgb = train_histgb_rf(X, y_grouped, group_names)
    grouped_histgb_metrics = evaluate_multivariate_rf(
        grouped_histgb_models, X_test_grouped_histgb, y_test_grouped_histgb, group_names,
        output_path=os.path.join(output_dir, f"grouped_classes_histgb_regression{suffix}.png"),
        feature_names=feature_names
    )
    save_rf_models(
        grouped_histgb_models, feature_names, group_names,
        output_path=os.path.join(output_dir, f"grouped_classes_histgb_models{suffix}.joblib")
    )
    
    # 11. Train multi-output model for grouped classes with HistGB
    print("\n11. Training and evaluating multi-output grouped HistGradientBoostingRegressor model...")
    histgb_multioutput_grouped_model, X_test_mo_g_histgb, y_test_mo_g_histgb, y_test_dict_mo_g_histgb = train_multioutput_histgb(X, y_grouped, group_names)
    histgb_grouped_mo_metrics = evaluate_multioutput_rf(
        histgb_multioutput_grouped_model, X_test_mo_g_histgb, y_test_mo_g_histgb, y_test_dict_mo_g_histgb, group_names,
        output_path=os.path.join(output_dir, f"grouped_classes_histgb_multioutput_regression{suffix}.png"),
        feature_names=feature_names
    )
    
    # Save multi-output grouped HistGB model
    joblib.dump({
        "model": histgb_multioutput_grouped_model,
        "feature_names": feature_names,
        "class_names": group_names
    }, os.path.join(output_dir, "histgb_multioutput_grouped_model.joblib"))
    
    # 12. Compile and save all performance metrics
    print("\n12. Compiling performance metrics...")
    metrics_df = pd.DataFrame()
    
    # Add individual RF model metrics
    for class_name, class_metrics in rf_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['individual_rf'],
            'class': [class_name],
            'r2': [class_metrics['r2']],
            'rmse': [class_metrics['rmse']],
            'pearson': [class_metrics['pearson']]
        })])
    
    # Add multi-output RF model metrics
    for class_name, class_metrics in rf_mo_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['multioutput_rf'],
            'class': [class_name],
            'r2': [class_metrics['r2']],
            'rmse': [class_metrics['rmse']],
            'pearson': [class_metrics['pearson']]
        })])
    
    # Add individual HistGB model metrics
    for class_name, class_metrics in histgb_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['individual_histgb'],
            'class': [class_name],
            'r2': [class_metrics['r2']],
            'rmse': [class_metrics['rmse']],
            'pearson': [class_metrics['pearson']]
        })])
    
    # Add multi-output HistGB model metricsdata
    for group_name, group_metrics in grouped_rf_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['individual_rf_grouped'],
            'class': [group_name],
            'r2': [group_metrics['r2']],
            'rmse': [group_metrics['rmse']],
            'pearson': [group_metrics['pearson']]
        })])
    
    # Add multi-output grouped RF model metrics
    for group_name, group_metrics in rf_grouped_mo_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['multioutput_rf_grouped'],
            'class': [group_name],
            'r2': [group_metrics['r2']],
            'rmse': [group_metrics['rmse']],
            'pearson': [group_metrics['pearson']]
        })])
    
    # Add grouped individual HistGB model metrics
    for group_name, group_metrics in grouped_histgb_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['individual_histgb_grouped'],
            'class': [group_name],
            'r2': [group_metrics['r2']],
            'rmse': [group_metrics['rmse']],
            'pearson': [group_metrics['pearson']]
        })])
    
    # Add multi-output grouped HistGB model metrics
    for group_name, group_metrics in histgb_grouped_mo_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['multioutput_histgb_grouped'],
            'class': [group_name],
            'r2': [group_metrics['r2']],
            'rmse': [group_metrics['rmse']],
            'pearson': [group_metrics['pearson']]
        })])
    
    # Create a comparison bar plot of R² scores
    plt.figure(figsize=(14, 10))
    
    # Create a pivot table for easier plotting
    pivot_df = metrics_df.pivot_table(index='class', columns='method', values='r2')
    
    # Plot as a bar chart
    pivot_df.plot(kind='bar', figsize=(14, 10))
    plt.title('Comparison of R² Scores Across Different Models and Classes')
    plt.ylabel('R² Score')
    plt.xlabel('Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_r2.png"))
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"), index=False)
    print(f"\nPerformance metrics saved to {os.path.join(output_dir, 'regression_metrics.csv')}")
    
    print("\nMultivariate regression analysis completed successfully!")

def evaluate_multioutput_rf(model, X_test, y_test, y_test_dict, class_names, output_path, feature_names=None):
    """
    Evaluate the performance of the multi-output RF model
    
    Args:
        model: The trained multi-output model
        X_test: Test features
        y_test: Test targets matrix
        y_test_dict: Dictionary of test targets by class name 
        class_names: List of target class names
        output_path: Path to save the evaluation plots
        feature_names: List of feature names
    
    Returns:
        metrics: Dictionary with performance metrics for each class
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Create a figure for the scatter plots
    n_cols = min(3, len(class_names))
    n_rows = (len(class_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    
    # If only one subplot, axes needs to be in a 2D array
    if len(class_names) == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Evaluate each output dimension
    for i, class_name in enumerate(class_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        y_true = y_test[:, i]
        y_pred_i = y_pred[:, i]
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred_i)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_i))
        pearson_coef, _ = pearsonr(y_true, y_pred_i)
        
        metrics[class_name] = {
            "r2": r2,
            "rmse": rmse,
            "pearson": pearson_coef
        }
        
        # Create scatter plot
        ax.scatter(y_true, y_pred_i, alpha=0.5, s=10)
        max_val = max(np.max(y_true), np.max(y_pred_i))
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Customize labels based on whether this is a sqrt-transformed class
        if class_name.startswith('sqrt_') or 'sqrt' in class_name:
            ax.set_xlabel("Actual sqrt(proportion)")
            ax.set_ylabel("Predicted sqrt(proportion)")
            ax.set_title(f"{class_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        else:
            ax.set_xlabel("Actual proportion")
            ax.set_ylabel("Predicted proportion")
            ax.set_title(f"{class_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
    
    # Hide empty subplots
    for i in range(len(class_names), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Evaluation plot saved to {output_path}")
    
    return metrics

if __name__ == "__main__":
    # Run the full workflow for WAP32
    wap = 32
    use_peat = True
    peat_suffix = "_peat" if use_peat else ""
    data_dir = f"data/samples/selection13/regression_wap{wap}_5wd{peat_suffix}"  # Updated to use directory with sqrt data
    output_dir = f"data/samples/selection13/regression_wap{wap}_5wd{peat_suffix}/results"

    run_multivariate_regression(data_dir, output_dir, wap_number=wap, use_sqrt=True, use_peat=use_peat)
    
    # # Optionally, also run without sqrt transformation for comparison
    # output_dir_no_sqrt = f"data/samples/selection13/regression_wap{wap}_5wd_sqrt/results_no_sqrt"
    # run_multivariate_regression(data_dir, output_dir_no_sqrt, wap_number=wap, use_sqrt=False)
