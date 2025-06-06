import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

def prepare_data_for_regression(csv_path, sentinel_features, feature_names):
    """
    Prepare data for regression by extracting features and targets from the CSV
    
    Args:
        csv_path: Path to the CSV file with class proportions
        sentinel_features: NumPy array of shape (n_features, rows, cols)
        feature_names: List of feature names
    
    Returns:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each class
        class_names: List of class names
    """
    # Load CSV with class proportions
    df = pd.read_csv(csv_path)
    
    # Find class columns (exclude metadata columns and the 'none' class)
    metadata_cols = ['col_s', 'row_s', 'valid_pixels', 'total_pixels', 'valid_proportion']
    class_cols = [col for col in df.columns if col not in metadata_cols 
                 and not col.startswith('band_') and not col.startswith('index_')
                 and col != 'none']  # Exclude the 'none' class
    
    print(f"Found {len(class_cols)} classes: {class_cols}")
    
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
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y_dict, class_cols

def group_classes(y_dict, class_names):
    """
    Group classes into 3 categories:
    1. [Lichen]
    2. [Chicoutai, Green Depression]
    3. [Dry Depression, Sphaignes, Dark-Depression, Wet Depression]
    
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
    group3 = ["dry_depression", "sphaignes", "black_depression", "watered_depression"]
    
    # Initialize arrays for groups
    n_samples = len(y_dict[class_names[0]])
    y_group1 = np.zeros(n_samples)
    y_group2 = np.zeros(n_samples)
    y_group3 = np.zeros(n_samples)
    
    # Sum proportions for each group
    for class_name in class_names:
        if class_name in group1:
            y_group1 += y_dict[class_name]
        elif class_name in group2:
            y_group2 += y_dict[class_name]
        elif class_name in group3:
            y_group3 += y_dict[class_name]
    
    # Create dictionary for grouped targets
    y_grouped = {
        "group1_lichen": y_group1,
        "group2_chicoutai_green": y_group2,
        "group3_dry_dark_wet_spha": y_group3
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
        avg_importance += models[class_name].feature_importances_
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
        importance = models[class_name].feature_importances_[indices]
        plt.bar(x + i * bar_width, importance, bar_width, alpha=0.7, label=class_name)
    
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
        min_samples_leaf=1,
        random_state=random_state,
        max_depth=None,
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
    
    # No feature importance plot for multi-output model since it's more complex
    
    return metrics

def run_multivariate_regression(csv_path, output_dir, wap_number=32):
    """
    Run the full multivariate regression workflow
    
    Args:
        csv_path: Path to the CSV file with class proportions
        output_dir: Directory to save the output
        wap_number: WAP site number
    """
    print("Starting multivariate Random Forest regression...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    sentinel_bands_dir = f"DataCubeS2/BandsS22023_WAP{wap_number}/mediane"
    sentinel_indices_dir = f"DataCubeS2/IndicesS22023_WAP{wap_number}/mediane"
    
    
    # 1. Load Sentinel features
    print("\n1. Loading Sentinel features...")
    sentinel_features, feature_names = load_all_sentinel_features(sentinel_indices_dir, sentinel_bands_dir)
    
    # 2. Prepare data for regression
    print("\n2. Preparing data for regression...")
    X, y_dict, class_names = prepare_data_for_regression(csv_path, sentinel_features, feature_names)
    
    # 3. Train and evaluate individual models for all 7 classes
    print("\n3. Training and evaluating individual models for all classes...")
    models, X_test, y_test_dict = train_multivariate_rf(X, y_dict, class_names)
    metrics = evaluate_multivariate_rf(
        models, X_test, y_test_dict, class_names,
        output_path=os.path.join(output_dir, "all_classes_regression.png"),
        feature_names=feature_names
    )
    save_rf_models(
        models, feature_names, class_names,
        output_path=os.path.join(output_dir, "all_classes_rf_models.joblib")
    )
    
    # 4. Train and evaluate multioutput model for all 7 classes
    print("\n4. Training and evaluating multi-output model for all classes...")
    # Combine targets into a matrix
    y_matrix = np.column_stack([y_dict[class_name] for class_name in class_names])
    multioutput_model, X_test_mo, y_test_mo, y_test_dict_mo = train_multioutput_rf(X, y_dict, class_names)
    metrics_mo = evaluate_multioutput_rf(
        multioutput_model, X_test_mo, y_test_mo, y_test_dict_mo, class_names,
        output_path=os.path.join(output_dir, "all_classes_multioutput_regression.png"),
        feature_names=feature_names
    )
    
    # Save multi-output model
    joblib.dump({
        "model": multioutput_model,
        "feature_names": feature_names,
        "class_names": class_names
    }, os.path.join(output_dir, "multioutput_rf_model.joblib"))
    
    # 5. Group classes and train models for the 3 grouped classes
    print("\n5. Training and evaluating models for grouped classes...")
    y_grouped, group_names = group_classes(y_dict, class_names)
    grouped_models, X_test_grouped, y_test_grouped = train_multivariate_rf(X, y_grouped, group_names)
    grouped_metrics = evaluate_multivariate_rf(
        grouped_models, X_test_grouped, y_test_grouped, group_names,
        output_path=os.path.join(output_dir, "grouped_classes_regression.png"),
        feature_names=feature_names
    )
    save_rf_models(
        grouped_models, feature_names, group_names,
        output_path=os.path.join(output_dir, "grouped_classes_rf_models.joblib")
    )
    
    # 6. Train multi-output model for grouped classes
    print("\n6. Training and evaluating multi-output model for grouped classes...")
    multioutput_grouped_model, X_test_mo_g, y_test_mo_g, y_test_dict_mo_g = train_multioutput_rf(X, y_grouped, group_names)
    metrics_mo_g = evaluate_multioutput_rf(
        multioutput_grouped_model, X_test_mo_g, y_test_mo_g, y_test_dict_mo_g, group_names,
        output_path=os.path.join(output_dir, "grouped_classes_multioutput_regression.png"),
        feature_names=feature_names
    )
    
    # Save multi-output grouped model
    joblib.dump({
        "model": multioutput_grouped_model,
        "feature_names": feature_names,
        "class_names": group_names
    }, os.path.join(output_dir, "multioutput_grouped_rf_model.joblib"))
    
    # 7. Save performance metrics
    metrics_df = pd.DataFrame()
    
    # Add individual model metrics
    for class_name, class_metrics in metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['individual_rf'],
            'class': [class_name],
            'r2': [class_metrics['r2']],
            'rmse': [class_metrics['rmse']],
            'pearson': [class_metrics['pearson']]
        })])
    
    # Add multi-output model metrics
    for class_name, class_metrics in metrics_mo.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['multioutput_rf'],
            'class': [class_name],
            'r2': [class_metrics['r2']],
            'rmse': [class_metrics['rmse']],
            'pearson': [class_metrics['pearson']]
        })])
    
    # Add grouped individual model metrics
    for group_name, group_metrics in grouped_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['individual_rf_grouped'],
            'class': [group_name],
            'r2': [group_metrics['r2']],
            'rmse': [group_metrics['rmse']],
            'pearson': [group_metrics['pearson']]
        })])
    
    # Add multi-output grouped model metrics
    for group_name, group_metrics in metrics_mo_g.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'method': ['multioutput_rf_grouped'],
            'class': [group_name],
            'r2': [group_metrics['r2']],
            'rmse': [group_metrics['rmse']],
            'pearson': [group_metrics['pearson']]
        })])
    
    metrics_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"), index=False)
    print(f"\nPerformance metrics saved to {os.path.join(output_dir, 'regression_metrics.csv')}")
    
    print("\nMultivariate Random Forest regression completed successfully!")

if __name__ == "__main__":
    # Run the full workflow for WAP32
    wap = 32
    csv_path = os.path.join(data_dir, f"class_proportions_WAP{wap_number}_filtered.csv")
    output_dir = f"data/samples/selection13/regression_rf_wap{wap}"
    
    run_multivariate_regression(data_dir, output_dir, wap_number=wap)
