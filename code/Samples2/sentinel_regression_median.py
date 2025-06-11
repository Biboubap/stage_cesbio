"""
Perform regression analysis on balanced datasets created by sentinel_proportion_median.py.
This script focuses on three grouped target classes:
1. Lichen
2. Chicoutai+Green (combined)
3. sqrt_through_proportion (sqrt of dry_depression + sphaignes + black_depression)
"""
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

def prepare_data_for_regression(csv_path, sentinel_features, feature_names, use_sqrt=True):
    """
    Prepare data for regression by extracting features and targets from the CSV
    
    Args:
        csv_path: Path to the CSV file with class proportions
        sentinel_features: NumPy array of shape (n_features, rows, cols)
        feature_names: List of feature names
        use_sqrt: Whether to prioritize sqrt_through_proportion over through_proportion
    
    Returns:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each group
        group_names: List of group names
    """
    # Load CSV with class proportions
    df = pd.read_csv(csv_path)
    
    # Define group columns to predict
    basic_groups = ['lichen', 'chicoutai_green']
    
    # Determine which through_proportion to use based on what's available
    if use_sqrt and 'sqrt_through_proportion' in df.columns:
        through_col = 'sqrt_through_proportion'
    elif 'through_proportion' in df.columns:
        through_col = 'through_proportion'
    elif 'sqrt_through_proportion' in df.columns:
        # Fallback to sqrt version if raw not available
        through_col = 'sqrt_through_proportion'
        print(f"Warning: through_proportion not found, using sqrt_through_proportion instead")
    else:
        through_col = None
        print(f"Warning: Neither through_proportion nor sqrt_through_proportion found in dataset")
    
    # Create target groups list
    group_cols = basic_groups.copy()
    if through_col:
        group_cols.append(through_col)
    
    # Check which columns are present in the dataframe
    available_groups = [col for col in group_cols if col in df.columns]
    print(f"Found {len(available_groups)} target groups: {available_groups}")
    
    # Prepare feature and target arrays
    X = []
    y_dict = {group_name: [] for group_name in available_groups}
    
    # For each pixel in the CSV
    for _, row in df.iterrows():
        col_s = int(row["col_s"])
        row_s = int(row["row_s"])
        
        # Extract features for this pixel
        if (0 <= row_s < sentinel_features.shape[1] and 
            0 <= col_s < sentinel_features.shape[2]):
            pixel_features = sentinel_features[:, row_s, col_s]
            X.append(pixel_features)
            
            # Extract target values for each group
            for group_name in available_groups:
                y_dict[group_name].append(row[group_name])
    
    X = np.array(X)
    for group_name in available_groups:
        y_dict[group_name] = np.array(y_dict[group_name])
        
        # Verify that all arrays have the same length
        if len(y_dict[group_name]) != len(X):
            print(f"Warning: Length mismatch for {group_name}: expected {len(X)}, got {len(y_dict[group_name])}")
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y_dict, available_groups

def train_multivariate_rf(X, y_dict, target_names, test_size=0.3, random_state=42):
    """
    Train a RandomForestRegressor for each target
    
    Args:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each target class
        target_names: List of target class names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        models: Dictionary of trained models for each target class
        X_test: Test features
        y_test_dict: Dictionary with test targets for each target class
    """
    # Split data into train and test sets
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    
    # Use the same train/test split for all targets
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    X_train, X_test = X[train_idx], X[test_idx]
    for target_name in target_names:
        y_train_dict[target_name] = y_dict[target_name][train_idx]
        y_test_dict[target_name] = y_dict[target_name][test_idx]
    
    # Train a model for each target class
    models = {}
    for target_name in target_names:
        print(f"Training RandomForest model for {target_name}...")
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=random_state,
            max_depth=None,
            max_features="sqrt",
            n_jobs=-1  # Use all available cores
        )
        rf.fit(X_train, y_train_dict[target_name])
        models[target_name] = rf
    
    return models, X_test, y_test_dict

def train_multioutput_rf(X, y_dict, target_names, test_size=0.3, random_state=42):
    """
    Train a multi-output RandomForest model that predicts all targets at once
    
    Args:
        X: Array of features for each sample
        y_dict: Dictionary with arrays of targets for each target class
        target_names: List of target class names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        model: The trained multi-output model
        X_test: Test features
        y_test: Test targets matrix
        y_test_dict: Dictionary of test targets by target name
    """
    # Combine all target variables into a single matrix
    y_matrix = np.column_stack([y_dict[target_name] for target_name in target_names])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_matrix, test_size=test_size, random_state=random_state
    )
    
    # Create and train multi-output model directly using RandomForestRegressor
    # which can handle multivariate outputs natively
    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=4,
        random_state=random_state,
        max_depth=None,
        max_features="sqrt",
        n_jobs=-1  # Use all available cores for faster training
    )
    
    print("Training multi-output Random Forest model...")
    model.fit(X_train, y_train)
    
    # Create y_test_dict for evaluation
    y_test_dict = {}
    for i, target_name in enumerate(target_names):
        y_test_dict[target_name] = y_test[:, i]
    
    return model, X_test, y_test, y_test_dict

def evaluate_regression_models(models, X_test, y_test_dict, target_names, output_path, feature_names=None):
    """
    Evaluate the performance of regression models and create plots
    
    Args:
        models: Dictionary of trained models for each target
        X_test: Test features
        y_test_dict: Dictionary with test targets for each target
        target_names: List of target names
        output_path: Path to save the evaluation plots
        feature_names: List of feature names used for training
    
    Returns:
        metrics: Dictionary with performance metrics for each target
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Create a figure for the scatter plots
    n_cols = min(3, len(target_names))
    n_rows = (len(target_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    
    # If only one subplot, axes needs to be in a 2D array
    if len(target_names) == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Evaluate each model
    for i, target_name in enumerate(target_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Make predictions
        model = models[target_name]
        y_pred = model.predict(X_test)
        y_true = y_test_dict[target_name]
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pearson_coef, _ = pearsonr(y_true, y_pred)
        
        metrics[target_name] = {
            "r2": r2,
            "rmse": rmse,
            "pearson": pearson_coef
        }
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Customize labels based on the target
        if target_name == 'sqrt_through_proportion':
            ax.set_xlabel("Actual sqrt(proportion)")
            ax.set_ylabel("Predicted sqrt(proportion)")
        else:
            ax.set_xlabel("Actual proportion")
            ax.set_ylabel("Predicted proportion")
            
        ax.set_title(f"{target_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
    
    # Hide empty subplots
    for i in range(len(target_names), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Evaluation plot saved to {output_path}")
    
    # Also create a feature importance plot if feature_names is provided
    if feature_names is not None:
        plot_feature_importance(models, target_names, output_path.replace('.png', '_feature_importance.png'), feature_names)
    
    return metrics

def evaluate_multioutput_rf(model, X_test, y_test, y_test_dict, target_names, output_path, feature_names=None):
    """
    Evaluate the performance of the multi-output RF model
    
    Args:
        model: The trained multi-output model
        X_test: Test features
        y_test: Test targets matrix
        y_test_dict: Dictionary of test targets by target name
        target_names: List of target names
        output_path: Path to save the evaluation plots
        feature_names: List of feature names
    
    Returns:
        metrics: Dictionary with performance metrics for each target
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Create a figure for the scatter plots
    n_cols = min(3, len(target_names))
    n_rows = (len(target_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    
    # If only one subplot, axes needs to be in a 2D array
    if len(target_names) == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Evaluate each output dimension
    for i, target_name in enumerate(target_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        y_true = y_test[:, i]
        y_pred_i = y_pred[:, i]
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred_i)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_i))
        pearson_coef, _ = pearsonr(y_true, y_pred_i)
        
        metrics[target_name] = {
            "r2": r2,
            "rmse": rmse,
            "pearson": pearson_coef
        }
        
        # Create scatter plot
        ax.scatter(y_true, y_pred_i, alpha=0.5, s=10)
        max_val = max(np.max(y_true), np.max(y_pred_i))
        ax.plot([0, max_val], [0, max_val], 'r--')
        
        # Customize labels based on the target
        if target_name == 'sqrt_through_proportion':
            ax.set_xlabel("Actual sqrt(proportion)")
            ax.set_ylabel("Predicted sqrt(proportion)")
        else:
            ax.set_xlabel("Actual proportion")
            ax.set_ylabel("Predicted proportion")
            
        ax.set_title(f"{target_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
    
    # Hide empty subplots
    for i in range(len(target_names), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Evaluation plot saved to {output_path}")
    
    # Create feature importance visualization if feature_names provided
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(model.feature_importances_)[::-1]
        plt.barh(range(len(sorted_idx)), model.feature_importances_[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title("Feature Importance (Multi-output RandomForest)")
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_feature_importance.png'))
        plt.close()
    
    return metrics

def plot_feature_importance(models, target_names, output_path, feature_names, top_n=20):
    """
    Plot feature importance for all models
    
    Args:
        models: Dictionary of trained models
        target_names: List of target names
        output_path: Path to save the plot
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    # Compute average importance across all models
    avg_importance = np.zeros(len(feature_names))
    
    for target_name in target_names:
        model = models[target_name]
        if hasattr(model, 'feature_importances_'):
            avg_importance += model.feature_importances_
    
    avg_importance /= len(target_names)
    
    # Sort by average importance
    indices = np.argsort(avg_importance)[::-1]
    
    # Limit to top_n features
    if len(indices) > top_n:
        indices = indices[:top_n]
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create bars for each target + average
    bar_width = 0.8 / (len(target_names) + 1)
    x = np.arange(len(indices))
    
    # Plot bars for each target
    for i, target_name in enumerate(target_names):
        model = models[target_name]
        if hasattr(model, 'feature_importances_'):
            plt.bar(x + i * bar_width, model.feature_importances_[indices], 
                    bar_width, alpha=0.7, label=target_name)
    
    # Plot average importance
    plt.bar(x + len(target_names) * bar_width, avg_importance[indices], 
            bar_width, color='black', alpha=0.7, label='Average')
    
    # Add labels and legend
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance by Target')
    plt.xticks(x + bar_width * (len(target_names) / 2), 
              [feature_names[i] for i in indices], rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")

def save_models(models, feature_names, target_names, output_path):
    """
    Save the trained models
    
    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
        target_names: List of target names
        output_path: Path to save the models
    """
    model_data = {
        "models": models,
        "feature_names": feature_names,
        "target_names": target_names
    }
    joblib.dump(model_data, output_path)
    print(f"Models saved to {output_path}")

def run_all_regressions(data_dir, output_dir, wap_number=32, use_peat=False, superresolution=True, use_sqrt=True):
    """
    Run all regressions (multi-output and individual) and produce consolidated output
    
    Args:
        data_dir: Directory containing the balanced data files
        output_dir: Directory to save the output
        wap_number: WAP site number
        use_peat: Whether to use peat-masked data
        superresolution: Whether to use 5m (True) or 10m (False) resolution data
        use_sqrt: Whether to use sqrt-transformed data for through proportion (True) or raw through proportion (False)
    """
    print("Starting regression analysis for all datasets...")
    print(f"Using sqrt transformation for through proportion: {use_sqrt}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    peat_suffix = "_peat" if use_peat else ""
    
    # Set resolution and path modifiers based on superresolution flag
    mediane_dir = "mediane" if superresolution else "mediane_10m"
    file_prefix = "" if superresolution else "10m_"
    
    # Define input paths
    sentinel_bands_dir = f"DataCubeS2/BandsS22023_WAP{wap_number}{peat_suffix}/{mediane_dir}"
    sentinel_indices_dir = f"DataCubeS2/IndicesS22023_WAP{wap_number}{peat_suffix}/{mediane_dir}"
    
    # Define through_proportion file and target based on use_sqrt
    through_file = "balanced_sqrt_through_proportion.csv" if use_sqrt else "balanced_through_proportion.csv"
    through_target = "sqrt_through_proportion" if use_sqrt else "through_proportion"
    
    # Fall back to alternative if file doesn't exist
    alternative_through_file = "balanced_through_proportion.csv" if use_sqrt else "balanced_sqrt_through_proportion.csv"
    alternative_through_target = "through_proportion" if use_sqrt else "sqrt_through_proportion"
    
    through_path = os.path.join(data_dir, through_file)
    if not os.path.exists(through_path):
        print(f"Warning: {through_file} not found, trying {alternative_through_file} instead")
        through_file = alternative_through_file
        through_target = alternative_through_target
        through_path = os.path.join(data_dir, through_file)
        if not os.path.exists(through_path):
            print(f"Error: Neither {through_file} nor {alternative_through_file} exist in {data_dir}")
    
    # Define datasets, their CSV paths, and corresponding target classes
    datasets = {
        'individual_lichen': {
            'path': os.path.join(data_dir, "balanced_lichen.csv"),
            'target': 'lichen'
        },
        'individual_chicoutai_green': {
            'path': os.path.join(data_dir, "balanced_chicoutai_green.csv"),
            'target': 'chicoutai_green'
        },
        'individual_through': {
            'path': os.path.join(data_dir, through_file),
            'target': through_target
        }
    }
    
    # Choose combined file based on what's available
    combined_file = "balanced_combined_sqrt.csv" if use_sqrt else "balanced_combined_raw.csv"
    combined_path = os.path.join(data_dir, combined_file)
    
    # Fall back to alternative if file doesn't exist
    if not os.path.exists(combined_path):
        alternative = "balanced_combined_raw.csv" if use_sqrt else "balanced_combined_sqrt.csv"
        alt_path = os.path.join(data_dir, alternative)
        if os.path.exists(alt_path):
            print(f"Warning: {combined_file} not found, using {alternative} instead")
            combined_file = alternative
            combined_path = alt_path
        else:
            # Final fallback to legacy combined.csv
            legacy_combined = "balanced_combined.csv"
            legacy_path = os.path.join(data_dir, legacy_combined)
            if os.path.exists(legacy_path):
                print(f"Warning: {combined_file} and {alternative} not found, using {legacy_combined} instead")
                combined_file = legacy_combined
                combined_path = legacy_path
    
    # Define multioutput dataset
    datasets['multioutput'] = {
        'path': combined_path,
        'targets': ['lichen', 'chicoutai_green', through_target]
    }
    
    # 1. Load Sentinel features
    print("\n1. Loading Sentinel features...")
    sentinel_features, feature_names = load_all_sentinel_features(
        indices_dir=sentinel_indices_dir, 
        bands_dir=sentinel_bands_dir
    )
    
    # Results storage
    all_metrics = {}
    
    # 2. Process each dataset for individual models
    for dataset_type, dataset_info in datasets.items():
        if dataset_type == 'multioutput':
            continue  # Skip multioutput for now, process it separately
          
        print(f"\n2. Processing {dataset_type} dataset...")
        csv_path = dataset_info['path']
        target_class = dataset_info['target']
        
        # Skip if file doesn't exist
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_path} does not exist, skipping {dataset_type}")
            continue
        
        # Prepare data for regression
        X, y_dict, available_groups = prepare_data_for_regression(
            csv_path=csv_path,
            sentinel_features=sentinel_features,
            feature_names=feature_names,
            use_sqrt=use_sqrt
        )
        
        # Ensure the target class is available
        if target_class not in available_groups:
            print(f"Warning: Target class '{target_class}' not found in dataset. Available groups: {available_groups}")
            continue
            
        print(f"Training RandomForest model for {target_class}...")
        
        # Train individual model for the specific target class
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=42,
            max_depth=None,
            max_features="sqrt",
            n_jobs=-1
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_dict[target_class], test_size=0.3, random_state=42
        )
        
        # Train and predict
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pearson_coef, _ = pearsonr(y_test, y_pred)
        
        all_metrics[dataset_type] = {
            'r2': r2,
            'rmse': rmse,
            'pearson': pearson_coef,
            'y_true': y_test,
            'y_pred': y_pred,
            'target_class': target_class
        }
        
        # Save the model
        joblib.dump({
            "model": rf,
            "feature_names": feature_names,
            "target_name": target_class
        }, os.path.join(output_dir, f"{dataset_type}_rf.joblib"))
    
    # 3. Now process the multioutput model
    if 'multioutput' in datasets:
        print("\n3. Processing multioutput dataset...")
        csv_path = datasets['multioutput']['path']
        
        # Prepare data for regression
        X, y_dict, available_groups = prepare_data_for_regression(
            csv_path=csv_path,
            sentinel_features=sentinel_features,
            feature_names=feature_names
        )
        
        # Train multi-output model
        print("Training multi-output RandomForest model...")
        
        # Combine all target variables into a single matrix
        # Only use available groups that are in our target list
        target_names = [tgt for tgt in datasets['multioutput']['targets'] if tgt in available_groups]
        y_matrix = np.column_stack([y_dict[target_name] for target_name in target_names])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_matrix, test_size=0.3, random_state=42
        )
        
        # Create and train multi-output model
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=42,
            max_depth=None,
            max_features="sqrt",
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics for each target
        multioutput_metrics = {}
        for i, target_name in enumerate(target_names):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            pearson_coef, _ = pearsonr(y_test[:, i], y_pred[:, i])
            
            multioutput_metrics[target_name] = {
                'r2': r2,
                'rmse': rmse,
                'pearson': pearson_coef,
                'y_true': y_test[:, i],
                'y_pred': y_pred[:, i]
            }
        
        all_metrics['multioutput'] = multioutput_metrics
        
        # Save the model
        joblib.dump({
            "model": model,
            "feature_names": feature_names,
            "target_names": target_names
        }, os.path.join(output_dir, "multioutput_rf.joblib"))
    
    # 4. Create a consolidated visualization
    print("\n4. Creating consolidated visualization...")
    create_consolidated_plot(all_metrics, output_dir, use_sqrt)
    
    # 5. Create summary of performance metrics
    print("\n5. Creating performance summary...")
    create_performance_summary(all_metrics, output_dir)

def create_consolidated_plot(all_metrics, output_dir, use_sqrt=True):
    """
    Create a consolidated plot showing all regression results
    
    Args:
        all_metrics: Dictionary containing metrics for all models
        output_dir: Directory to save output
        use_sqrt: Whether sqrt transformation was used for through proportion
    """
    # Define target names based on use_sqrt parameter
    if use_sqrt:
        target_names = ['lichen', 'chicoutai_green', 'sqrt_through_proportion']
        dataset_types = ['individual_lichen', 'individual_chicoutai_green', 'individual_through']
    else:
        target_names = ['lichen', 'chicoutai_green', 'through_proportion']
        dataset_types = ['individual_lichen', 'individual_chicoutai_green', 'individual_through']
    
    # Create a 2x3 grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot individual models (top row)
    for col, dataset_type in enumerate(dataset_types):
        ax = axes[0, col]
        
        if dataset_type in all_metrics:
            metrics = all_metrics[dataset_type]
            y_true = metrics['y_true']
            y_pred = metrics['y_pred']
            r2 = metrics['r2']
            rmse = metrics['rmse']
            pearson = metrics['pearson']
            target_class = metrics['target_class']
            
            # Plot scatter
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([0, max_val], [0, max_val], 'r--')
            
            # Add title and metrics
            title = f"Individual {target_class}\n" \
                    f"R² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson:.3f}"
            ax.set_title(title)
            
            # Customize labels based on target
            if target_class.startswith('sqrt_'):
                ax.set_xlabel("Actual sqrt(proportion)")
                ax.set_ylabel("Predicted sqrt(proportion)")
            else:
                ax.set_xlabel("Actual proportion")
                ax.set_ylabel("Predicted proportion")
                
            ax.grid(alpha=0.3)
            ax.set_xlim(0, max_val * 1.05)
            ax.set_ylim(0, max_val * 1.05)
        else:
            ax.text(0.5, 0.5, f"No data for {dataset_type}", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Plot multi-output model results (bottom row) only if using sqrt transform
    if use_sqrt and 'multioutput' in all_metrics:
        for col, target_class in enumerate(target_names):
            ax = axes[1, col]
            
            if target_class in all_metrics['multioutput']:
                metrics = all_metrics['multioutput'][target_class]
                
                y_true = metrics['y_true']
                y_pred = metrics['y_pred']
                r2 = metrics['r2']
                rmse = metrics['rmse']
                pearson = metrics['pearson']
                
                # Plot scatter
                ax.scatter(y_true, y_pred, alpha=0.5, s=10)
                max_val = max(np.max(y_true), np.max(y_pred))
                ax.plot([0, max_val], [0, max_val], 'r--')
                
                # Add title and metrics
                title = f"Multi-output {target_class}\n" \
                        f"R² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson:.3f}"
                ax.set_title(title)
                
                # Customize labels based on target
                if target_class.startswith('sqrt_'):
                    ax.set_xlabel("Actual sqrt(proportion)")
                    ax.set_ylabel("Predicted sqrt(proportion)")
                else:
                    ax.set_xlabel("Actual proportion")
                    ax.set_ylabel("Predicted proportion")
                
                ax.grid(alpha=0.3)
                ax.set_xlim(0, max_val * 1.05)
                ax.set_ylim(0, max_val * 1.05)
            else:
                ax.text(0.5, 0.5, f"No multi-output data for {target_class}", 
                       horizontalalignment='center', verticalalignment='center')
                ax.set_xticks([])
                ax.set_yticks([])
    else:
        # If not using sqrt transform, hide the bottom row
        for col in range(3):
            axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consolidated_regression_results.png"), dpi=300)
    plt.close()

def create_performance_summary(all_metrics, output_dir):
    """
    Create a summary plot and table of all performance metrics
    
    Args:
        all_metrics: Dictionary containing metrics for all models
        output_dir: Directory to save output
    """
    # Define target classes
    target_names = ['lichen', 'chicoutai_green', 'through_proportion', 'sqrt_through_proportion']
    metrics_data = []
    
    # Collect metrics for individual models
    dataset_types = ['individual_lichen', 'individual_chicoutai_green', 'individual_through', 'individual_sqrt_through']

    for dataset_type in dataset_types:
        if dataset_type in all_metrics:
            metrics = all_metrics[dataset_type]
            target_class = metrics['target_class']
            
            metrics_data.append({
                'Target': target_class,
                'Model Type': 'Individual',
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'Pearson r': metrics['pearson']
            })
    
    # Collect metrics for multi-output model
    if 'multioutput' in all_metrics:
        for target_class in target_names:
            if target_class in all_metrics['multioutput']:
                metrics = all_metrics['multioutput'][target_class]
                
                metrics_data.append({
                    'Target': target_class,
                    'Model Type': 'Multi-output',
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'Pearson r': metrics['pearson']
                })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save as CSV
    metrics_df.to_csv(os.path.join(output_dir, "performance_metrics_summary.csv"), index=False)
    
    # Create a bar plot comparing R² values
    plt.figure(figsize=(12, 8))
    
    # Set up positions for the bars
    x = np.arange(len(target_names))
    width = 0.35
    
    # Extract R² values for individual and multi-output models for each target
    individual_r2 = []
    multioutput_r2 = []
    
    for target in target_names:
        # Find R² for individual model
        ind_r2 = metrics_df[(metrics_df['Target'] == target) & 
                           (metrics_df['Model Type'] == 'Individual')]['R²'].values
        individual_r2.append(ind_r2[0] if len(ind_r2) > 0 else np.nan)
        
        # Find R² for multi-output model
        mo_r2 = metrics_df[(metrics_df['Target'] == target) & 
                           (metrics_df['Model Type'] == 'Multi-output')]['R²'].values
        multioutput_r2.append(mo_r2[0] if len(mo_r2) > 0 else np.nan)
    
    # Create bars
    plt.bar(x - width/2, individual_r2, width, label='Individual Model', color='steelblue')
    plt.bar(x + width/2, multioutput_r2, width, label='Multi-output Model', color='darkorange')
    
    # Add labels, title and legend
    plt.xlabel('Target Class')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison: Individual vs. Multi-output')
    plt.xticks(x, target_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(individual_r2):
        if not np.isnan(v):
            plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(multioutput_r2):
        if not np.isnan(v):
            plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_summary.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Parameters
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    use_sqrt = False          # Use sqrt-transformed through proportion (True) or raw through proportion (False)
    peat_suffix = "_peat" if use_peat else ""
    resolution_suffix = "" if superresolution else "_10m"
    
    
    data_dir = f"data/samples/selection14/regression_wap{wap}_no_chicoutai{peat_suffix}{resolution_suffix}/balanced"
    output_dir = f"data/samples/selection14/regression_wap{wap}_no_chicoutai{peat_suffix}{resolution_suffix}/regression_results"
    
    # Add suffix to output directory when not using sqrt transform
    if not use_sqrt:
        output_dir += "_raw"
    
    # Run all regressions and create consolidated output
    run_all_regressions(
        data_dir=data_dir,
        output_dir=output_dir,
        wap_number=wap,
        use_peat=use_peat,
        superresolution=superresolution,
        use_sqrt=use_sqrt
    )

    print("\nAll regression analyses completed successfully!")
    print(f"Results saved to {output_dir}")
