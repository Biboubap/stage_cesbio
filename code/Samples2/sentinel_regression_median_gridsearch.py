"""
Perform regression analysis on balanced datasets created by sentinel_proportion_median_2.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from osgeo import gdal
import glob
import os
import joblib
from scipy.stats import pearsonr
import sys

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
    index_files = sorted(glob.glob(os.path.join(indices_dir, "*.tif"))) if indices_dir else []
    
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

def prepare_data_for_regression(csv_path, sentinel_features, feature_names, target_class):
    """
    Prepare data for regression by extracting features and the specific target from the CSV
    
    Args:
        csv_path: Path to the CSV file with class proportions
        sentinel_features: NumPy array of shape (n_features, rows, cols)
        feature_names: List of feature names
        target_class: Name of the target class to predict
    
    Returns:
        X: Array of features for each sample
        y: Array of target values
        found_target: Boolean indicating if target was found
    """
    # Load CSV with class proportions
    df = pd.read_csv(csv_path)
    
    # Check if target class exists in dataframe
    if target_class not in df.columns:
        print(f"Warning: Target class '{target_class}' not found in dataset.")
        return None, None, False
    
    print(f"Found target: {target_class}")
    
    # Prepare feature and target arrays
    X = []
    y = []
    
    # For each pixel in the CSV
    for _, row in df.iterrows():
        col_s = int(row["col_s"])
        row_s = int(row["row_s"])
        
        # Extract features for this pixel
        if (0 <= row_s < sentinel_features.shape[1] and 
            0 <= col_s < sentinel_features.shape[2]):
            pixel_features = sentinel_features[:, row_s, col_s]
            X.append(pixel_features)
            
            # Extract target value
            y.append(row[target_class])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    return X, y, True

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
        
        # Generic labels for all targets
        ax.set_xlabel("Actual value")
        ax.set_ylabel("Predicted value")
            
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

def perform_grid_search(X, y, target_class, param_grid=None, cv=5, scoring='r2'):
    """
    Perform grid search to find the best hyperparameters for a RandomForest model
    for a specific target class.
    
    Args:
        X: Features matrix
        y: Target values
        target_class: Name of the target class (for printing purposes)
        param_grid: Grid of parameters to search
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        
    Returns:
        best_model: The best model found
        best_params: The best parameters found
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    
    # Split the data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create the model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Create GridSearchCV object
    print(f"\nStarting GridSearchCV for {target_class} with {cv}-fold cross-validation...")
    print(f"Parameter grid: {param_grid}")
    print(f"Scoring metric: {scoring}")
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"\nGrid search completed for {target_class}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV {scoring} score: {grid_search.best_score_:.3f}")
    
    # Get the best model and validate on the validation set
    best_model = grid_search.best_estimator_
    val_score = best_model.score(X_val, y_val)
    val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_pearson, _ = pearsonr(y_val, val_pred)
    
    print(f"Validation set R² score: {val_score:.3f}")
    print(f"Validation set RMSE: {val_rmse:.3f}")
    print(f"Validation set Pearson r: {val_pearson:.3f}")
    
    # Print detailed CV results for each parameter combination
    print("\nDetailed cross-validation results:")
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values('rank_test_score')
    
    # Select only useful columns for display
    display_columns = ['rank_test_score', 'mean_test_score', 'std_test_score', 
                      'mean_train_score', 'std_train_score']
    
    # Add parameter columns
    param_columns = [col for col in cv_results.columns if col.startswith('param_')]
    display_columns = param_columns + display_columns
    
    # Display top 10 results
    print(cv_results[display_columns].head(10))
    
    return best_model, grid_search.best_params_

def run_grid_search_for_target(data_dir, output_dir, target_class, sentinel_features, feature_names, param_grid=None):
    """
    Run grid search for a specific target class
    
    Args:
        data_dir: Directory containing the balanced data files
        output_dir: Directory to save the output
        target_class: Name of the target class to perform grid search on
        sentinel_features: NumPy array of Sentinel features
        feature_names: List of feature names
        param_grid: Grid of parameters to search
    
    Returns:
        best_model: The best model found
        best_params: The best parameters found
    """
    # Create filename for the CSV
    csv_file = f"balanced_{target_class}.csv"
    csv_path = os.path.join(data_dir, csv_file)
    
    # Skip if file doesn't exist
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist, skipping {target_class}")
        return None, None
    
    # Prepare data for regression - focusing only on the target class column
    X, y, target_found = prepare_data_for_regression(
        csv_path=csv_path,
        sentinel_features=sentinel_features,
        feature_names=feature_names,
        target_class=target_class
    )
    
    # Skip if target not found
    if not target_found:
        return None, None
        
    print(f"Starting grid search for {target_class}...")
    
    # Perform grid search
    best_model, best_params = perform_grid_search(
        X=X, 
        y=y, 
        target_class=target_class,
        param_grid=param_grid
    )
    
    # Create a safe filename version of the target class
    safe_filename = target_class.replace('/', '_')
    
    # Save the best model
    joblib.dump({
        "model": best_model,
        "feature_names": feature_names,
        "target_name": target_class,
        "best_params": best_params
    }, os.path.join(output_dir, f"{safe_filename}_best_model.joblib"))
    
    print(f"Best model for {target_class} saved to {os.path.join(output_dir, f'{safe_filename}_best_model.joblib')}")
    
    # Save grid search results to text file
    result_text_path = os.path.join(output_dir, f"grid_search_results_{safe_filename}.txt")
    
    # Get validation metrics from the perform_grid_search function
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    val_score = best_model.score(X_val, y_val)
    val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_pearson, _ = pearsonr(y_val, val_pred)
    
    # Create the text file with results
    with open(result_text_path, 'w') as f:
        f.write(f"Grid search completed for {target_class}\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best CV r2 score: {best_model.score(X_train, y_train):.3f}\n")
        f.write(f"Validation set R² score: {val_score:.3f}\n")
        f.write(f"Validation set RMSE: {val_rmse:.3f}\n")
        f.write(f"Validation set Pearson r: {val_pearson:.3f}\n")
    
    print(f"Grid search results saved to {result_text_path}")
    
    return best_model, best_params

def run_all_regressions(data_dir, output_dir, wap_number=32, use_peat=False, superresolution=True, classes=None, moy5m=False, grid_search=False, grid_search_target=None):
    """
    Run individual regressions for each target class specified in the classes list
    
    Args:
        data_dir: Directory containing the balanced data files
        output_dir: Directory to save the output
        wap_number: WAP site number
        use_peat: Whether to use peat-masked data
        superresolution: Whether to use 5m (True) or 10m (False) resolution data
        classes: List of target classes to perform regression on
        moy5m: Whether to use moy5m data
        grid_search: Whether to perform grid search
        grid_search_target: Specific target for grid search (None to use all classes)
    """
    print(f"Starting regression analysis for {len(classes)} individual classes: {classes}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    peat_suffix = "_peat" if use_peat else ""
    
    # Set resolution and path modifiers based on superresolution flag
    file_prefix = "" if not moy5m else "10m_"
    resolution_suffix = "_10m" if not superresolution else ""

    # Define input paths
    sentinel_bands_dir = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_bands_10m/"
    sentinel_indices_dir = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_indices_10m/"
    
    # 1. Load Sentinel features
    print("\n1. Loading Sentinel features...")
    sentinel_features, feature_names = load_all_sentinel_features(
        indices_dir=sentinel_indices_dir, 
        bands_dir=sentinel_bands_dir
    )
    
    # Check if grid search is requested for a specific target
    if grid_search and grid_search_target:
        # Create a grid search output directory
        grid_search_dir = os.path.join(output_dir, "grid_search_results")
        os.makedirs(grid_search_dir, exist_ok=True)
        
        print(f"\nRunning grid search for specific target: {grid_search_target}")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        best_model, best_params = run_grid_search_for_target(
            data_dir=data_dir,
            output_dir=grid_search_dir,
            target_class=grid_search_target,
            sentinel_features=sentinel_features,
            feature_names=feature_names,
            param_grid=param_grid
        )
        
        # Early return if only grid search was requested
        if grid_search_target not in classes:
            print(f"Grid search completed for {grid_search_target}.")
            return
            
    # Results storage
    all_metrics = {}
    
    # 2. Process each target class in the list
    for target_class in classes:
        print(f"\n2. Processing {target_class} class...")
        
        # If grid search is requested for all classes
        if grid_search and grid_search_target is None:
            # Create a grid search output directory
            grid_search_dir = os.path.join(output_dir, "grid_search_results")
            os.makedirs(grid_search_dir, exist_ok=True)
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5, 10],
                'max_features': ['sqrt', "log2"]
            }
            
            best_model, best_params = run_grid_search_for_target(
                data_dir=data_dir,
                output_dir=grid_search_dir,
                target_class=target_class,
                sentinel_features=sentinel_features,
                feature_names=feature_names,
                param_grid=param_grid
            )
            
            # Use the best model for this target if found
            rf_params = best_params if best_params else {}
        else:
            # Use default parameters if not doing grid search
            rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 5,
                'min_samples_split': 10,
                'max_features': "sqrt",
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Create filename for the CSV
        csv_file = f"balanced_{target_class}.csv"
        csv_path = os.path.join(data_dir, csv_file)
        
        # Skip if file doesn't exist
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_path} does not exist, skipping {target_class}")
            continue
        
        # Prepare data for regression - focusing only on the target class column
        X, y, target_found = prepare_data_for_regression(
            csv_path=csv_path,
            sentinel_features=sentinel_features,
            feature_names=feature_names,
            target_class=target_class
        )
        
        # Skip if target not found
        if not target_found:
            continue
            
        print(f"Training RandomForest model for {target_class} with parameters: {rf_params}")
        
        # Train individual model for the specific target class
        rf = RandomForestRegressor(**rf_params)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train and predict
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        train_r2 = rf.score(X_train, y_train)
        print(f"Training R² for {target_class}: {train_r2:.3f}")   

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        print(f"R² for {target_class}: {r2:.3f}")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pearson_coef, _ = pearsonr(y_test, y_pred)
        
        all_metrics[target_class] = {
            'r2': r2,
            'rmse': rmse,
            'pearson': pearson_coef,
            'y_true': y_test,
            'y_pred': y_pred,
            'target_class': target_class
        }
        
        # Create a safe filename version of the target class
        safe_filename = target_class.replace('/', '_')
        
        # Save the model
        joblib.dump({
            "model": rf,
            "feature_names": feature_names,
            "target_name": target_class,
            "parameters": rf_params
        }, os.path.join(output_dir, f"{safe_filename}_rf.joblib"))
        
        # Create individual plot for this model
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5, s=10)
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        # Use generic labels for all targets
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
            
        plt.title(f"{target_class}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{safe_filename}_regression.png"))
        plt.close()
        
        # Plot feature importance
        if hasattr(rf, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort(rf.feature_importances_)[::-1]
            top_n = min(20, len(sorted_idx))
            top_idx = sorted_idx[:top_n]
            plt.barh(range(top_n), rf.feature_importances_[top_idx])
            plt.yticks(range(top_n), [feature_names[i] for i in top_idx])
            plt.title(f"Feature Importance for {target_class}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{safe_filename}_feature_importance.png"))
            plt.close()
    
    # Create performance summary
    print("\n3. Creating performance summary...")
    create_performance_summary(all_metrics, output_dir)

def create_performance_summary(all_metrics, output_dir):
    """
    Create a summary table of performance metrics for individual models
    
    Args:
        all_metrics: Dictionary containing metrics for all models
        output_dir: Directory to save output
    """
    metrics_data = []
    
    # Collect metrics for individual models
    for dataset_type, metrics in all_metrics.items():
        metrics_data.append({
            'Target': metrics['target_class'],
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'Pearson r': metrics['pearson']
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save as CSV
    metrics_df.to_csv(os.path.join(output_dir, "performance_metrics_summary.csv"), index=False)
    
    # Create a bar plot comparing R² values
    plt.figure(figsize=(10, 6))
    
    # Create bars sorted by R² value
    metrics_df_sorted = metrics_df.sort_values('R²', ascending=False)
    bars = plt.bar(metrics_df_sorted['Target'], metrics_df_sorted['R²'], color='steelblue')
    
    # Add labels and title
    plt.xlabel('Target Class')
    plt.ylabel('R² Score')
    plt.title('Model Performance (R²)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_summary.png"), dpi=300)
    plt.close()
    
    # Create RMSE comparison
    plt.figure(figsize=(10, 6))
    
    # Create bars sorted by RMSE (lower is better)
    metrics_df_sorted = metrics_df.sort_values('RMSE')
    bars = plt.bar(metrics_df_sorted['Target'], metrics_df_sorted['RMSE'], color='lightcoral')
    
    # Add labels and title
    plt.xlabel('Target Class')
    plt.ylabel('RMSE')
    plt.title('Model Performance (RMSE)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_summary.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Parameters
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    moy5m = False
    peat_suffix = "_peat" if use_peat else ""
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m_suffix = "_moy5m" if moy5m else ""

    # Define the list of classes for regression
    classes = ["sqrt_Pure_Lichen", "sqrt_Degraded_Lichen", "sqrt_Green", "sqrt_all_lichen", "sqrt_through_proportion"]

    data_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/balanced"
    output_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_results_gridsearch"
    
    # Check for grid search arguments
    grid_search = True
    grid_search_target = "sqrt_all_lichen"
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--grid-search":
        grid_search = True
        if len(sys.argv) > 2:
            grid_search_target = sys.argv[2]
            print(f"Running grid search for specific target: {grid_search_target}")
        else:
            print("Running grid search for all targets")
    
    # Run individual regressions for each class in the list
    run_all_regressions(
        data_dir=data_dir,
        output_dir=output_dir,
        wap_number=wap,
        use_peat=use_peat,
        superresolution=superresolution,
        classes=classes,
        moy5m=moy5m,
        grid_search=grid_search,
        grid_search_target=grid_search_target
    )

    print("\nAll regression analyses completed successfully!")
    print(f"Results saved to {output_dir}")

