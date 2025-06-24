"""
Perform regression analysis on balanced datasets created by sentinel_proportion_median_2.py
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


def prepare_regression_data(data_dir, wap_number=32, use_peat=False, superresolution=True, moy5m=False):
    """
    Prepare regression data by loading Sentinel features and class proportions
    
    Args:
        data_dir: Directory containing the balanced data files
        wap_number: WAP site number
        use_peat: Whether to use peat-masked data
        superresolution: Whether to use 5m (True) or 10m (False) resolution data
        moy5m: Whether to use moy5m data
        
    Returns:
        dict: Dictionary containing sentinel features, feature names, and other metadata
    """
    print("Preparing regression data...")
    
    # Set resolution and path modifiers based on parameters
    peat_suffix = "_peat" if use_peat else ""
    resolution_suffix = "_10m" if not superresolution else ""
    file_prefix = "" if not moy5m else "10m_"

    # Define input paths
    sentinel_bands_dir = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_bands_10m/"
    sentinel_indices_dir = f"DataCubeS2/WAP{wap_number}{peat_suffix}{resolution_suffix}/mediane_indices_10m/"
    
    # Load Sentinel features
    print("Loading Sentinel features...")
    sentinel_features, feature_names = load_all_sentinel_features(
        indices_dir=sentinel_indices_dir, 
        bands_dir=sentinel_bands_dir
    )
    
    # Return prepared data
    return {
        "sentinel_features": sentinel_features,
        "feature_names": feature_names,
        "data_dir": data_dir
    }

def run_one_regression(prepared_data, output_dir, target_class, rf_params=None):
    """
    Run regression for a specific target class with custom RF parameters
    
    Args:
        prepared_data: Dictionary with prepared regression data from prepare_regression_data
        output_dir: Directory to save output files
        target_class: Target class for regression (e.g., "sqrt_Pure_Lichen")
        rf_params: Optional dictionary with RandomForestRegressor parameters
        
    Returns:
        dict: Dictionary with regression results and metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract prepared data
    sentinel_features = prepared_data["sentinel_features"]
    feature_names = prepared_data["feature_names"]
    data_dir = prepared_data["data_dir"]
    
    # Default RF parameters
    if rf_params is None:
        rf_params = {
            "n_estimators": 200,
            "min_samples_leaf": 1,
            "min_samples_split": 10,
            "random_state": 42,
            "max_depth": 10,
            "max_features": "sqrt",
            "n_jobs": -1
        }
    
    print(f"\nProcessing {target_class} class with custom RF parameters...")
    
    # Create filename for the CSV
    csv_file = f"balanced_{target_class}.csv"
    csv_path = os.path.join(data_dir, csv_file)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist, skipping {target_class}")
        return None
    
    # Prepare data for regression - focusing only on the target class column
    X, y, target_found = prepare_data_for_regression(
        csv_path=csv_path,
        sentinel_features=sentinel_features,
        feature_names=feature_names,
        target_class=target_class
    )
    
    # Skip if target not found
    if not target_found:
        return None
        
    print(f"Training RandomForest model for {target_class} with custom parameters...")
    print(f"RF parameters: {rf_params}")
    
    # Create RF model with specified parameters
    rf = RandomForestRegressor(**rf_params)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train and predict
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_coef, _ = pearsonr(y_test, y_pred)
    
    # Check if this is a sqrt_ class and calculate additional metrics
    nonsqrt_metrics = None
    if target_class.startswith("sqrt_"):
        # Calculate metrics for squared values (original values)
        y_test_squared = y_test ** 2
        y_pred_squared = y_pred ** 2
        
        # Calculate squared metrics
        r2_squared = r2_score(y_test_squared, y_pred_squared)
        rmse_squared = np.sqrt(mean_squared_error(y_test_squared, y_pred_squared))
        pearson_coef_squared, _ = pearsonr(y_test_squared, y_pred_squared)
        
        nonsqrt_metrics = {
            'r2': r2_squared,
            'rmse': rmse_squared,
            'pearson': pearson_coef_squared
        }
        
        print(f"Original sqrt metrics - R²: {r2:.3f}, RMSE: {rmse:.3f}, Pearson r: {pearson_coef:.3f}")
        print(f"Non-sqrt metrics    - R²: {r2_squared:.3f}, RMSE: {rmse_squared:.3f}, Pearson r: {pearson_coef_squared:.3f}")
    
    # Create a safe filename version of the target class
    safe_filename = target_class.replace('/', '_')
    
    # Save the model
    model_filepath = os.path.join(output_dir, f"{safe_filename}_rf.joblib")
    joblib.dump({
        "model": rf,
        "feature_names": feature_names,
        "target_name": target_class,
        "rf_params": rf_params,
        "nonsqrt_metrics": nonsqrt_metrics
    }, model_filepath)
    
    # Create individual plot for this model
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    # Use generic labels
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    
    title = f"{target_class}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, r = {pearson_coef:.3f}"
    
    # Add non-sqrt metrics to title if available
    if nonsqrt_metrics:
        title += f"\nNon-sqrt R² = {nonsqrt_metrics['r2']:.3f}, RMSE = {nonsqrt_metrics['rmse']:.3f}, r = {nonsqrt_metrics['pearson']:.3f}"
    
    plt.title(title)
    plt.grid(alpha=0.3)
    
    regression_plot_path = os.path.join(output_dir, f"{safe_filename}_regression.png")
    plt.savefig(regression_plot_path)
    plt.close()
    
    # If it's a sqrt_ class, also create a plot with squared values
    if nonsqrt_metrics:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_squared, y_pred_squared, alpha=0.5, s=10)
        max_val = max(np.max(y_test_squared), np.max(y_pred_squared))
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        plt.xlabel("Actual value (squared)")
        plt.ylabel("Predicted value (squared)")
        
        plt.title(f"{target_class} (Non-sqrt values)\nR² = {nonsqrt_metrics['r2']:.3f}, RMSE = {nonsqrt_metrics['rmse']:.3f}, r = {nonsqrt_metrics['pearson']:.3f}")
        plt.grid(alpha=0.3)
        
        nonsqrt_plot_path = os.path.join(output_dir, f"{safe_filename}_nonsqrt_regression.png")
        plt.savefig(nonsqrt_plot_path)
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
        
        feature_importance_path = os.path.join(output_dir, f"{safe_filename}_feature_importance.png")
        plt.savefig(feature_importance_path)
        plt.close()
    
    # Return results
    results = {
        'model': rf,
        'model_path': model_filepath,
        'target_class': target_class,
        'r2': r2,
        'rmse': rmse,
        'pearson': pearson_coef,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': feature_names,
        'regression_plot': regression_plot_path,
        'nonsqrt_metrics': nonsqrt_metrics
    }
    
    print(f"Regression complete for {target_class}")
    
    return results

def run_all_regressions(data_dir, output_dir, wap_number=32, use_peat=False, superresolution=True, classes=None, rf_params_list=None, moy5m=False):
    """
    Run individual regressions for each target class specified in the classes list
    
    This function now uses prepare_regression_data and run_one_regression
    
    Args:
        data_dir: Directory containing the balanced data files
        output_dir: Directory to save the output
        wap_number: WAP site number
        use_peat: Whether to use peat-masked data
        superresolution: Whether to use 5m (True) or 10m (False) resolution data
        classes: List of target classes to perform regression on
        rf_params_list: Optional list of dictionaries with RF parameters for each class 
                       (if provided, must match the length of classes)
        moy5m: Whether to use moy5m data
    """
    print(f"Starting regression analysis for {len(classes)} individual classes: {classes}")
    
    # Prepare data once for all regressions
    prepared_data = prepare_regression_data(
        data_dir=data_dir,
        wap_number=wap_number,
        use_peat=use_peat,
        superresolution=superresolution,
        moy5m=moy5m
    )
    
    # Check if RF parameters are provided
    if rf_params_list is not None:
        if len(rf_params_list) != len(classes):
            print(f"Warning: Number of RF parameter sets ({len(rf_params_list)}) doesn't match number of classes ({len(classes)})")
            print("Using default RF parameters for all classes")
            rf_params_list = None
    
    # Results storage
    all_results = {}
    
    # Run regression for each target class
    for i, target_class in enumerate(classes):
        # Get RF parameters for this class if provided
        if rf_params_list is not None:
            rf_params = rf_params_list[i]
            print(f"Using custom RF parameters for {target_class}: {rf_params}")
        else:
            # Use default RF parameters
            rf_params = {
                "n_estimators": 200,
                "min_samples_leaf": 1,
                "min_samples_split": 10,
                "random_state": 42,
                "max_depth": 10,
                "max_features": "sqrt",
                "n_jobs": -1
            }
        
        # Run regression for this target class with specified parameters
        results = run_one_regression(
            prepared_data=prepared_data,
            output_dir=output_dir,
            target_class=target_class,
            rf_params=rf_params
        )
        
        if results is not None:
            all_results[target_class] = results
    
    # Create performance summary across all classes
    if all_results:
        create_comparative_plots(all_results, output_dir)
    
    return all_results

def create_comparative_plots(all_results, output_dir):
    """
    Create comparative plots across all processed classes
    
    Args:
        all_results: Dictionary with results for all classes
        output_dir: Directory to save output
    """
    # Create performance metrics for comparison
    metrics_data = []
    sqrt_classes = []
    
    # Collect metrics for each model
    for target_class, results in all_results.items():
        metrics_entry = {
            'Target': target_class,
            'R²': results['r2'],
            'RMSE': results['rmse'],
            'Pearson r': results['pearson']
        }
        
        # Add non-sqrt metrics if available
        if results.get('nonsqrt_metrics') is not None:
            metrics_entry['Non-sqrt R²'] = results['nonsqrt_metrics']['r2']
            metrics_entry['Non-sqrt RMSE'] = results['nonsqrt_metrics']['rmse']
            metrics_entry['Non-sqrt Pearson r'] = results['nonsqrt_metrics']['pearson']
            sqrt_classes.append(target_class)
        
        metrics_data.append(metrics_entry)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save as CSV
    metrics_df.to_csv(os.path.join(output_dir, "performance_metrics_summary.csv"), index=False)
    
    # Sort by R² (for consistent ordering in both plots)
    metrics_df_sorted = metrics_df.sort_values('R²', ascending=False)
    consistent_order = metrics_df_sorted['Target'].tolist()
    
    # Create a bar plot comparing R² values
    plt.figure(figsize=(12, 7))
    
    # Use consistent order for both plots
    bars = plt.bar(consistent_order, metrics_df_sorted['R²'], color='steelblue')
    
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
    
    # Create RMSE comparison using the same order as R² plot
    plt.figure(figsize=(12, 7))
    
    # Reindex the dataframe to match the consistent order
    metrics_for_rmse = metrics_df.set_index('Target').loc[consistent_order].reset_index()
    
    # Create bars with the same order as R² plot
    bars = plt.bar(metrics_for_rmse['Target'], metrics_for_rmse['RMSE'], color='lightcoral')
    
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
    
    # Create comparison plot for sqrt classes (if any)
    if sqrt_classes:
        # Filter metrics for sqrt classes and extract target name without 'sqrt_' prefix
        sqrt_metrics = []
        for _, row in metrics_df.iterrows():
            target = row['Target']
            if target in sqrt_classes:
                sqrt_metrics.append({
                    'Target': target.replace('sqrt_', ''),
                    'R² (sqrt)': row['R²'],
                    'R² (non-sqrt)': row['Non-sqrt R²']
                })
        
        sqrt_df = pd.DataFrame(sqrt_metrics)
        sqrt_df_sorted = sqrt_df.sort_values('R² (sqrt)', ascending=False)
        
        # Create comparison bar plot
        plt.figure(figsize=(12, 7))
        
        # Set up bar positions
        bar_width = 0.35
        x = np.arange(len(sqrt_df_sorted))
        
        # Plot bars
        plt.bar(x - bar_width/2, sqrt_df_sorted['R² (sqrt)'], bar_width, label='R² with sqrt transformation', color='steelblue')
        plt.bar(x + bar_width/2, sqrt_df_sorted['R² (non-sqrt)'], bar_width, label='R² on original scale', color='lightcoral')
        
        # Add labels and title
        plt.xlabel('Target Class')
        plt.ylabel('R² Score')
        plt.title('Comparison of R² Scores: sqrt transformation vs. original scale')
        plt.xticks(x, sqrt_df_sorted['Target'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (r2_sqrt, r2_nonsqrt) in enumerate(zip(sqrt_df_sorted['R² (sqrt)'], sqrt_df_sorted['R² (non-sqrt)'])):
            plt.text(i - bar_width/2, r2_sqrt + 0.01, f'{r2_sqrt:.3f}', ha='center', va='bottom')
            plt.text(i + bar_width/2, r2_nonsqrt + 0.01, f'{r2_nonsqrt:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sqrt_vs_nonsqrt_comparison.png"), dpi=300)
        plt.close()
    
    print(f"Created comparative performance plots in {output_dir}")

def cross_validation(data_dir, output_dir, wap_number=32, use_peat=False, superresolution=True, classes=None, rf_params_list=None, moy5m=False, n_folds=5):
    """
    Run cross-validation regression for each target class using n_folds cross-validation
    
    Args:
        data_dir: Directory containing the balanced data files
        output_dir: Directory to save the output
        wap_number: WAP site number
        use_peat: Whether to use peat-masked data
        superresolution: Whether to use 5m (True) or 10m (False) resolution data
        classes: List of target classes to perform regression on
        rf_params_list: Optional list of dictionaries with RF parameters for each class
        moy5m: Whether to use moy5m data
        n_folds: Number of CV folds (default: 5)
        
    Returns:
        Dictionary containing cross-validation results for each class
    """
    print(f"Starting {n_folds}-fold cross-validation for {len(classes)} individual classes: {classes}")
    
    # Create cross-validation output directory
    cv_output_dir = os.path.join(output_dir, f"cv_{n_folds}_folds")
    os.makedirs(cv_output_dir, exist_ok=True)
    
    # Prepare data once for all regressions
    prepared_data = prepare_regression_data(
        data_dir=data_dir,
        wap_number=wap_number,
        use_peat=use_peat,
        superresolution=superresolution,
        moy5m=moy5m
    )
    
    # Check if RF parameters are provided
    if rf_params_list is not None:
        if len(rf_params_list) != len(classes):
            print(f"Warning: Number of RF parameter sets ({len(rf_params_list)}) doesn't match number of classes ({len(classes)})")
            print("Using default RF parameters for all classes")
            rf_params_list = None
    
    # Storage for results
    cv_results = {}
    
    # Storage for metrics across folds
    r2_metrics = {cls: [] for cls in classes}
    rmse_metrics = {cls: [] for cls in classes}
    pearson_metrics = {cls: [] for cls in classes}
    
    # For non-sqrt metrics
    nonsqrt_r2_metrics = {cls: [] for cls in classes if cls.startswith('sqrt_')}
    nonsqrt_rmse_metrics = {cls: [] for cls in classes if cls.startswith('sqrt_')}
    nonsqrt_pearson_metrics = {cls: [] for cls in classes if cls.startswith('sqrt_')}
    
    # Results for CSV tables
    csv_r2_data = {}
    csv_rmse_data = {}
    csv_pearson_data = {}
    
    # Process each target class
    for class_idx, target_class in enumerate(classes):
        print(f"\n{'='*80}\nProcessing {target_class} with {n_folds}-fold cross-validation")
        
        # Get RF parameters for this class if provided
        if rf_params_list is not None:
            rf_params = rf_params_list[class_idx]
            print(f"Using custom RF parameters for {target_class}: {rf_params}")
        else:
            # Use default RF parameters
            rf_params = {
                "n_estimators": 200,
                "min_samples_leaf": 1,
                "min_samples_split": 10,
                "random_state": 42,
                "max_depth": 10,
                "max_features": "sqrt",
                "n_jobs": -1
            }
            
        # Create filename for the CSV
        csv_file = f"balanced_{target_class}.csv"
        csv_path = os.path.join(data_dir, csv_file)
            
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_path} does not exist, skipping {target_class}")
            continue
            
        # Prepare data for this class
        X, y, target_found = prepare_data_for_regression(
            csv_path=csv_path,
            sentinel_features=prepared_data["sentinel_features"],
            feature_names=prepared_data["feature_names"],
            target_class=target_class
        )
            
        if not target_found or X is None:
            print(f"Warning: Target class '{target_class}' not found or data preparation failed, skipping.")
            continue
        
        # Initialize metrics for this class
        class_r2_values = []
        class_rmse_values = []
        class_pearson_values = []
        class_y_test_all = []
        class_y_pred_all = []
        
        # For sqrt classes
        class_nonsqrt_r2_values = []
        class_nonsqrt_rmse_values = []
        class_nonsqrt_pearson_values = []
        class_nonsqrt_y_test_all = []
        class_nonsqrt_y_pred_all = []
        
        # Initialize model for feature importance (will be averaged)
        feature_importances = np.zeros(len(prepared_data["feature_names"]))
        
        # Create CV splitter
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Run cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"  Processing fold {fold+1}/{n_folds}")
            
            # Split data for this fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            rf = RandomForestRegressor(**rf_params)
            rf.fit(X_train, y_train)
            
            # Predict
            y_pred = rf.predict(X_test)
            
            # Calculate metrics for this fold
            fold_r2 = r2_score(y_test, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            fold_pearson, _ = pearsonr(y_test, y_pred)
            
            # Store metrics
            class_r2_values.append(fold_r2)
            class_rmse_values.append(fold_rmse)
            class_pearson_values.append(fold_pearson)
            
            # Store predictions and true values
            class_y_test_all.extend(y_test.tolist())
            class_y_pred_all.extend(y_pred.tolist())
            
            # Add to feature importances (for averaging later)
            feature_importances += rf.feature_importances_
            
            # Store for CSV output
            csv_r2_data.setdefault(target_class, {})[f"fold_{fold+1}"] = fold_r2
            csv_rmse_data.setdefault(target_class, {})[f"fold_{fold+1}"] = fold_rmse
            csv_pearson_data.setdefault(target_class, {})[f"fold_{fold+1}"] = fold_pearson
            
            # For sqrt classes, also calculate metrics on original scale
            if target_class.startswith("sqrt_"):
                # Calculate metrics for squared values
                y_test_squared = y_test ** 2
                y_pred_squared = y_pred ** 2
                
                # Calculate metrics
                fold_nonsqrt_r2 = r2_score(y_test_squared, y_pred_squared)
                fold_nonsqrt_rmse = np.sqrt(mean_squared_error(y_test_squared, y_pred_squared))
                fold_nonsqrt_pearson, _ = pearsonr(y_test_squared, y_pred_squared)
                
                # Store metrics
                class_nonsqrt_r2_values.append(fold_nonsqrt_r2)
                class_nonsqrt_rmse_values.append(fold_nonsqrt_rmse)
                class_nonsqrt_pearson_values.append(fold_nonsqrt_pearson)
                
                # Store for plotting later
                class_nonsqrt_y_test_all.extend(y_test_squared.tolist())
                class_nonsqrt_y_pred_all.extend(y_pred_squared.tolist())
        
        # Calculate average metrics across folds
        avg_r2 = np.mean(class_r2_values)
        avg_rmse = np.mean(class_rmse_values)
        avg_pearson = np.mean(class_pearson_values)
        
        # Add means to CSV data
        csv_r2_data[target_class]["mean"] = avg_r2
        csv_rmse_data[target_class]["mean"] = avg_rmse
        csv_pearson_data[target_class]["mean"] = avg_pearson
        
        # Store CV metrics
        r2_metrics[target_class] = class_r2_values
        rmse_metrics[target_class] = class_rmse_values
        pearson_metrics[target_class] = class_pearson_values
        
        # For sqrt classes, store non-sqrt metrics
        nonsqrt_metrics = None
        if target_class.startswith("sqrt_"):
            avg_nonsqrt_r2 = np.mean(class_nonsqrt_r2_values)
            avg_nonsqrt_rmse = np.mean(class_nonsqrt_rmse_values)
            avg_nonsqrt_pearson = np.mean(class_nonsqrt_pearson_values)
            
            nonsqrt_r2_metrics[target_class] = class_nonsqrt_r2_values
            nonsqrt_rmse_metrics[target_class] = class_nonsqrt_rmse_values
            nonsqrt_pearson_metrics[target_class] = class_nonsqrt_pearson_values
            
            nonsqrt_metrics = {
                "r2_values": class_nonsqrt_r2_values,
                "rmse_values": class_nonsqrt_rmse_values,
                "pearson_values": class_nonsqrt_pearson_values,
                "avg_r2": avg_nonsqrt_r2,
                "avg_rmse": avg_nonsqrt_rmse,
                "avg_pearson": avg_nonsqrt_pearson,
                "y_test": class_nonsqrt_y_test_all,
                "y_pred": class_nonsqrt_y_pred_all
            }
            
            print(f"  Cross-validation metrics (sqrt scale): R²={avg_r2:.3f}±{np.std(class_r2_values):.3f}, RMSE={avg_rmse:.3f}±{np.std(class_rmse_values):.3f}")
            print(f"  Cross-validation metrics (original scale): R²={avg_nonsqrt_r2:.3f}±{np.std(class_nonsqrt_r2_values):.3f}, RMSE={avg_nonsqrt_rmse:.3f}±{np.std(class_nonsqrt_rmse_values):.3f}")
        else:
            print(f"  Cross-validation metrics: R²={avg_r2:.3f}±{np.std(class_r2_values):.3f}, RMSE={avg_rmse:.3f}±{np.std(class_rmse_values):.3f}")
        
        # Average feature importances
        feature_importances /= n_folds
        
        # Create a safe filename version of the target class
        safe_filename = target_class.replace('/', '_')
        
        # Create scatter plot of predictions vs. actual values using all folds
        plt.figure(figsize=(8, 8))
        plt.scatter(class_y_test_all, class_y_pred_all, alpha=0.5, s=10)
        max_val = max(np.max(class_y_test_all), np.max(class_y_pred_all))
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        
        # Create title with metrics
        title = f"{target_class} (CV)\nAvg R²: {avg_r2:.3f}, Avg RMSE: {avg_rmse:.3f}, Avg Pearson r: {avg_pearson:.3f}"
        if nonsqrt_metrics:
            title += f"\nNon-sqrt Avg R²: {avg_nonsqrt_r2:.3f}, Avg RMSE: {avg_nonsqrt_rmse:.3f}"
            
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(cv_output_dir, f"{safe_filename}_cv_regression.png"))
        plt.close()
        
        # Also create non-sqrt plot if applicable
        if nonsqrt_metrics:
            plt.figure(figsize=(8, 8))
            plt.scatter(class_nonsqrt_y_test_all, class_nonsqrt_y_pred_all, alpha=0.5, s=10)
            max_val = max(np.max(class_nonsqrt_y_test_all), np.max(class_nonsqrt_y_pred_all))
            plt.plot([0, max_val], [0, max_val], 'r--')
            plt.xlabel("Actual value (original scale)")
            plt.ylabel("Predicted value (original scale)")
            plt.title(f"{target_class} (CV, original scale)\nAvg R²: {avg_nonsqrt_r2:.3f}, Avg RMSE: {avg_nonsqrt_rmse:.3f}, Avg Pearson r: {avg_nonsqrt_pearson:.3f}")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(cv_output_dir, f"{safe_filename}_cv_nonsqrt_regression.png"))
            plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sorted_idx = np.argsort(feature_importances)[::-1]
        top_n = min(20, len(sorted_idx))
        top_idx = sorted_idx[:top_n]
        plt.barh(range(top_n), feature_importances[top_idx])
        plt.yticks(range(top_n), [prepared_data["feature_names"][i] for i in top_idx])
        plt.title(f"Average Feature Importance for {target_class} ({n_folds}-fold CV)")
        plt.tight_layout()
        plt.savefig(os.path.join(cv_output_dir, f"{safe_filename}_cv_feature_importance.png"))
        plt.close()
        
        # Store results for this class
        cv_results[target_class] = {
            "r2_values": class_r2_values,
            "rmse_values": class_rmse_values,
            "pearson_values": class_pearson_values,
            "avg_r2": avg_r2,
            "avg_rmse": avg_rmse,
            "avg_pearson": avg_pearson,
            "y_test": class_y_test_all,
            "y_pred": class_y_pred_all,
            "feature_importances": feature_importances,
            "nonsqrt_metrics": nonsqrt_metrics
        }
    
    # Create and save the summary CSV tables
    # First, convert the dictionaries to dataframes
    fold_cols = [f"fold_{i+1}" for i in range(n_folds)] + ["mean"]
    
    # R2 Table
    r2_df = pd.DataFrame.from_dict(csv_r2_data, orient='index')
    r2_df = r2_df.reindex(columns=fold_cols)  # Reorder columns
    r2_df.to_csv(os.path.join(cv_output_dir, "r2_by_fold.csv"))
    
    # RMSE Table
    rmse_df = pd.DataFrame.from_dict(csv_rmse_data, orient='index')
    rmse_df = rmse_df.reindex(columns=fold_cols)  # Reorder columns
    rmse_df.to_csv(os.path.join(cv_output_dir, "rmse_by_fold.csv"))
    
    # Pearson Table
    pearson_df = pd.DataFrame.from_dict(csv_pearson_data, orient='index')
    pearson_df = pearson_df.reindex(columns=fold_cols)  # Reorder columns
    pearson_df.to_csv(os.path.join(cv_output_dir, "pearson_by_fold.csv"))
    
    # Create a combined CSV with mean metrics
    combined_metrics = []
    for target_class in classes:
        if target_class in cv_results:
            entry = {
                'Target': target_class,
                'Avg R²': cv_results[target_class]['avg_r2'],
                'Std R²': np.std(cv_results[target_class]['r2_values']),
                'Avg RMSE': cv_results[target_class]['avg_rmse'],
                'Std RMSE': np.std(cv_results[target_class]['rmse_values']),
                'Avg Pearson r': cv_results[target_class]['avg_pearson'],
                'Std Pearson r': np.std(cv_results[target_class]['pearson_values'])
            }
            
            # Add non-sqrt metrics if available
            if cv_results[target_class]['nonsqrt_metrics']:
                entry['Avg Non-sqrt R²'] = cv_results[target_class]['nonsqrt_metrics']['avg_r2']
                entry['Std Non-sqrt R²'] = np.std(cv_results[target_class]['nonsqrt_metrics']['r2_values'])
                entry['Avg Non-sqrt RMSE'] = cv_results[target_class]['nonsqrt_metrics']['avg_rmse']
                entry['Std Non-sqrt RMSE'] = np.std(cv_results[target_class]['nonsqrt_metrics']['rmse_values'])
                entry['Avg Non-sqrt Pearson r'] = cv_results[target_class]['nonsqrt_metrics']['avg_pearson']
                entry['Std Non-sqrt Pearson r'] = np.std(cv_results[target_class]['nonsqrt_metrics']['pearson_values'])
            
            combined_metrics.append(entry)
    
    if combined_metrics:
        combined_df = pd.DataFrame(combined_metrics)
        combined_df.to_csv(os.path.join(cv_output_dir, "combined_cv_metrics.csv"), index=False)
    
    # Create comparative plots based on the CV results
    if len(cv_results) > 0:
        # Sort targets by average R² for consistent ordering across all plots
        sorted_targets = sorted(cv_results.keys(), key=lambda x: cv_results[x]['avg_r2'], reverse=True)
        
        # Create position indices for the bars
        x = np.arange(len(sorted_targets))
        
        # ---------- 1. R² comparison plot for sqrt values ----------
        plt.figure(figsize=(12, 7))
        
        # Plot R² bars with error bars
        r2_values = [cv_results[target]['avg_r2'] for target in sorted_targets]
        r2_errors = [np.std(cv_results[target]['r2_values']) for target in sorted_targets]
        
        bars = plt.bar(x, r2_values, yerr=r2_errors, capsize=5, color='steelblue')
        
        # Add labels and title
        plt.xlabel('Target Class')
        plt.ylabel('R² Score')
        plt.title(f'Model Performance (sqrt-transformed R²) with {n_folds}-fold CV')
        plt.xticks(x, sorted_targets, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val, err in zip(bars, r2_values, r2_errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{val:.3f}±{err:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cv_output_dir, "cv_sqrt_r2_summary.png"), dpi=300)
        plt.close()
        
        # ---------- 2. RMSE comparison plot for sqrt values ----------
        plt.figure(figsize=(12, 7))
        
        # Use the same order as R² plot
        rmse_values = [cv_results[target]['avg_rmse'] for target in sorted_targets]
        rmse_errors = [np.std(cv_results[target]['rmse_values']) for target in sorted_targets]
        
        bars = plt.bar(x, rmse_values, yerr=rmse_errors, capsize=5, color='lightcoral')
        
        # Add labels and title
        plt.xlabel('Target Class')
        plt.ylabel('RMSE')
        plt.title(f'Model Performance (sqrt-transformed RMSE) with {n_folds}-fold CV')
        plt.xticks(x, sorted_targets, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val, err in zip(bars, rmse_values, rmse_errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{val:.3f}±{err:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cv_output_dir, "cv_sqrt_rmse_summary.png"), dpi=300)
        plt.close()
        
        # ---------- 3. Pearson R comparison plot for sqrt values ----------
        plt.figure(figsize=(12, 7))
        
        # Use the same order as R² plot
        pearson_values = [cv_results[target]['avg_pearson'] for target in sorted_targets]
        pearson_errors = [np.std(cv_results[target]['pearson_values']) for target in sorted_targets]
        
        bars = plt.bar(x, pearson_values, yerr=pearson_errors, capsize=5, color='mediumseagreen')
        
        # Add labels and title
        plt.xlabel('Target Class')
        plt.ylabel('Pearson r')
        plt.title(f'Model Performance (sqrt-transformed Pearson r) with {n_folds}-fold CV')
        plt.xticks(x, sorted_targets, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val, err in zip(bars, pearson_values, pearson_errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{val:.3f}±{err:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cv_output_dir, "cv_sqrt_pearson_summary.png"), dpi=300)
        plt.close()
        
        # Now create plots for the non-sqrt (original scale) metrics
        # Only include classes with sqrt_ prefix that have nonsqrt metrics
        sqrt_classes = [c for c in sorted_targets if c.startswith('sqrt_') and cv_results[c]['nonsqrt_metrics']]
        
        if sqrt_classes:
            # Create x positions for the bars
            x_nonsqrt = np.arange(len(sqrt_classes))
            
            # ---------- 4. R² comparison plot for non-sqrt values (original scale) ----------
            plt.figure(figsize=(12, 7))
            
            # Get values and errors for original scale (squared values)
            nonsqrt_r2_values = [cv_results[target]['nonsqrt_metrics']['avg_r2'] for target in sqrt_classes]
            nonsqrt_r2_errors = [np.std(cv_results[target]['nonsqrt_metrics']['r2_values']) for target in sqrt_classes]
            
            bars = plt.bar(x_nonsqrt, nonsqrt_r2_values, yerr=nonsqrt_r2_errors, capsize=5, color='steelblue')
            
            # Add labels and title
            plt.xlabel('Target Class')
            plt.ylabel('R² Score')
            plt.title(f'Model Performance (original scale R²) with {n_folds}-fold CV')
            plt.xticks(x_nonsqrt, [t.replace('sqrt_', '') for t in sqrt_classes], rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val, err in zip(bars, nonsqrt_r2_values, nonsqrt_r2_errors):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{val:.3f}±{err:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(cv_output_dir, "cv_original_r2_summary.png"), dpi=300)
            plt.close()
            
            # ---------- 5. RMSE comparison plot for non-sqrt values (original scale) ----------
            plt.figure(figsize=(12, 7))
            
            # Get values and errors for original scale
            nonsqrt_rmse_values = [cv_results[target]['nonsqrt_metrics']['avg_rmse'] for target in sqrt_classes]
            nonsqrt_rmse_errors = [np.std(cv_results[target]['nonsqrt_metrics']['rmse_values']) for target in sqrt_classes]
            
            bars = plt.bar(x_nonsqrt, nonsqrt_rmse_values, yerr=nonsqrt_rmse_errors, capsize=5, color='lightcoral')
            
            # Add labels and title
            plt.xlabel('Target Class')
            plt.ylabel('RMSE')
            plt.title(f'Model Performance (original scale RMSE) with {n_folds}-fold CV')
            plt.xticks(x_nonsqrt, [t.replace('sqrt_', '') for t in sqrt_classes], rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val, err in zip(bars, nonsqrt_rmse_values, nonsqrt_rmse_errors):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{val:.3f}±{err:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(cv_output_dir, "cv_original_rmse_summary.png"), dpi=300)
            plt.close()
            
            # ---------- 6. Pearson R comparison plot for non-sqrt values (original scale) ----------
            plt.figure(figsize=(12, 7))
            
            # Get values and errors for original scale
            nonsqrt_pearson_values = [cv_results[target]['nonsqrt_metrics']['avg_pearson'] for target in sqrt_classes]
            nonsqrt_pearson_errors = [np.std(cv_results[target]['nonsqrt_metrics']['pearson_values']) for target in sqrt_classes]
            
            bars = plt.bar(x_nonsqrt, nonsqrt_pearson_values, yerr=nonsqrt_pearson_errors, capsize=5, color='mediumseagreen')
            
            # Add labels and title
            plt.xlabel('Target Class')
            plt.ylabel('Pearson r')
            plt.title(f'Model Performance (original scale Pearson r) with {n_folds}-fold CV')
            plt.xticks(x_nonsqrt, [t.replace('sqrt_', '') for t in sqrt_classes], rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val, err in zip(bars, nonsqrt_pearson_values, nonsqrt_pearson_errors):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{val:.3f}±{err:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(cv_output_dir, "cv_original_pearson_summary.png"), dpi=300)
            plt.close()
            
            # ---------- 7. Direct sqrt vs. non-sqrt R² comparison plot ----------
            plt.figure(figsize=(12, 7))
            
            # Set up bar positions
            bar_width = 0.35
            
            # Extract R² values for only sqrt classes in the same order
            sqrt_r2_values = [cv_results[target]['avg_r2'] for target in sqrt_classes]
            sqrt_r2_errors = [np.std(cv_results[target]['r2_values']) for target in sqrt_classes]
            
            # Plot bars
            plt.bar(x_nonsqrt - bar_width/2, sqrt_r2_values, bar_width, 
                    yerr=sqrt_r2_errors, capsize=5, 
                    label='R² with sqrt transformation', color='steelblue')
            
            plt.bar(x_nonsqrt + bar_width/2, nonsqrt_r2_values, bar_width, 
                    yerr=nonsqrt_r2_errors, capsize=5,
                    label='R² on original scale', color='lightcoral')
            
            # Add labels and title
            plt.xlabel('Target Class')
            plt.ylabel('R² Score')
            plt.title(f'Comparison of R² Scores: sqrt vs. original scale ({n_folds}-fold CV)')
            plt.xticks(x_nonsqrt, [t.replace('sqrt_', '') for t in sqrt_classes], rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(cv_output_dir, "cv_sqrt_vs_nonsqrt_comparison.png"), dpi=300)
            plt.close()

    print(f"\nCross-validation completed successfully!")
    print(f"Results saved to {cv_output_dir}")
    
    return cv_results


if __name__ == "__main__":
    # Parameters
    wap = 32
    use_peat = False
    superresolution = False  # Use 5m resolution (True) or 10m resolution (False)
    moy5m = False
    peat_suffix = "_peat" if use_peat else ""
    resolution_suffix = "_5m" if superresolution else "_10m"
    moy5m_suffix = "_moy5m" if moy5m else ""

    # Define paths
    data_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/balanced"
    output_dir = f"data/regressions/regression_merged_model/regression_wap{wap}{peat_suffix}{resolution_suffix}{moy5m_suffix}/regression_results"
    
    # Example 1: Run all regressions with default parameters
    classes = ["sqrt_Pure_Lichen", "sqrt_Degraded_Lichen", "sqrt_Green", "sqrt_all_lichen", "sqrt_through_proportion"]
    
    # run_all_regressions(
    #     data_dir=data_dir,
    #     output_dir=output_dir,
    #     wap_number=wap,
    #     use_peat=use_peat,
    #     superresolution=superresolution,
    #     classes=classes,
    #     moy5m=moy5m
    # )

    # Example 2: Run all regressions with custom parameters for each class
    rf_params_list = [
        {  # Parameters for sqrt_Pure_Lichen
            'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200,
            "n_jobs": -1
        },
        {  # Parameters for sqrt_Degraded_Lichen
            'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100,
            "n_jobs": -1
        },
        {  # Parameters for sqrt_Green
            'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100,
            "n_jobs": -1
        },
        {  # Parameters for sqrt_all_lichen
            'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200,
            "n_jobs": -1
        },
        { # Parameters for sqrt_through_proportion
            'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100,
            "n_jobs": -1
        }
    ]
    
    # run_all_regressions(
    #     data_dir=data_dir,
    #     output_dir=output_dir,
    #     wap_number=wap,
    #     use_peat=use_peat,
    #     superresolution=superresolution,
    #     classes=classes,
    #     rf_params_list=rf_params_list,  # Pass custom parameters for each class
    #     moy5m=moy5m
    # )
    
    # Example 3: Prepare data once and run a single regression
    # prepared_data = prepare_regression_data(
    #     data_dir=data_dir,
    #     wap_number=wap,
    #     use_peat=use_peat,
    #     superresolution=superresolution,
    #     moy5m=moy5m
    # )
    
    # custom_rf_params = {
    #     "n_estimators": 300,
    #     "min_samples_leaf": 2,
    #     "min_samples_split": 5,
    #     "max_depth": 15,
    #     "max_features": "sqrt",
    #     "n_jobs": -1
    # }
    
    # results = run_one_regression(
    #     prepared_data=prepared_data,
    #     output_dir=output_dir,
    #     target_class="sqrt_Pure_Lichen",
    #     rf_params=custom_rf_params
    # )
    
    # Example 4: Run cross-validation
    cv_results = cross_validation(
        data_dir=data_dir,
        output_dir=output_dir,
        wap_number=wap,
        use_peat=use_peat,
        superresolution=superresolution,
        classes=classes,
        rf_params_list=rf_params_list,  # Pass custom parameters for each class
        moy5m=moy5m,
        n_folds=5  # Default is 5 folds
    )
    
    
    
    print("\nRegression analysis completed successfully!")
    print(f"Results saved to {output_dir}")

