"""
Train Regression Model

This script trains a Random Forest regression model on balanced proportion data.
It's the final step in the regression model creation pipeline and includes:
- Automatic detection of the target variable from the input data
- Optional grid search for parameter optimization
- Cross-validation to evaluate model performance
- Site-specific performance evaluation
- Feature importance analysis
- Comprehensive visualization and reporting

Usage:
  python train_regression_model.py input.json output_path [--grid-search]

Example:
  python train_regression_model.py balanced_lichen_proportion.json ./results --grid-search
"""
import os
import json
import argparse
from unicodedata import category
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import joblib
from scipy.stats import pearsonr
from datetime import datetime
import time

def parse_arguments():
    """
    Parse command line arguments for the regression model training script.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train a Random Forest regression model on balanced proportion data")
    parser.add_argument("input_json", help="Path to the balanced JSON file")
    parser.add_argument("output_path", help="Directory to save the outputs")
    parser.add_argument("--grid-search", action="store_true", help="Perform grid search to find best parameters")
    
    return parser.parse_args()

def load_json_data(json_path):
    """
    Load proportion data from a JSON file.
    
    Args:
        json_path: Path to the input JSON file
        
    Returns:
        List of dictionaries containing features and proportions
    """
    print(f"Loading data from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from JSON file")
    return data

def detect_category(json_data):
    """
    Automatically detect which proportion category to predict from the JSON data.
    
    This allows the script to work with different proportion types (lichen, green, trough)
    without requiring manual specification.
    
    Args:
        json_data: List of dictionaries with features and proportions
        
    Returns:
        category: Target category name (e.g., 'lichen_proportion')
    """
    # Find first item with proportions
    category = None
    
    for item in json_data:
        if 'proportions' in item and isinstance(item['proportions'], dict):
            categories = list(item['proportions'].keys())
            if len(categories) > 0:
                category = categories[0]
                if len(categories) > 1:
                    print(f"WARNING: Multiple categories found in proportions: {categories}")
                    print(f"Using first category: {category}")
                break
    
    if category is None:
        raise ValueError("No valid category found in the JSON data. Check that your data has a 'proportions' field.")
    
    return category

def extract_features_and_target(json_data, category):
    """
    Extract features and target values from JSON data.
    
    This function:
    1. Extracts feature values and target proportions
    2. Creates consistent feature ordering
    3. Identifies and counts samples from different sources (sites)
    
    Args:
        json_data: List of dictionaries with features and proportions
        category: Target category to predict (e.g., 'lichen_proportion')
        
    Returns:
        X: Features array
        y: Target values array
        feature_names: List of feature names
        sources: List of source names for each sample
        site_names: List of unique site names
        source_counts: Dictionary with counts per source
    """
    # Extract features, targets and sources
    X_list = []
    y_list = []
    sources = []
    feature_names = None
    
    # Count samples by source
    source_counts = {}
    
    for item in json_data:
        # Skip items without features or proportions
        if 'features' not in item or 'proportions' not in item or category not in item['proportions']:
            continue
        
        # Extract features
        features = item['features']
        
        # Initialize feature_names from first item
        if feature_names is None:
            feature_names = list(features.keys())
        
        # Extract feature values in consistent order
        feature_values = [features.get(name, 0) for name in feature_names]
        X_list.append(feature_values)
        
        # Extract target proportion
        y_list.append(item['proportions'][category])
        
        # Extract source (site name)
        source = item.get('source', 'unknown')
        sources.append(source)
        
        # Count samples by source
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Check if we have any valid data
    if len(X_list) == 0:
        raise ValueError(f"No valid samples found with category '{category}'. Check your JSON data format.")
    
    # Convert lists to arrays
    X = np.array(X_list)
    y = np.array(y_list)
    sources = np.array(sources)
    
    # Get unique site names
    site_names = np.unique(sources)
    
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target category: {category}")
    print(f"Found {len(site_names)} different sites: {', '.join(site_names)}")
    
    # Print samples per site
    for site in site_names:
        count = source_counts.get(site, 0)
        print(f"  {site}: {count} samples ({count/len(sources)*100:.1f}%)")
    
    return X, y, feature_names, sources, site_names, source_counts

def train_with_grid_search(X, y, feature_names):
    """
    Train a Random Forest regressor using grid search to find optimal parameters.
    
    This function:
    1. Splits data into train (70%) and validation (30%) sets
    2. Tests different parameter combinations
    3. Finds the combination that yields the best performance
    4. Returns the trained model with the best parameters
    
    Args:
        X: Features array
        y: Target array
        feature_names: List of feature names
        
    Returns:
        rf: Trained RandomForest model with best parameters
        best_params: Best parameters found
        feature_importance: Feature importance values
    """
    print("\nPerforming grid search to find best parameters using 70/30 split...")
    
    # # Parameter grid
   
    param_grid = PARAM_GRID
        
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Perform our own grid search using 70/30 split
    best_score = -np.inf
    best_params = None
    best_model = None
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    
    print(f"Evaluating {total_combinations} parameter combinations...")
    start_time = time.time()
    
    # Track all results to report top combinations
    all_results = []
    
    # Generate all parameter combinations
    import itertools
    keys = param_grid.keys()
    param_combinations = itertools.product(*param_grid.values())
    
    # Evaluate each combination
    for i, params in enumerate(param_combinations, 1):
        # Create parameter dictionary
        param_dict = dict(zip(keys, params))
        
        # Train model with these parameters
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **param_dict)
        rf.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = rf.score(X_val, y_val)
        
        # Store result
        all_results.append((val_score, param_dict))
        
        # Update best if better
        if val_score > best_score:
            best_score = val_score
            best_params = param_dict
            best_model = rf
        
        # Progress update
        if i % 10 == 0 or i == total_combinations:
            elapsed = time.time() - start_time
            print(f"Evaluated {i}/{total_combinations} combinations ({i/total_combinations*100:.1f}%) in {elapsed:.1f}s")
    
    search_time = time.time() - start_time
    
    # Sort results by score (descending) - using only the first element (score) for sorting
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    # Display top 3 results
    print("\nTop 3 parameter combinations:")
    for i, (score, params) in enumerate(all_results[:3], 1):
        print(f"{i}. R² = {score:.4f}, Parameters: {params}")
    
    print(f"\nGrid search completed in {search_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best validation R² score: {best_score:.4f}")
    
    return best_model, best_params, best_model.feature_importances_

def train_with_default_params(X, y):
    """
    Train a Random Forest regressor using default parameters.
    
    Used when grid search is not requested or for the final model training
    after parameters are determined.
    
    Args:
        X: Features array
        y: Target array
        
    Returns:
        rf: Trained RandomForest model
        params: Parameters used
        feature_importance: Feature importance values
    """
    print("\nTraining Random Forest model with default parameters...")
    
    # Default parameters
    params = DEFAULT_PARAMS 
    
    # Create and train the model
    rf = RandomForestRegressor(**params)
    rf.fit(X, y)
    
    return rf, params, rf.feature_importances_

def perform_cross_validation(X, y, rf_params, sources):
    """
    Perform 5-fold cross-validation and track performance by site.
    
    This function:
    1. Splits data into 5 folds
    2. Trains and evaluates on each fold
    3. Collects predictions and sources for site-specific evaluation
    4. Calculates performance metrics with standard deviations
    
    Args:
        X: Features array
        y: Target array
        rf_params: RandomForest parameters
        sources: Array of source names for each sample
        
    Returns:
        cv_results: Dictionary with cross-validation results
    """
    print("\nPerforming 5-fold cross-validation...")
    
    # Create CV splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for metrics
    r2_values = []
    rmse_values = []
    pearson_values = []
    y_test_all = []
    y_pred_all = []
    sources_test_all = []  # Track the source for each test sample
    feature_importances = np.zeros(X.shape[1])
    
    # Run cross-validation
    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"  Processing fold {fold}/5")
        
        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sources_test = sources[test_idx]  # Get sources for test samples
        
        # Train model
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pearson, _ = pearsonr(y_test, y_pred)
        
        # Store metrics
        r2_values.append(r2)
        rmse_values.append(rmse)
        pearson_values.append(pearson)
        
        # Store predictions and sources for later site-specific analysis
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
        sources_test_all.extend(sources_test)
        
        # Accumulate feature importance
        feature_importances += rf.feature_importances_
        
        fold += 1
    
    # Calculate average metrics
    avg_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    avg_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    avg_pearson = np.mean(pearson_values)
    std_pearson = np.std(pearson_values)
    
    # Average feature importance
    feature_importances /= 5
    
    # Convert lists to arrays for plotting
    y_test_all = np.array(y_test_all)
    y_pred_all = np.array(y_pred_all)
    sources_test_all = np.array(sources_test_all)
    
    print(f"Cross-validation results:")
    print(f"  R²: {avg_r2:.3f} ± {std_r2:.3f}")
    print(f"  RMSE: {avg_rmse:.3f} ± {std_rmse:.3f}")
    print(f"  Pearson r: {avg_pearson:.3f} ± {std_pearson:.3f}")
    
    # Return results
    cv_results = {
        'r2_values': r2_values,
        'rmse_values': rmse_values,
        'pearson_values': pearson_values,
        'avg_r2': avg_r2,
        'std_r2': std_r2,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'avg_pearson': avg_pearson,
        'std_pearson': std_pearson,
        'y_test': y_test_all,
        'y_pred': y_pred_all,
        'sources_test': sources_test_all,
        'feature_importances': feature_importances
    }
    
    return cv_results

def compute_site_metrics_from_cv(cv_results, site_names):
    """
    Compute site-specific metrics from cross-validation results.
    
    This provides insight into how the model performs on different geographic sites,
    helping identify potential biases or weaknesses.
    
    Args:
        cv_results: Dictionary with cross-validation results
        site_names: List of unique site names
        
    Returns:
        site_metrics: Dictionary with metrics for each site based on CV predictions
    """
    print("\nCalculating site-specific metrics from cross-validation results:")
    
    y_test = cv_results['y_test']
    y_pred = cv_results['y_pred']
    sources = cv_results['sources_test']
    
    site_metrics = {}
    
    for site in site_names:
        # Get indices for this site
        site_indices = np.where(sources == site)[0]
        
        # Skip if not enough samples
        if len(site_indices) < 10:
            print(f"  Skipping {site}: not enough test samples ({len(site_indices)})")
            continue
        
        # Extract site data
        y_site_true = y_test[site_indices]
        y_site_pred = y_pred[site_indices]
        
        # Calculate metrics
        r2 = r2_score(y_site_true, y_site_pred)
        rmse = np.sqrt(mean_squared_error(y_site_true, y_site_pred))
        pearson_coef, _ = pearsonr(y_site_true, y_site_pred)
        
        print(f"  {site}: R²={r2:.3f}, RMSE={rmse:.3f}, r={pearson_coef:.3f}")
        
        # Store metrics
        site_metrics[site] = {
            'r2': r2,
            'rmse': rmse,
            'pearson': pearson_coef,
            'y_true': y_site_true,
            'y_pred': y_site_pred,
            'n_samples': len(y_site_true)
        }
    
    return site_metrics

def plot_regression_results(y_true, y_pred, output_path, title=None, metrics=None):
    """
    Create scatter plot comparing predicted vs. actual values.
    
    This visualization helps assess the model's accuracy and identify any
    systematic errors or biases.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        output_path: Path to save the plot
        title: Plot title (optional)
        metrics: Dictionary with metrics to include in title (optional)
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=2)
    
    # Add identity line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Set labels
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    
    # Create title
    if title is None:
        title = "Regression Results"
    
    # Add metrics to title if provided
    if metrics:
        # Handle both regular metrics and cross-validation metrics (which use 'avg_' prefix)
        r2_key = 'avg_r2' if 'avg_r2' in metrics else 'r2'
        rmse_key = 'avg_rmse' if 'avg_rmse' in metrics else 'rmse'
        pearson_key = 'avg_pearson' if 'avg_pearson' in metrics else 'pearson'
        
        if r2_key in metrics:
            title += f"\nR² = {metrics[r2_key]:.3f}"
        if rmse_key in metrics:
            title += f", RMSE = {metrics[rmse_key]:.3f}"
        if pearson_key in metrics:
            title += f", r = {metrics[pearson_key]:.3f}"
        
        # Add standard deviations if provided (for cross-validation)
        if 'std_r2' in metrics and r2_key in metrics:
            title = title.replace(f"R² = {metrics[r2_key]:.3f}", 
                                 f"R² = {metrics[r2_key]:.3f} ± {metrics['std_r2']:.3f}")
        if 'std_rmse' in metrics and rmse_key in metrics:
            title = title.replace(f"RMSE = {metrics[rmse_key]:.3f}", 
                                 f"RMSE = {metrics[rmse_key]:.3f} ± {metrics['std_rmse']:.3f}")
        if 'std_pearson' in metrics and pearson_key in metrics:
            title = title.replace(f"r = {metrics[pearson_key]:.3f}", 
                                 f"r = {metrics[pearson_key]:.3f} ± {metrics['std_pearson']:.3f}")
    
    plt.title(title)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(feature_importance, feature_names, output_path, top_n=20):
    """
    Create bar plot of feature importance.
    
    This visualization helps identify which features (bands or indices)
    are most useful for predicting the target variable.
    
    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names
        output_path: Path to save the plot
        top_n: Number of top features to show (default: 20)
    """
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    # Limit to top N features
    top_idx = sorted_idx[:min(top_n, len(sorted_idx))]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_idx)), feature_importance[top_idx])
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def plot_site_comparisons(site_metrics, output_path, category_name, is_cv=False):
    """
    Create a grid of scatter plots showing model performance for each site.
    
    This visualization helps identify differences in model performance
    across different geographic locations.
    
    Args:
        site_metrics: Dictionary with metrics for each site
        output_path: Path to save the plot
        category_name: Name of the target category
        is_cv: Whether metrics are from cross-validation (default: False)
    """
    # Determine grid size based on number of sites
    n_sites = len(site_metrics)
    if n_sites == 0:
        return
    
    n_cols = min(3, n_sites)
    n_rows = (n_sites + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    
    # Handle single subplot case
    if n_sites == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create a scatter plot for each site
    for i, (site_name, metrics) in enumerate(site_metrics.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=3.5)
        
        # Add identity line
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Set labels
        ax.set_xlabel("Actual value")
        ax.set_ylabel("Predicted value")
        
        # Set title with metrics
        title = f"{site_name} (n={metrics['n_samples']})\n"
        title += f"R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}, r = {metrics['pearson']:.3f}"
        ax.set_title(title)
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_sites, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Add overall title
    cv_text = "(Cross-Validation)" if is_cv else ""
    fig.suptitle(f"Per-site performance for {category_name} {cv_text}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save and close
    plt.savefig(output_path)
    plt.close()

def create_stats_report(category, n_samples, site_counts, metrics, model_params, output_path, site_metrics=None, cv_results=None):
    """
    Create a comprehensive text report with statistics and model information.
    
    This report summarizes:
    1. Dataset composition and sample distribution
    2. Model parameters
    3. Overall performance metrics
    4. Site-specific performance metrics
    
    Args:
        category: Target category name
        n_samples: Total number of samples
        site_counts: Dictionary with counts per source
        metrics: Dictionary with model metrics
        model_params: Dictionary with model parameters
        output_path: Path to save the report
        site_metrics: Dictionary with per-site metrics (optional)
        cv_results: Dictionary with cross-validation results (optional)
    """
    with open(output_path, 'w') as f:
        # Write header with timestamp
        f.write(f"Regression Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Dataset information
        f.write(f"Target Category: {category}\n")
        f.write(f"Total Samples: {n_samples}\n\n")
        
        # Sample counts per site
        f.write("Sample Distribution:\n")
        for site, count in site_counts.items():
            percentage = count / n_samples * 100
            f.write(f"  {site}: {count} samples ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Model parameters
        f.write("Random Forest Parameters:\n")
        for param, value in model_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        # Overall performance metrics
        f.write("Overall Model Performance:\n")
        if cv_results:
            # Cross-validation results
            f.write(f"  [Cross-Validation Results, 5 folds]\n")
            f.write(f"  R²: {cv_results['avg_r2']:.4f} ± {cv_results['std_r2']:.4f}\n")
            f.write(f"  RMSE: {cv_results['avg_rmse']:.4f} ± {cv_results['std_rmse']:.4f}\n")
            f.write(f"  Pearson r: {cv_results['avg_pearson']:.4f} ± {cv_results['std_pearson']:.4f}\n")
        else:
            # Simple train/test split results
            f.write(f"  [70/30 Train/Test Split]\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"  Pearson r: {metrics['pearson']:.4f}\n")
        f.write("\n")
        
        # Per-site performance if available
        if site_metrics:
            f.write("Performance by Site:\n")
            for site, site_metric in site_metrics.items():
                f.write(f"  {site} (n={site_metric['n_samples']}):\n")
                f.write(f"    R²: {site_metric['r2']:.4f}\n")
                f.write(f"    RMSE: {site_metric['rmse']:.4f}\n")
                f.write(f"    Pearson r: {site_metric['pearson']:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("End of Report\n")

def main():
    """
    Main function orchestrating the regression model training process.
    
    Workflow:
    1. Load and prepare data
    2. Train model (with grid search if requested)
    3. Perform cross-validation
    4. Evaluate site-specific performance
    5. Generate visualizations
    6. Create statistics report
    7. Save the final model
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting regression analysis")
    print(f"{'='*80}")
    
    # Load JSON data
    json_data = load_json_data(args.input_json)
    
    # Automatically detect category from data
    category = detect_category(json_data)
    
    # Extract features and target
    X, y, feature_names, sources, site_names, site_counts = extract_features_and_target(json_data, category)
    
    # Train model based on specified options
    if args.grid_search:
        # Train with grid search (70/30 split)
        rf, model_params, feature_importance = train_with_grid_search(X, y, feature_names)
    else:
        # Train with default parameters on all data
        rf, model_params, feature_importance = train_with_default_params(X, y)
    
    # Perform cross-validation with the selected parameters
    cv_results = perform_cross_validation(X, y, model_params, sources)
    
    # Plot CV results for all sites combined
    plot_regression_results(
        cv_results['y_test'], 
        cv_results['y_pred'],
        os.path.join(args.output_path, f"{category}_cv_regression.png"),
        title=f"{category} (Cross-Validation)",
        metrics=cv_results
    )
    
    # Calculate site-specific metrics from cross-validation results
    site_metrics_cv = compute_site_metrics_from_cv(cv_results, site_names)
    
    # Plot per-site comparisons based on cross-validation
    plot_site_comparisons(
        site_metrics_cv,
        os.path.join(args.output_path, f"{category}_site_comparison_cv.png"),
        category,
        is_cv=True
    )
    
    # Use CV feature importance
    feature_importance = cv_results['feature_importances']
    
    # Plot feature importance
    plot_feature_importance(
        feature_importance,
        feature_names,
        os.path.join(args.output_path, f"{category}_feature_importance.png")
    )
    
    # Train final model on all data (for deployment purposes)
    rf, _, _ = train_with_default_params(X, y) if not args.grid_search else (rf, model_params, feature_importance)
    
    # Save model
    model_path = os.path.join(args.output_path, f"{category}_model.joblib")
    joblib.dump({
        "model": rf,
        "feature_names": feature_names,
        "target_name": category,
        "parameters": model_params
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Create statistics report
    stats_path = os.path.join(args.output_path, f"{category}_stats.txt")
    create_stats_report(
        category=category,
        n_samples=len(y),
        site_counts=site_counts,
        metrics=cv_results,
        model_params=model_params,
        output_path=stats_path,
        site_metrics=site_metrics_cv,  # Use CV-based site metrics
        cv_results=cv_results
    )
    print(f"Statistics report saved to {stats_path}")
    
    print(f"\nRegression analysis for {category} completed successfully!")



PARAM_GRID = {
#         'n_estimators': [100],               # 100 for speed, 300 for stability
#         'max_depth': [15],               # None = no limit, also try controlled depths
#         'min_samples_split': [2, 5],               # 2 is default, 5-10 to limit overfitting
#         'min_samples_leaf': [2, 4],                 # more leaves = less overfitting
#         'max_features': ['sqrt', 'log2']          # sqrt or log2 to reduce complexity, 0.8 for wider testing
#     }
#  # param_grid = {
        'n_estimators': [200, 300],               # 100 for speed, 300 for stability
        'max_depth': [15, 30, 50],               # None = no limit, also try controlled depths
        'min_samples_split': [2, 5, 10],               # 2 is default, 5-10 to limit overfitting
        'min_samples_leaf': [2, 4],                 # more leaves = less overfitting
        'max_features': ['sqrt', 'log2', 0.8]          # sqrt or log2 to reduce complexity, 0.8 for wider testing
    }

DEFAULT_PARAMS = {
        'n_estimators': 300, 
        'max_depth': 30, 
        'min_samples_split': 2, 
        'min_samples_leaf': 2, 
        'max_features': 0.8,
        'random_state': 42,
        'n_jobs': -1
}

if __name__ == "__main__":
    main()
"""
python code/final_codes/regression/create_regression_model/train_regression_model.py \
    data/regressions/regression_multisite/balanced_6/balanced_lichen_proportion.json \
    data/regressions/regression_multisite/results_8/lichen \
   --grid-search 

python code/final_codes/regression/create_regression_model/train_regression_model.py \
    data/regressions/regression_multisite/balanced_6/balanced_trough_proportion.json \
    data/regressions/regression_multisite/results_8/trough \
   --grid-search

"""


