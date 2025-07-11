#!/usr/bin/env python3
"""
Train Classification Model

This script trains a Random Forest classification model on sample data from JSON files.
It can optionally perform grid search to find optimal parameters.

Usage:
  python train_classification_model.py input.json output_path [--grid-search]

Example:
  python train_classification_model.py merged_samples.json ./model_output --grid-search
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from datetime import datetime
import time
import seaborn as sns
from collections import Counter

# Define default parameters


def parse_arguments():
    """
    Parse command line arguments for the classification model training script.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train a Random Forest classification model on sample data")
    parser.add_argument("input_json", help="Path to the filtered JSON file with samples")
    parser.add_argument("output_path", help="Directory to save the model and reports")
    parser.add_argument("--grid-search", action="store_true", help="Perform grid search to find best parameters")
    
    return parser.parse_args()

def get_available_features(samples, exclude_temp=True):
    """
    Detect available features in the sample data.
    
    Args:
        samples: List of sample dictionaries
        exclude_temp: Whether to exclude temperature-related features
        
    Returns:
        List of available feature names
    """
    # List of all possible features, in order
    all_features = [
        "r_mean", "g_mean", "b_mean",
        "r_var", "g_var", "b_var",
        "r_n_mean", "g_n_mean", "b_n_mean",
        "t_mean", "t_n_mean", "t_var",
        "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean",
        "z_mean", "z_var", "z_moins_z_n", "z_moins_z_large"
    ]
    
    # If excluding temperature features
    if exclude_temp:
        temp_features = ["t_mean", "t_n_mean", "t_var", "t_large_mean"]
        all_features = [f for f in all_features if f not in temp_features]
    
    # Find the first non-None sample to detect available features
    for s in samples:
        present = [f for f in all_features if f in s and s[f] is not None]
        # We assume all samples have the same features
        return present
    
    return []

def load_samples(json_path, exclude_temp=True):
    """
    Load sample data from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        exclude_temp: Whether to exclude temperature-related features
        
    Returns:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        feature_names: List of feature names
        class_counts: Dictionary counting samples per class
    """
    print(f"Loading samples from {json_path}...")
    
    with open(json_path) as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    print(f"Loaded {len(samples)} samples from JSON file")
    
    feature_names = get_available_features(samples, exclude_temp)
    print(f"Detected {len(feature_names)} features: {', '.join(feature_names)}")
    
    features = []
    labels = []
    for s in samples:
        feat = []
        skip = False
        for f in feature_names:
            val = s.get(f, None)
            if val is None:
                skip = True
                break
            feat.append(val)
        
        if skip or s.get("category", None) is None:
            continue
        
        features.append(feat)
        labels.append(s["category"])
    
    class_counts = Counter(labels)
    print(f"Extracted {len(features)} valid samples with {len(feature_names)} features")
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} samples")
    
    return np.array(features), np.array(labels), feature_names, class_counts

def train_with_grid_search(X_train, y_train, feature_names):
    """
    Train a Random Forest classifier using grid search to find optimal parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        
    Returns:
        best_model: Best trained model
        best_params: Best parameters found
    """
    print("\nPerforming grid search to find optimal parameters...")
    
    # Base parameters that remain constant
    base_params = {
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create base classifier
    base_clf = RandomForestClassifier(**base_params)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=PARAM_GRID,
        scoring='f1_weighted',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Start timer
    start_time = time.time()
    
    # Perform grid search
    print(f"Evaluating {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_leaf']) * len(PARAM_GRID['min_samples_split'])} parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    # End timer
    elapsed_time = time.time() - start_time
    
    # Get results
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    
    # Add base parameters to best parameters
    for key, value in base_params.items():
        best_params[key] = value
    
    print(f"Grid search completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score (weighted F1): {best_score:.4f}")
    
    return best_model, best_params

def train_with_default_params(X_train, y_train):
    """
    Train a Random Forest classifier using default parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        clf: Trained classifier
        params: Parameters used for training
    """
    print("\nTraining with default parameters...")
    
    # Create and train classifier
    clf = RandomForestClassifier(**DEFAULT_PARAMS)
    clf.fit(X_train, y_train)
    
    return clf, DEFAULT_PARAMS

def create_confusion_matrix_plot(y_true, y_pred, class_names, output_path):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_feature_importance_plot(model, feature_names, output_path):
    """
    Create and save a feature importance visualization.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_stats_report(model, params, X_train, y_train, X_test, y_test, 
                       feature_names, class_counts, output_path):
    """
    Create a comprehensive statistics report for the trained model.
    
    Args:
        model: Trained model
        params: Model parameters
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: List of feature names
        class_counts: Counter with class distribution
        output_path: Path to save the report
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Create report file
    report_path = os.path.join(output_path, "model_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Classification Model Report\n")
        f.write("=========================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset information
        f.write("Dataset Information\n")
        f.write("------------------\n")
        f.write(f"Total samples: {len(X_train) + len(X_test)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        # Class distribution
        f.write("Class Distribution\n")
        f.write("-----------------\n")
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count} samples\n")
        f.write("\n")
        
        # Features
        f.write("Features\n")
        f.write("--------\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write(f"Features: {', '.join(feature_names)}\n\n")
        
        # Model parameters
        f.write("Model Parameters\n")
        f.write("---------------\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        # Model performance
        f.write("Model Performance\n")
        f.write("----------------\n")
        f.write(f"Training accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")
        f.write(f"Training F1 score (weighted): {train_f1:.4f}\n")
        f.write(f"Test F1 score (weighted): {test_f1:.4f}\n\n")
        
        # Classification report
        f.write("Classification Report (Test Set)\n")
        f.write("-----------------------------\n")
        f.write(classification_report(y_test, y_test_pred))
        f.write("\n")
        
        # Confusion matrix
        f.write("Confusion Matrix (Test Set)\n")
        f.write("-------------------------\n")
        cm = confusion_matrix(y_test, y_test_pred)
        class_names = np.unique(np.concatenate((y_train, y_test)))
        
        # Format confusion matrix
        cm_str = ""
        cm_str += " " * 15
        for cls in class_names:
            cm_str += f"{cls:>10} "
        cm_str += "\n"
        
        for i, cls in enumerate(class_names):
            cm_str += f"{cls:>15} "
            for j in range(len(class_names)):
                cm_str += f"{cm[i, j]:>10d} "
            cm_str += "\n"
        
        f.write(cm_str)
        f.write("\n")
        
        # Feature importance
        f.write("Feature Importance\n")
        f.write("-----------------\n")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in indices:
            f.write(f"{feature_names[i]}: {importances[i]:.4f}\n")
    
    print(f"Statistics report saved to {report_path}")
    
    # Create confusion matrix visualization
    create_confusion_matrix_plot(
        y_test, y_test_pred, 
        class_names, 
        os.path.join(output_path, "confusion_matrix.png")
    )
    
    # Create feature importance visualization
    create_feature_importance_plot(
        model, 
        feature_names, 
        os.path.join(output_path, "feature_importance.png")
    )

def main():
    """Main function for training the classification model."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting classification model training")
    print(f"{'='*80}")
    
    # Load samples
    features, labels, feature_names, class_counts = load_samples(args.input_json)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples")
    
    # Train model
    if args.grid_search:
        model, params = train_with_grid_search(X_train, y_train, feature_names)
    else:
        model, params = train_with_default_params(X_train, y_train)
    
    # Create statistics report
    create_stats_report(
        model, params, 
        X_train, y_train, 
        X_test, y_test, 
        feature_names, class_counts, 
        args.output_path
    )
    
    # Save model
    model_path = os.path.join(args.output_path, "classification_model.joblib")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "parameters": params,
        "class_names": list(class_counts.keys())
    }, model_path)
    
    print(f"Model saved to {model_path}")
    print(f"Training completed successfully!")

DEFAULT_PARAMS = {
    'n_estimators': 300,
    'max_depth': 200, 
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_leaf': 3,
    'min_samples_split': 4,
    'max_features': 'sqrt'
}

# Define grid search parameters
PARAM_GRID = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, None],
    'min_samples_leaf': [2, 3, 5, 8],
    'min_samples_split': [3, 4, 5, 10]
}

if __name__ == "__main__":
    main()

"""
Example usage:

python code/final_codes/classification/create_classification_model/train_classification_model.py \
       data/samples/selection14/merged_no_chicoutai.json \
        data/selection_test/model_output

python code/final_codes/classification/create_classification_model/train_classification_model.py \
    data/samples/selection14/merged_no_chicoutai.json\
    data/selection_test/model_output_grid \
    --grid-search
"""
