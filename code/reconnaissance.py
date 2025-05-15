import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

def load_samples(json_path):
    with open(json_path) as f:
        data = json.load(f)
    samples = data["samples"]
    features = []
    labels = []
    for s in samples:
        feat = [
            s["r_mean"], s["g_mean"], s["b_mean"],
            s["r_n_mean"], s["g_n_mean"], s["b_n_mean"],
            s["r_var"], s["g_var"], s["b_var"],
            s["z_var"],
            s["delta_z_x"], s["delta_z_y"]
        ]
        if None in feat or s["category"] is None:
            continue
        features.append(feat)
        labels.append(s["category"])
    return np.array(features), np.array(labels)

def train_random_forest(features, labels, test_size=0.3, random_state=42, n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return clf, X_test, y_test

def plot_decision_tree(clf, feature_names=None, class_names=None, tree_idx=0, max_depth=3):
    estimator = clf.estimators_[tree_idx]
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        max_depth=max_depth,
        fontsize=10
    )
    plt.title(f"Arbre de décision n°{tree_idx} du Random Forest")
    plt.show()

import joblib

if __name__ == "__main__":
    features, labels = load_samples("data/samples/lichen_sphegnes_balanced/lichen_sphegnes_balanced.json")
    clf, X_test, y_test = train_random_forest(features, labels, n_estimators=100, random_state=42)
    # Noms des features pour l'affichage
    feature_names = [
        "r_mean", "g_mean", "b_mean",
        "r_n_mean", "g_n_mean", "b_n_mean",
        "r_var", "g_var", "b_var",
        "z_var", "delta_z_x", "delta_z_y"
    ]
    class_names = np.unique(labels).astype(str)
    plot_decision_tree(clf, feature_names=feature_names, class_names=class_names, tree_idx=0, max_depth=3)

    # Sauvegarde du modèle
    joblib.dump(clf, "data/samples/lichen_sphegnes_balanced/random_forest_model.joblib")
    #clf = joblib.load("random_forest_model.joblib")