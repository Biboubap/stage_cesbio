import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def load_samples(json_path):
    with open(json_path) as f:
        data = json.load(f)
    samples = data["samples"]
    features = []
    labels = []
    for s in samples:
        feat = [
            s["r_mean"], s["g_mean"], s["b_mean"],
            s["r_var"], s["g_var"], s["b_var"],
            s["r_n_mean"], s["g_n_mean"], s["b_n_mean"],
            s["t_mean"], s["t_n_mean"],
            s["z_var"], s["z_moins_z_n"]
        ]
        if None in feat or s["category"] is None:
            continue
        features.append(feat)
        labels.append(s["category"])
    return np.array(features), np.array(labels)

def train_random_forest(features, labels, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt"    # Nombre de features considérées à chaque split (standard pour RF)
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    return clf, X_test, y_test

if __name__ == "__main__":
    features, labels = load_samples("data/samples/selection3/pop3_merged.json")
    clf, X_test, y_test = train_random_forest(features, labels)
    # Sauvegarde du modèle
    import joblib
    joblib.dump(clf, "data/samples/selection3/random_forest_model3.joblib")


# def grid_search_rf(features, labels, test_size=0.3, random_state=42):
#     X_train, X_test, y_train, y_test = train_test_split(
#         features, labels, test_size=test_size, random_state=random_state, stratify=labels
#     )
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [10, 15, 20],
#         'min_samples_leaf': [1, 5, 10, 20],
#         'min_samples_split': [2, 5, 10, 20, 40]
#     }
#     base_params = dict(
#         class_weight="balanced",
#         random_state=42,
#         n_jobs=-1,
#         max_features="sqrt"
#     )
#     rf = RandomForestClassifier(**base_params)
#     grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
#     grid.fit(X_train, y_train)
#     print("Best parameters:", grid.best_params_)
#     print("Best CV score:", grid.best_score_)
#     # Évaluation sur le test set
#     y_pred = grid.predict(X_test)
#     print("Test accuracy:", accuracy_score(y_test, y_pred))
#     print("Test classification report:\n", classification_report(y_test, y_pred))
#     return grid.best_estimator_, X_test, y_test

# if __name__ == "__main__":
#     features, labels = load_samples("data/samples/selection3/pop3_merged.json")
#     clf, X_test, y_test = grid_search_rf(features, labels)
#     # Sauvegarde du meilleur modèle
#     import joblib
#     joblib.dump(clf, "data/samples/selection3/random_forest_best_model.joblib")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Chargement des données
features, labels = load_samples("data/samples/selection3/pop3_merged.json")

# Définition du modèle (avec tes paramètres)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=3,
    min_samples_split=6,
    max_features="sqrt"
)

# Validation croisée (ici 5 folds)
scores = cross_val_score(clf, features, labels, cv=5, scoring='f1_weighted', n_jobs=-1)
print("F1-score (validation croisée, 5 folds) :", scores)
print("Moyenne F1-score :", np.mean(scores))
print("Écart-type F1-score :", np.std(scores))