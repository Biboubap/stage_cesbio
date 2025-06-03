import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def get_available_features(samples, exclude_temp=False):
    """
    Détecte dynamiquement les features disponibles dans le JSON.
    Retourne la liste des features utilisables.
    
    Args:
        samples: Liste des échantillons
        exclude_temp: Si True, exclut les features liées à la température
    """
    # Liste de toutes les features possibles, dans l'ordre
    all_features = [
        "r_mean", "g_mean", "b_mean",
        "r_var", "g_var", "b_var",
        "r_n_mean", "g_n_mean", "b_n_mean",
        "t_mean", "t_n_mean", "t_var",
        "r_large_mean", "g_large_mean", "b_large_mean", "t_large_mean",
        "z_var", "z_moins_z_n", "z_moins_z_large"
    ]
    
    # Si on exclut les features de température
    if exclude_temp:
        temp_features = ["t_mean", "t_n_mean", "t_var", "t_large_mean"]
        all_features = [f for f in all_features if f not in temp_features]
    
    # Prend le premier sample non None pour détecter les features présentes
    for s in samples:
        present = [f for f in all_features if f in s and s[f] is not None]
        # On considère qu'une feature est utilisable si elle est présente dans ce sample
        # (on suppose que tous les samples sont cohérents)
        return present
    return []

def load_samples(json_path, exclude_temp=False):
    with open(json_path) as f:
        data = json.load(f)
    samples = data["samples"]
    feature_names = get_available_features(samples, exclude_temp)
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
    return np.array(features), np.array(labels), feature_names

def train_random_forest(features, labels, feature_names, test_size=0.2, random_state=42):
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
        min_samples_split=3,
        max_features="sqrt"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    # Affiche les features utilisées
    print("Features utilisées :", feature_names)
    return clf, X_test, y_test, feature_names

if __name__ == "__main__":
    features, labels, feature_names = load_samples("data/samples/selection11/pop_merged/pop_merged.json", exclude_temp=True)
    clf, X_test, y_test, feature_names = train_random_forest(features, labels, feature_names)

    # Sauvegarde du modèle et des features utilisées
    import joblib
    joblib.dump({"model": clf, "feature_names": feature_names}, "data/samples/selection11/pop_merged/model_pp_ld_fo.joblib")







def grid_search_rf(features, labels, feature_names, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [1, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20, 40]
    }
    base_params = dict(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )
    rf = RandomForestClassifier(**base_params)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    y_pred = grid.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Test classification report:\n", classification_report(y_test, y_pred))
    print("Features utilisées :", feature_names)
    return grid.best_estimator_, X_test, y_test, feature_names

# if __name__ == "__main__":
#     features, labels = load_samples("data/samples/selection8/merged_pop/merged_pop8.json")
#     clf, X_test, y_test = grid_search_rf(features, labels)
#     # Sauvegarde du meilleur modèle
#     # import joblib
#     # joblib.dump(clf, "data/samples/selection3/random_forest_best_model5.joblib")

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score

# # Chargement des données
# features, labels = load_samples("data/samples/selection3/pop3_merged.json")

# # Définition du modèle (avec tes paramètres)
# clf = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=15,
#     class_weight="balanced",
#     random_state=42,
#     n_jobs=-1,
#     min_samples_leaf=3,
#     min_samples_split=6,
#     max_features="sqrt"
# )

# # Validation croisée (ici 5 folds)
# scores = cross_val_score(clf, features, labels, cv=5, scoring='f1_weighted', n_jobs=-1)
# print("F1-score (validation croisée, 5 folds) :", scores)
# print("Moyenne F1-score :", np.mean(scores))
# print("Écart-type F1-score :", np.std(scores))