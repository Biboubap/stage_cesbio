import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Charger les samples
with open("data/samples/lichen_sphegnes_balanced/lichen_sphegnes_balanced.json") as f:
    data = json.load(f)

samples = data["samples"]

# 2. Extraire features et labels
features = []
labels = []
for s in samples:
    # Exemple: on prend les couleurs moyennes et les pentes
    feat = [
        s["r_mean"], s["g_mean"], s["b_mean"],
        s["r_n_mean"], s["g_n_mean"], s["b_n_mean"],
        s["r_var"], s["g_var"], s["b_var"],
        s["z_var"],
        s["delta_z_x"], s["delta_z_y"]
    ]
    if None in feat or s["category"] is None:
        continue  # On saute les samples incomplets
    features.append(feat)
    labels.append(s["category"])

features = np.array(features)
labels = np.array(labels)

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 4. Entraînement
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Évaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Prédiction sur de nouveaux samples
# new_features = ... (array de shape [n_samples, n_features])
# predicted_categories = clf.predict(new_features)