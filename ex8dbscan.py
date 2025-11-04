#Cell 0: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')  # Authorize once
#Cell 1: Set Paths

import os

BASE_DIR = 'https://docs.google.com/document/d/1-c04kOHV3S9TFya4La-xZi7wdl31d5XQ/edit#heading=h.tvxuuvj86erp'  # change if you like
os.makedirs(BASE_DIR, exist_ok=True)

ARTIFACT_PATH = os.path.join(BASE_DIR, 'dbscan_artifacts.joblib')
print('Artifacts will be saved to:', ARTIFACT_PATH)
#Cell 2: Train & Save Artifacts to Drive
# --- TRAIN & SAVE (to Google Drive) ---
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

# Data: 3 blobs + 60 outliers
X, _ = make_blobs(n_samples=1500, centers=[[1,1],[5,5],[9,1]],
                  cluster_std=[0.35, 0.45, 0.4], random_state=42)
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-2, high=12, size=(60, 2))
X = np.vstack([X, outliers])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
eps = 0.25
min_samples = 8
db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
labels = db.fit_predict(X_scaled)

# Core samples + NN index for inference
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[db.core_sample_indices_] = True
core_points = X_scaled[core_mask]
core_labels = labels[core_mask]
nn = NearestNeighbors(n_neighbors=5).fit(core_points)

# Save all artifacts to Drive
artifacts = {
    "scaler": scaler,
    "dbscan": db,
    "eps": eps,
    "core_points": core_points,
    "core_labels": core_labels,
    "nn_index": nn
}
joblib.dump(artifacts, ARTIFACT_PATH)
import os, glob
print('Listing artifacts in:', BASE_DIR)
print(glob.glob(os.path.join(BASE_DIR, '*')))
import numpy as np
import joblib

def assign_label(new_points, artifact_path=ARTIFACT_PATH):
    """DBSCAN-style inference using saved core points in Drive."""
    art = joblib.load(artifact_path)
    scaler = art["scaler"]
    eps = art["eps"]
    nn = art["nn_index"]
    core_points = art["core_points"]
    core_labels = art["core_labels"]

    Z = scaler.transform(np.asarray(new_points))
    inds_list = nn.radius_neighbors(Z, radius=eps, return_distance=False)

    out = []
    for inds in inds_list:
        if len(inds) == 0:
            out.append(-1)  # anomaly
        else:
            lbls = core_labels[inds]
            vals, cnts = np.unique(lbls, return_counts=True)
            out.append(int(vals[np.argmax(cnts)]))
    return np.array(out)

# Try a few points
new_samples = np.array([
    [1.1, 0.9],
    [4.9, 5.2],
    [9.2, 1.1],
    [15.0, -3.0]  # likely anomaly
])
print("Assigned labels (-1 = anomaly):", assign_label(new_samples))
