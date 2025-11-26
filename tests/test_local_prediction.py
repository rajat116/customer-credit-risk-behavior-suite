import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import joblib
import pandas as pd

from src.utils import load_scaler
from src.config import MODELS_DIR

# -------------------------------------------------
# Load feature list (same for PD + LGD)
# -------------------------------------------------
with open(MODELS_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

# -------------------------------------------------
# Example input (10 features)
# -------------------------------------------------
example = np.array([[0.35, 42, 1, 0.25, 5500, 6, 0, 1, 0, 2]])

#example = np.array([[0.98, 22, 3, 1.00, 1500, 12, 4, 0, 2, 3]])

# Convert NUMPY → DATAFRAME WITH VALID COLUMN NAMES
example_df = pd.DataFrame(example, columns=feature_cols)

# -------------------------------------------------
# PD MODEL
# -------------------------------------------------
pd_scaler = load_scaler("scaler_pd.pkl")
pd_model = joblib.load(MODELS_DIR / "pd_model.pkl")

pd_scaled = pd_scaler.transform(example_df)          # DF → array (OK)
pd_pred = pd_model.predict_proba(pd_scaled)[:, 1]
print("PD:", pd_pred)

# -------------------------------------------------
# LGD MODEL
# -------------------------------------------------
lgd_scaler = load_scaler("scaler_lgd.pkl")
lgd_model = joblib.load(MODELS_DIR / "lgd_model.pkl")

# IMPORTANT FIX: use example_df, not numpy array
lgd_scaled = lgd_scaler.transform(example_df)
lgd_scaled = lgd_scaler.transform(example_df)
lgd_scaled_df = pd.DataFrame(lgd_scaled, columns=feature_cols)
lgd_pred = lgd_model.predict(lgd_scaled_df)
print("LGD:", lgd_pred)

# -------------------------------------------------
# Expected Loss
# -------------------------------------------------
print("Expected Loss:", pd_pred * lgd_pred)

# -------------------------------------------------
# Clustering
# -------------------------------------------------
kmeans = joblib.load(MODELS_DIR / "kmeans_segmentation.pkl")
cluster = kmeans.predict(pd_scaled)
print("Cluster:", cluster)

# -------------------------------------------------
# Isolation Forest
# -------------------------------------------------
isf = joblib.load(MODELS_DIR / "iforest_behavior.pkl")
anomaly = isf.decision_function(pd_scaled)
print("Anomaly:", anomaly)