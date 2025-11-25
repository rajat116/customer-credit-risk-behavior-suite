# service/model_loader.py
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.config import MODELS_DIR
from src.utils import load_scaler


@lru_cache()
def get_feature_cols():
    with open(MODELS_DIR / "feature_cols.json") as f:
        return json.load(f)


@lru_cache()
def get_pd_model():
    return joblib.load(MODELS_DIR / "pd_model.pkl")


@lru_cache()
def get_lgd_model():
    return joblib.load(MODELS_DIR / "lgd_model.pkl")


@lru_cache()
def get_kmeans():
    return joblib.load(MODELS_DIR / "kmeans_segmentation.pkl")


@lru_cache()
def get_iforest():
    return joblib.load(MODELS_DIR / "iforest_behavior.pkl")


@lru_cache()
def get_scaler():
    return load_scaler("scaler_pd.pkl")


def prepare_inputs(features_list):
    X = np.array(features_list, dtype=float).reshape(1, -1)
    scaler = get_scaler()
    X_s = scaler.transform(X)
    return X_s