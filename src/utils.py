# src/utils.py
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from .config import MODELS_DIR


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def train_test_split_xy(
    df: pd.DataFrame,
    target_col: str,
    test_size=0.2,
    random_state=42,
    stratify=True
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If regression OR only one value in y → do NOT stratify
    if not stratify or y.nunique() <= 1:
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
    else:
        # Classification case with multiple classes → stratify
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )


def fit_scaler(X_train, X_test, scaler_name: str = "scaler.pkl"):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ensure_dir(MODELS_DIR)
    joblib.dump(scaler, MODELS_DIR / scaler_name)
    return X_train_s, X_test_s


def load_scaler(scaler_name: str = "scaler.pkl"):
    return joblib.load(MODELS_DIR / scaler_name)