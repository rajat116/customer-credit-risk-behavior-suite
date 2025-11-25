# src/train_unsupervised.py
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from .config import DATA_PROCESSED, MODELS_DIR
from .utils import load_csv, load_scaler
import json


def main():
    df = load_csv(DATA_PROCESSED)

    # Load feature columns used by PD model
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Select and clean features
    X = df[feature_cols].copy()

    # ---- FIX: impute missing values ----
    X = X.fillna(X.median())

    # Load scaler from PD model and transform
    scaler = load_scaler("scaler_pd.pkl")
    X_s = scaler.transform(X)

    # Unsupervised models
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_s)

    isolation = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    isolation.fit(X_s)

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_segmentation.pkl")
    joblib.dump(isolation, MODELS_DIR / "iforest_behavior.pkl")

    print("Saved KMeans and IsolationForest models.")


if __name__ == "__main__":
    main()