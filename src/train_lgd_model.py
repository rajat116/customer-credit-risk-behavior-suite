# src/train_lgd_model.py
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from .config import DATA_PROCESSED, MODELS_DIR
from .utils import load_csv, train_test_split_xy, fit_scaler, save_json


def main():
    df = load_csv(DATA_PROCESSED)

    # Create synthetic loss_amount: only non-zero for defaulted customers
    rng = np.random.default_rng(42)
    df["loss_amount"] = df["default"] * (
        1000 + 4000 * rng.random(len(df))
    )

    # Filter to defaulted ones for LGD modeling
    df_lgd = df[df["default"] == 1].copy()

    target_col = "loss_amount"
    X_train, X_test, y_train, y_test = train_test_split_xy(
        df_lgd,
        target_col=target_col,
        stratify=False
    )

    feature_cols = list(X_train.columns)
    save_json(feature_cols, MODELS_DIR / "feature_cols_lgd.json")

    X_train_s, X_test_s = fit_scaler(X_train, X_test, scaler_name="scaler_lgd.pkl")

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"LGD RMSE: {rmse:.2f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lgd_model.pkl")
    print("Saved LGD model to models/lgd_model.pkl")


if __name__ == "__main__":
    main()