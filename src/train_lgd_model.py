# src/train_lgd_model.py
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from .config import DATA_PROCESSED, MODELS_DIR
from .utils import load_csv, train_test_split_xy, fit_scaler, save_json


def main():
    df = load_csv(DATA_PROCESSED)

    # -----------------------------------------
    # 1. Load the same feature set as PD model
    # -----------------------------------------
    import json

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # -----------------------------------------
    # 2. Build a realistic, stable LGD target
    #    (bounded, non-negative, correlated with risk)
    # -----------------------------------------
    rng = np.random.default_rng(42)

    # Cap extreme values to avoid exploding targets
    util = df["util"].clip(0, 2.0)
    dr = df["DebtRatio"].clip(0, 5.0)
    income = df["MonthlyIncome"].clip(0, 20000.0)
    num_30_59 = df["num_30_59"].clip(0, 10)
    num_60_89 = df["num_60_89"].clip(0, 10)
    num_90 = df["num_90"].clip(0, 10)
    num_real_estate = df["num_real_estate"].clip(0, 5)
    num_credit_lines = df["num_credit_lines"].clip(0, 30)
    dependents = df["NumberOfDependents"].fillna(0).clip(0, 6)

    # Base LGD (in currency units)
    base = (
        1000.0
        + 2000.0 * util
        + 1500.0 * dr
        + 500.0 * (num_30_59 + num_60_89 + num_90)
        + 300.0 * num_real_estate
        + 200.0 * (num_credit_lines > 12).astype(float)
        + 400.0 * dependents
    )

    # Add some noise but keep it reasonable
    noise = rng.normal(0, 500.0, size=len(df))

    # Only defaulted customers have loss; others = 0
    df["loss_amount"] = df["default"] * (base + noise)

    # Enforce LGD ≥ 0 and clip extreme upper tail
    df["loss_amount"] = df["loss_amount"].clip(lower=0.0, upper=20000.0)

    # -----------------------------------------
    # 3. Restrict to defaulted rows for LGD
    # -----------------------------------------
    df_lgd = df[df["default"] == 1].copy()

    # Drop any rows with NaNs in target or features
    df_lgd = df_lgd.dropna(subset=["loss_amount"] + feature_cols)

    # -----------------------------------------
    # 4. Train/test split on defaulted population
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split_xy(
        df_lgd,
        target_col="loss_amount",
        stratify=False,
    )

    # Keep only the agreed PD feature set
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # Save LGD feature list (same as PD)
    save_json(feature_cols, MODELS_DIR / "feature_cols_lgd.json")

    # -----------------------------------------
    # 5. Scale inputs
    # -----------------------------------------
    X_train_s, X_test_s = fit_scaler(X_train, X_test, scaler_name="scaler_lgd.pkl")

    # -----------------------------------------
    # 6. Train LightGBM LGD model
    # -----------------------------------------
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

    # -----------------------------------------
    # 7. Evaluate & save
    # -----------------------------------------
    y_pred = model.predict(X_test_s)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"LGD RMSE: {rmse:.2f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lgd_model.pkl")
    print("Saved LGD model to models/lgd_model.pkl")


if __name__ == "__main__":
    main()