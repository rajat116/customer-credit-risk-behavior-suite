# src/train_pd_model.py
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from .config import DATA_PROCESSED, MODELS_DIR
from .utils import load_csv, train_test_split_xy, fit_scaler, save_json


def main():
    df = load_csv(DATA_PROCESSED)

    # ----------------------------------------------------
    # Explicit, stable feature ordering
    # ----------------------------------------------------
    feature_cols = [
        "util",
        "age",
        "num_30_59",
        "DebtRatio",
        "MonthlyIncome",
        "num_credit_lines",
        "num_90",
        "num_real_estate",
        "num_60_89",
        "NumberOfDependents",
    ]

    # Save explicit feature order (very important)
    save_json(feature_cols, MODELS_DIR / "feature_cols.json")

    # Use explicit feature list from now on
    X = df[feature_cols]
    y = df["default"]

    # ----------------------------------------------------
    # Split (internally removes y)
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split_xy(
        df,
        target_col="default",
        stratify=True,
    )

    # Now align to feature_cols again
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # ----------------------------------------------------
    # Scale inputs
    # ----------------------------------------------------
    X_train_s, X_test_s = fit_scaler(X_train, X_test, scaler_name="scaler_pd.pkl")

    # ----------------------------------------------------
    # Train PD Model (XGBoost)
    # ----------------------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train_s, y_train)

    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    print(f"PD AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))

    # ----------------------------------------------------
    # Save model
    # ----------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "pd_model.pkl")
    print("Saved PD model to models/pd_model.pkl")


if __name__ == "__main__":
    main()