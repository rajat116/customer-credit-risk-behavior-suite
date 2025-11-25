# src/train_timeseries.py
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from .config import DATA_PROCESSED, MODELS_DIR
from .utils import load_csv, ensure_dir
import joblib


def main():
    df = load_csv(DATA_PROCESSED)

    # Simulate monthly default rate from the binary defaults
    # Assume each row is a customer; we'll create fake monthly indices
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    months = 24
    df["month"] = np.repeat(np.arange(months), n // months + 1)[:n]

    ts = df.groupby("month")["default"].mean().rename("default_rate")

    # Fit ARIMA to monthly default_rate
    model = ARIMA(ts, order=(1, 0, 1))
    result = model.fit()

    ensure_dir(MODELS_DIR)
    joblib.dump(result, MODELS_DIR / "arima_default_rate.pkl")

    # Also save the time series for plotting
    ts.to_csv(MODELS_DIR / "default_rate_timeseries.csv", index=True)
    print("Saved ARIMA model and default rate time series.")


if __name__ == "__main__":
    main()