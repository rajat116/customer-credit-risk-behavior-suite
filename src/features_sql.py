# src/features_sql.py
import duckdb
import pandas as pd
from .config import DATA_RAW, DATA_PROCESSED
from .utils import ensure_dir


def build_features():
    # Load raw CSV into DuckDB
    con = duckdb.connect(database=":memory:")
    con.execute(
        "CREATE TABLE credit_raw AS SELECT * FROM read_csv_auto(?, header=True)",
        [str(DATA_RAW)],
    )

    query = """
        SELECT
            "SeriousDlqin2yrs" AS default,
            "RevolvingUtilizationOfUnsecuredLines" AS util,
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse" AS num_30_59,
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans" AS num_credit_lines,
            "NumberOfTimes90DaysLate" AS num_90,
            "NumberRealEstateLoansOrLines" AS num_real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse" AS num_60_89,
            "NumberOfDependents"
        FROM credit_raw
    """

    df = con.execute(query).fetch_df()

    # Basic cleaning: drop rows with missing key columns
    df = df.dropna(subset=["MonthlyIncome", "NumberOfDependents"], how="any")

    ensure_dir(DATA_PROCESSED.parent)
    df.to_csv(DATA_PROCESSED, index=False)
    print(f"Saved processed features to {DATA_PROCESSED}")


if __name__ == "__main__":
    build_features()