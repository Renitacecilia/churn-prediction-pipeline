"""Extract -> Transform -> Load: raw CSV into a clean SQLite table."""
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src import config

log = logging.getLogger(__name__)
REQUIRED = [config.ID_COL, config.TARGET, *config.FEATURES]


def extract(path: Path = config.RAW_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` "
            "(or `python -m src.make_sample_data` for synthetic data)."
        )
    df = pd.read_csv(path)
    log.info("Extracted %d rows from %s", len(df), path)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if (df["tenure"] < 0).any():
        raise ValueError("tenure contains negative values")
    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges contains negative values")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data. Pure function: easy to unit test."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates(subset=config.ID_COL)

    # TotalCharges arrives as text with blanks for brand-new customers (tenure == 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = df["TotalCharges"].isna()
    df.loc[blank, "TotalCharges"] = df.loc[blank, "MonthlyCharges"] * df.loc[blank, "tenure"]

    mapped = df[config.TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if mapped.isna().any():
        raise ValueError("Churn column contains values other than 'Yes'/'No'")
    df[config.TARGET] = mapped.astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    validate(df)
    log.info("Transformed data: %d rows, %d columns", *df.shape)
    return df.reset_index(drop=True)


def load(df: pd.DataFrame, db_path: Path = config.DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
    log.info("Loaded %d rows into %s (table: customers)", len(df), db_path)


def run() -> pd.DataFrame:
    df = transform(extract())
    load(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
