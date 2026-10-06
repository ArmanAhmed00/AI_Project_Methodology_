"""Data preparation: load the raw Excel file, clean it and split train/test.

Run with ``python -m churn.data``.
"""

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)

#: Harmonise duplicated category labels found in the raw data.
CATEGORY_FIXES = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


def load_raw(path=config.RAW_DATA_PATH, sheet=config.RAW_SHEET_NAME):
    """Load the raw Kaggle dataset.

    Args:
        path: Path to ``E Commerce Dataset.xlsx`` (a ``.csv`` is accepted too).
        sheet: Excel sheet holding the data (the first sheet is a dictionary).

    Returns:
        pandas.DataFrame: Raw data, one row per customer.
    """
    path = str(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        logger.warning("Sheet %s not found, falling back to second sheet", sheet)
        return pd.read_excel(path, sheet_name=1)


def clean(df):
    """Clean the raw data.

    * drops duplicated customers,
    * harmonises inconsistent category labels,
    * keeps missing values (they are imputed inside the model pipeline so that
      the exact same logic is applied at serving time).

    Args:
        df: Raw dataframe.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    df = df.drop_duplicates(subset=config.ID_COL).copy()
    for col, mapping in CATEGORY_FIXES.items():
        df[col] = df[col].replace(mapping)
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def split(df, test_size=config.TEST_SIZE, seed=config.RANDOM_STATE):
    """Stratified train/test split on the churn label.

    Args:
        df: Cleaned dataframe.
        test_size: Fraction kept for the hold-out test set.
        seed: Random seed.

    Returns:
        tuple: ``(train_df, test_df)``.
    """
    return train_test_split(
        df, test_size=test_size, stratify=df[config.TARGET], random_state=seed
    )


def main(raw_path=config.RAW_DATA_PATH):
    """Run the full data preparation step and write CSVs to ``data/``."""
    df = load_raw(raw_path)
    logger.info("Raw shape: %s", df.shape)
    df = clean(df)
    config.INTERIM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.INTERIM_DATA_PATH, index=False)

    train, test = split(df)
    config.TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    logger.info(
        "Train %s / Test %s | churn rate %.3f",
        train.shape,
        test.shape,
        df[config.TARGET].mean(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default=str(config.RAW_DATA_PATH))
    main(parser.parse_args().raw_path)
