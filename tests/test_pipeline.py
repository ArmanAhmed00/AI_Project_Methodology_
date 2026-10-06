"""Unit tests for data cleaning and feature engineering."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn import config
from churn.data import clean
from churn.features import FeatureBuilder, build_pipeline, split_xy


def make_df(n=40, seed=0):
    """Small random dataframe with the raw schema."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {c: rng.integers(0, 10, n).astype(float) for c in config.NUMERICAL_FEATURES}
    )
    df["OrderCount"] = rng.integers(0, 5, n)  # includes zeros
    df["PreferredLoginDevice"] = rng.choice(["Phone", "Mobile Phone", "Computer"], n)
    df["PreferredPaymentMode"] = rng.choice(["CC", "COD", "UPI"], n)
    df["Gender"] = rng.choice(["Male", "Female"], n)
    df["PreferedOrderCat"] = rng.choice(["Mobile", "Fashion"], n)
    df["MaritalStatus"] = rng.choice(["Single", "Married"], n)
    df[config.ID_COL] = range(n)
    df[config.TARGET] = rng.integers(0, 2, n)
    df.loc[0, "Tenure"] = np.nan
    return df


def test_clean_harmonises_categories():
    out = clean(make_df())
    assert "Phone" not in set(out["PreferredLoginDevice"])
    assert not {"CC", "COD"} & set(out["PreferredPaymentMode"])
    assert "Mobile" not in set(out["PreferedOrderCat"])


def test_feature_builder_handles_zero_orders():
    out = FeatureBuilder().transform(make_df())
    assert set(config.ENGINEERED_FEATURES) <= set(out.columns)
    assert not np.isinf(out["CashbackPerOrder"]).any()


def test_pipeline_fits_with_missing_values():
    X, y = split_xy(clean(make_df()))
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
