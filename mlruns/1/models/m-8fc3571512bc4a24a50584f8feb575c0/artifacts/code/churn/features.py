"""Feature engineering and preprocessing pipeline.

Everything here lives *inside* the sklearn pipeline, so the served model
accepts raw customer records and applies identical transformations.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn import config


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Add business-driven features to the raw customer table.

    New columns:
        * ``CashbackPerOrder``: average cashback received per order,
        * ``CouponPerOrder``: share of orders with a coupon,
        * ``IsNewCustomer``: 1 if tenure is at most 1 month.
    """

    def fit(self, X, y=None):  # noqa: D102  (sklearn API)
        return self

    def transform(self, X):
        """Return a copy of ``X`` with engineered columns appended."""
        X = pd.DataFrame(X).copy()
        orders = X["OrderCount"].replace(0, np.nan)
        X["CashbackPerOrder"] = X["CashbackAmount"] / orders
        X["CouponPerOrder"] = X["CouponUsed"] / orders
        X["IsNewCustomer"] = (X["Tenure"] <= 1).astype(float)
        X.loc[X["Tenure"].isna(), "IsNewCustomer"] = np.nan
        return X


def build_preprocessor():
    """Impute numericals (median) and one-hot encode categoricals.

    Returns:
        sklearn.compose.ColumnTransformer: Unfitted preprocessor.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERICAL_FEATURES + config.ENGINEERED_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator):
    """Chain feature building, preprocessing and a classifier.

    Args:
        estimator: Any sklearn-compatible classifier.

    Returns:
        sklearn.pipeline.Pipeline: ``features -> preprocess -> model``.
    """
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def split_xy(df):
    """Split a dataframe into raw features ``X`` and target ``y``.

    Numerical columns are cast to float so that the MLflow signature accepts
    missing values at serving time.
    """
    X = df[config.RAW_FEATURES].copy()
    X[config.NUMERICAL_FEATURES] = X[config.NUMERICAL_FEATURES].astype("float64")
    return X, df[config.TARGET]
