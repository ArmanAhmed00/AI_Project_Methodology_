"""Batch inference with the registered model.

Example::

    python -m churn.predict --input data/processed/test.csv \\
        --output data/processed/predictions.csv
"""

import argparse
import logging

import mlflow
import pandas as pd

from churn import config

logger = logging.getLogger(__name__)


def load_model(model_uri=None):
    """Load a model from the MLflow Model Registry.

    Args:
        model_uri: MLflow URI, defaults to ``models:/churn-classifier@champion``.

    Returns:
        The loaded sklearn pipeline.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = model_uri or f"models:/{config.REGISTERED_MODEL_NAME}@champion"
    return mlflow.sklearn.load_model(model_uri)


def predict(df, model=None, threshold=0.5):
    """Score customers.

    Args:
        df: Dataframe with the raw feature columns.
        model: Fitted pipeline (loaded from the registry if ``None``).
        threshold: Probability above which a customer is flagged as churner.

    Returns:
        pandas.DataFrame: ``df`` with ``churn_probability`` and ``churn_pred``.
    """
    model = model or load_model()
    out = df.copy()
    out["churn_probability"] = model.predict_proba(df[config.RAW_FEATURES])[:, 1]
    out["churn_pred"] = (out["churn_probability"] >= threshold).astype(int)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(config.TEST_PATH))
    p.add_argument(
        "--output", default=str(config.DATA_DIR / "processed" / "predictions.csv")
    )
    p.add_argument("--model-uri", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    res = predict(pd.read_csv(a.input), load_model(a.model_uri), a.threshold)
    res.to_csv(a.output, index=False)
    logger.info("Wrote %d predictions to %s", len(res), a.output)
