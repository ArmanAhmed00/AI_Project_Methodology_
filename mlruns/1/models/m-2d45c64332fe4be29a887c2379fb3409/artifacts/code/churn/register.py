"""Model Registry: register the best run and promote it with an alias.

Run with ``python -m churn.register``.
"""

import argparse
import logging
import shutil

import mlflow
from mlflow import MlflowClient

from churn import config

logger = logging.getLogger(__name__)


def register_best(metric="cv_roc_auc_mean", model_family="xgboost", alias="champion"):
    """Register the best run of the experiment and set an alias on it.

    Args:
        metric: Metric used to rank runs (higher is better).
        model_family: Restrict to one model family (``None`` = all).
        alias: Alias assigned to the new version (e.g. ``champion``).

    Returns:
        mlflow.entities.model_registry.ModelVersion: The registered version.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    filt = f"tags.model_family = '{model_family}'" if model_family else ""
    best = client.search_runs(
        [exp.experiment_id],
        filter_string=filt,
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )[0]
    logger.info(
        "Best run %s (%s=%.4f)",
        best.info.run_name,
        metric,
        best.data.metrics[metric],
    )
    mv = mlflow.register_model(
        f"runs:/{best.info.run_id}/model", config.REGISTERED_MODEL_NAME
    )
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, alias, mv.version)
    client.update_model_version(
        config.REGISTERED_MODEL_NAME,
        mv.version,
        description=f"Best {model_family} run by {metric}: "
        f"{best.data.metrics[metric]:.4f}",
    )
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, mv.version, "validation_status", "approved"
    )
    logger.info(
        "Registered %s v%s with alias @%s",
        config.REGISTERED_MODEL_NAME,
        mv.version,
        alias,
    )
    export_dir = config.MODELS_DIR / "serving_model"
    shutil.rmtree(export_dir, ignore_errors=True)
    mlflow.artifacts.download_artifacts(
        f"models:/{config.REGISTERED_MODEL_NAME}@{alias}", dst_path=str(export_dir)
    )
    logger.info("Exported champion model to %s (used by the Dockerfile)", export_dir)
    return mv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="cv_roc_auc_mean")
    p.add_argument("--model-family", default="xgboost")
    p.add_argument("--alias", default="champion")
    a = p.parse_args()
    register_best(a.metric, a.model_family or None, a.alias)
