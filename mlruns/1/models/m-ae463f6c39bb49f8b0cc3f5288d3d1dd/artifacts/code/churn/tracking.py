"""MLflow helpers shared by training and explanation scripts."""

import os

import mlflow

from churn import config

_PARENT_RUN = None


def start_run(run_name):
    """Start an MLflow run that works both standalone and under ``mlflow run``.

    When launched through *MLflow Projects*, a parent run already exists
    (``MLFLOW_RUN_ID``); it is resumed once and every training run is then
    logged as a nested child run.

    Args:
        run_name: Name displayed in the MLflow UI.

    Returns:
        mlflow.ActiveRun: Context manager for the run.
    """
    global _PARENT_RUN
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if _PARENT_RUN is None and "MLFLOW_RUN_ID" in os.environ:
        _PARENT_RUN = mlflow.start_run(run_id=os.environ.pop("MLFLOW_RUN_ID"))
    elif _PARENT_RUN is None:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return mlflow.start_run(run_name=run_name, nested=_PARENT_RUN is not None)
