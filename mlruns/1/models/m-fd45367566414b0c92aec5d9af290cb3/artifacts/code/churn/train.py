"""Model training with MLflow tracking.

Examples::

    python -m churn.train --model xgboost --n-estimators 300 --max-depth 4
    python -m churn.train --sweep            # several runs for comparison
"""

import argparse
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import skops.io as skops_io  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from churn import config  # noqa: E402
from churn.features import build_pipeline, split_xy  # noqa: E402
from churn.tracking import start_run  # noqa: E402

logger = logging.getLogger(__name__)

#: Hyper-parameter sets explored by ``--sweep``.
SWEEP = [
    {"model": "logreg", "C": 1.0},
    {"model": "random_forest", "n_estimators": 300, "max_depth": 8},
    {"model": "random_forest", "n_estimators": 500, "max_depth": None},
    {"model": "xgboost", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
    {"model": "xgboost", "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
    {"model": "xgboost", "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05},
]


def make_estimator(model, n_estimators=300, max_depth=5, learning_rate=0.05, C=1.0):
    """Instantiate the requested classifier.

    Args:
        model: One of ``xgboost``, ``random_forest``, ``logreg``.
        n_estimators: Number of trees (tree models).
        max_depth: Maximum tree depth (tree models).
        learning_rate: Boosting learning rate (xgboost).
        C: Inverse regularisation strength (logreg).

    Returns:
        A sklearn-compatible classifier.
    """
    if model == "xgboost":
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if model == "logreg":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model}")


def evaluate(pipe, X, y, threshold=0.5):
    """Compute classification metrics on a dataset.

    Returns:
        dict: metric name -> value.
    """
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def _log_figures(pipe, X, y, run_name):
    """Save and log ROC curve and confusion matrix."""
    config.TRAINING_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_estimator(pipe, X, y, ax=axes[0])
    axes[0].set_title("ROC curve (test)")
    ConfusionMatrixDisplay.from_estimator(pipe, X, y, ax=axes[1], colorbar=False)
    axes[1].set_title("Confusion matrix (test)")
    fig.tight_layout()
    path = config.TRAINING_FIG_DIR / f"{run_name}_eval.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="figures")


def train(params, register=False):
    """Train one model, log everything to MLflow and return the run id.

    Args:
        params: Dict with ``model`` plus its hyper-parameters.
        register: If True, register the logged model in the Model Registry.

    Returns:
        str: MLflow run id.
    """
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    model_name = params["model"]
    hp = {k: v for k, v in params.items() if k != "model"}
    pipe = build_pipeline(make_estimator(model_name, **hp))
    run_name = model_name + "_" + "_".join(f"{k}{v}" for k, v in hp.items())

    with start_run(run_name) as run:
        mlflow.set_tags({"model_family": model_name, "project": "retailgenius-churn"})
        mlflow.log_params({"model": model_name, **hp})
        mlflow.log_params(
            {"n_train": len(X_train), "n_test": len(X_test), "cv_folds": 5}
        )

        cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", cv_auc.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_auc.std())

        pipe.fit(X_train, y_train)
        for split_name, (X, y) in {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
        }.items():
            for k, v in evaluate(pipe, X, y).items():
                mlflow.log_metric(f"{split_name}_{k}", v)

        _log_figures(pipe, X_test, y_test, run_name)

        signature = infer_signature(X_train, pipe.predict_proba(X_train))
        # skops (secure format, MLflow default): we explicitly trust the
        # types of the pipeline we just built ourselves.
        trusted = skops_io.get_untrusted_types(data=skops_io.dumps(pipe))
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            serialization_format="skops",
            skops_trusted_types=trusted,
            pyfunc_predict_fn="predict_proba",
            signature=signature,
            input_example=X_train.head(3),
            code_paths=[str(config.PROJECT_ROOT / "src" / "churn")],
            registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
        )

        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipe, config.MODELS_DIR / f"{run_name}.joblib")
        logger.info(
            "%s | cv_auc=%.4f | test_auc=%.4f",
            run_name,
            cv_auc.mean(),
            roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
        )
        return run.info.run_id


def _none_or_int(value):
    return None if str(value).lower() in {"none", "", "-1"} else int(value)


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train churn models with MLflow")
    p.add_argument("--model", default="xgboost")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=_none_or_int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--register", action="store_true")
    p.add_argument("--sweep", action="store_true", help="run the SWEEP grid")
    args = p.parse_args()

    if args.sweep:
        for params in SWEEP:
            train(params)
        return

    params = {"model": args.model}
    if args.model == "logreg":
        params["C"] = args.C
    else:
        params.update(n_estimators=args.n_estimators, max_depth=args.max_depth)
        if args.model == "xgboost":
            params["learning_rate"] = args.learning_rate
    train(params, register=args.register)


if __name__ == "__main__":
    main()
