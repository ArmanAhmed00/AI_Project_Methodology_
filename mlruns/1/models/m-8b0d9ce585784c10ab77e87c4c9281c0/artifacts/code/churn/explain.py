"""Explainable AI (Part 3): SHAP explanations of the champion model.

Run with ``python -m churn.explain``. Figures go to ``reports/figures/shap``
and are also logged to MLflow under a ``shap-explanations`` run.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn import config  # noqa: E402
from churn.features import split_xy  # noqa: E402
from churn.predict import load_model  # noqa: E402
from churn.tracking import start_run  # noqa: E402

logger = logging.getLogger(__name__)
OUT = config.SHAP_DIR


def _save(name):
    """Save the current matplotlib figure to the SHAP folder."""
    path = OUT / name
    plt.gcf().savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("saved %s", path.name)
    return path


def transform(pipe, X):
    """Apply the feature + preprocessing steps of the pipeline.

    Returns:
        pandas.DataFrame: Model-ready matrix with readable column names.
    """
    pre = pipe[:-1]
    Xt = pre.transform(X)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame(Xt, columns=names, index=X.index)


def compute_shap(pipe, Xt):
    """Build a TreeExplainer and compute Shapley values.

    Returns:
        tuple: ``(explainer, shap.Explanation)`` for the churn class (log-odds).
    """
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer(Xt)
    if sv.values.ndim == 3:  # e.g. random forest returns one slice per class
        sv = sv[:, :, 1]
    return explainer, sv


def pick_customer(pipe, X, y):
    """Pick an illustrative customer: the true churner with highest risk."""
    proba = pipe.predict_proba(X)[:, 1]
    churners = np.where(y.values == 1)[0]
    return int(churners[np.argmax(proba[churners])]), proba


def run(model_uri=None, max_display=15):
    """Generate every SHAP visualisation requested by the assignment."""
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = load_model(model_uri)
    X, y = split_xy(pd.read_csv(config.TEST_PATH))
    Xt = transform(pipe, X)
    explainer, sv = compute_shap(pipe, Xt)
    pd.DataFrame(sv.values, columns=Xt.columns).to_csv(
        OUT / "shap_values_test.csv", index=False
    )

    idx, proba = pick_customer(pipe, X, y)
    logger.info("Explaining test row %d (p_churn=%.3f)", idx, proba[idx])

    # 1. Explanation for one specific point (force plot)
    shap.force_plot(
        sv.base_values[idx],
        sv.values[idx],
        Xt.iloc[idx].round(2),
        matplotlib=True,
        show=False,
        text_rotation=20,
    )
    _save("01_force_plot_single.png")
    shap.save_html(
        str(OUT / "01_force_plot_single.html"),
        shap.force_plot(sv.base_values[idx], sv.values[idx], Xt.iloc[idx]),
    )

    # 2. Explanations for all points at once (stacked force + heatmap)
    shap.save_html(
        str(OUT / "02_force_plot_all.html"),
        shap.force_plot(explainer.expected_value, sv.values, Xt),
    )
    shap.plots.heatmap(sv, max_display=max_display, show=False)
    _save("02_heatmap_all_points.png")

    # 3. Summary plot for each class (class 0 = -SHAP of class 1 in log-odds)
    shap.summary_plot(sv.values, Xt, max_display=max_display, show=False)
    plt.title("Summary plot – class 1 (Churn)")
    _save("03_summary_class1_churn.png")
    shap.summary_plot(-sv.values, Xt, max_display=max_display, show=False)
    plt.title("Summary plot – class 0 (Stay)")
    _save("03_summary_class0_stay.png")
    shap.summary_plot(
        [-sv.values, sv.values],
        Xt,
        plot_type="bar",
        class_names=["Stay (0)", "Churn (1)"],
        max_display=max_display,
        show=False,
    )
    _save("03_summary_bar_per_class.png")

    # 4. Going further for the specific point
    shap.plots.waterfall(sv[idx], max_display=max_display, show=False)
    _save("04_waterfall_single.png")
    shap.plots.bar(sv[idx], max_display=max_display, show=False)
    _save("04_bar_single.png")
    shap.plots.bar(sv, max_display=max_display, show=False)
    _save("05_mean_shap_bar.png")
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    _save("06_beeswarm.png")

    top = np.argsort(np.abs(sv.values).mean(0))[::-1][:4]
    for rank, j in enumerate(top, 1):
        feat = Xt.columns[j]
        shap.plots.scatter(sv[:, feat], color=sv, show=False)
        _save(f"07_dependence_{rank}_{feat}.png")

    with start_run("shap-explanations"):
        mlflow.set_tag("stage", "xai")
        mlflow.log_param("explained_row", idx)
        mlflow.log_metric("explained_row_p_churn", float(proba[idx]))
        mlflow.log_artifacts(str(OUT), artifact_path="shap")

    importance = (
        pd.Series(np.abs(sv.values).mean(0), index=Xt.columns)
        .sort_values(ascending=False)
        .round(4)
    )
    importance.to_csv(OUT / "mean_abs_shap.csv", header=["mean_abs_shap"])
    return importance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-uri", default=None)
    print(run(p.parse_args().model_uri).head(10))
