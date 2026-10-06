"""Sphinx configuration for the RetailGenius churn project."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "RetailGenius Churn Prediction"
author = "EPITA AI PM Group"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_mock_imports = ["xgboost", "shap", "mlflow", "skops"]
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
