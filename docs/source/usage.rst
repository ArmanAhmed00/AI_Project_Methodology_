Usage
=====

.. code-block:: bash

   poetry install                 # or: pip install -r requirements-dev.txt
   make data                      # clean + split
   make sweep                     # several MLflow runs
   make register                  # best run -> Model Registry (@champion)
   make explain                   # SHAP figures
   make serve                     # REST API on :5001
   make ui                        # MLflow UI on :5000

Pipeline steps
--------------

1. :mod:`churn.data` – load the Excel file, harmonise labels, stratified split.
2. :mod:`churn.features` – engineered features, imputation, one-hot encoding.
3. :mod:`churn.train` – model training, cross-validation, MLflow tracking.
4. :mod:`churn.register` – Model Registry versioning and aliasing.
5. :mod:`churn.predict` – batch inference from the registry.
6. :mod:`churn.explain` – SHAP explainability.
