# RetailGenius – Customer Churn Prediction

EPITA · AI Project Methodology · Graded project (Parts 2 & 3).
A production-style ML project that predicts e-commerce customer churn, with MLflow tracking, a model registry, local serving and SHAP explainability.

Dataset: [E-Commerce Customer Churn (Kaggle)](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction). Put `E Commerce Dataset.xlsx` in `data/raw/`.

## Project structure (adapted from cookiecutter-data-science)

```
├── data/
│   ├── raw/            <- original Kaggle file (immutable)
│   ├── interim/        <- cleaned data
│   └── processed/      <- train / test splits, predictions
├── docs/               <- Sphinx documentation (make docs)
├── models/             <- trained pipelines (.joblib) + exported champion model
├── notebooks/          <- EDA notebook
├── reports/            <- report + figures (training, shap)
├── src/churn/
│   ├── config.py       <- paths, columns, MLflow settings
│   ├── data.py         <- 1. data preparation
│   ├── features.py     <- 2. feature engineering + preprocessing pipeline
│   ├── train.py        <- 3. training + MLflow tracking
│   ├── register.py     <- 4. MLflow Model Registry
│   ├── predict.py      <- 5. batch inference
│   ├── explain.py      <- 6. SHAP explainability (Part 3)
│   └── tracking.py     <- MLflow helper
├── tests/              <- pytest unit tests
├── MLproject           <- MLflow Projects entry points
├── python_env.yaml     <- environment used by `mlflow run`
├── pyproject.toml      <- Poetry dependencies + black config
├── requirements*.txt   <- pip alternative
├── Dockerfile          <- bonus: cloud deployment image
└── Makefile            <- shortcuts for every step
```

## Setup

Python **3.13** for the whole group (see `.python-version`).

```bash
poetry install && poetry shell          # recommended
# or
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
```

## Run the pipeline

```bash
make data        # clean + stratified split
make sweep       # 6 MLflow runs (LogReg, RandomForest, XGBoost variants)
make register    # best XGBoost run -> registry "churn-classifier" @champion
make explain     # SHAP plots -> reports/figures/shap
make predict     # batch scoring -> data/processed/predictions.csv
make ui          # MLflow UI -> http://127.0.0.1:5000
```

The same steps as an **MLflow Project**:

```bash
mlflow run . -e main --env-manager local --experiment-name retailgenius-churn
mlflow run . -e train -P model=xgboost -P max_depth=4 --env-manager local --experiment-name retailgenius-churn
```

## Serve the model (local REST API)

```bash
make serve       # http://127.0.0.1:5001/invocations
curl -X POST http://127.0.0.1:5001/invocations \
     -H "Content-Type: application/json" -d @data/sample_request.json
# -> {"predictions": [[p_stay, p_churn], ...]}
```

## Quality

```bash
make format lint test docs
```

* PEP8: `black` + `flake8` (config in `pyproject.toml` / `.flake8`)
* Docs: Sphinx with autodoc + napoleon (Google docstrings) → `docs/build/html/index.html`

## Bonus: cloud deployment

```bash
docker build -t churn-api .
docker run -p 8080:8080 churn-api
```

The image serves the exported champion model and can be deployed as is to Render, Google Cloud Run, Azure Container Apps or AWS App Runner.
