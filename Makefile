.PHONY: install data train sweep register explain predict serve ui lint format test docs all

PY = PYTHONPATH=src python
PORT ?= 5001

install:        ## install deps with poetry (or: pip install -r requirements-dev.txt)
	poetry install

data:           ## clean raw data and split train/test
	$(PY) -m churn.data

train:          ## train one XGBoost model
	$(PY) -m churn.train --model xgboost

sweep:          ## train several models/hyper-parameters (multiple MLflow runs)
	$(PY) -m churn.train --sweep

register:       ## register best run in the Model Registry (alias @champion)
	$(PY) -m churn.register

explain:        ## SHAP explanations -> reports/figures/shap
	$(PY) -m churn.explain

predict:        ## batch inference on the test set
	$(PY) -m churn.predict

serve:          ## local REST inference server
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
	  -m "models:/churn-classifier@champion" -p $(PORT) --env-manager local

ui:             ## MLflow UI on http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

format:
	black src tests docs/source/conf.py

lint:
	black --check src tests && flake8 src tests

test:
	pytest -q

docs:           ## build Sphinx HTML docs -> docs/build/html
	sphinx-build -b html docs/source docs/build/html

all: data sweep register explain predict
