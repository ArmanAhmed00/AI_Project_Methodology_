"""Central configuration: paths, column names and MLflow settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "E Commerce Dataset.xlsx"
RAW_SHEET_NAME = "E Comm"
INTERIM_DATA_PATH = DATA_DIR / "interim" / "churn_clean.csv"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SHAP_DIR = FIGURES_DIR / "shap"
TRAINING_FIG_DIR = FIGURES_DIR / "training"

ID_COL = "CustomerID"
TARGET = "Churn"

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

NUMERICAL_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

#: Features created by :mod:`churn.features`.
ENGINEERED_FEATURES = ["CashbackPerOrder", "CouponPerOrder", "IsNewCustomer"]

RAW_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT = "retailgenius-churn"
REGISTERED_MODEL_NAME = "churn-classifier"
