"""Configuración compartida, independiente del directorio de ejecución."""

from pathlib import Path

PROJECT_NAME = "Chuard"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "0.1.0"
RANDOM_STATE = 42
CV_FOLDS = 5
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
LOGISTIC_MAX_ITER = 1000
DATASET_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models/best_model.joblib"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "reports/modeling/splits.json"
DEFAULT_RISK_LIMITS = {"medium": 0.35, "high": 0.65}
NEW_CUSTOMER_MONTHS = 6
MAX_TOP_FACTORS = 5
# Límites iniciales de desarrollo, pendientes de medir con modelo/datos reales.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
SHAP_BACKGROUND_SIZE = 100
SHAP_MAX_EVALS = 195
SHAP_MAX_GLOBAL_ROWS = 100
SHAP_GLOBAL_DISPLAY_FEATURES = 15
SHAP_LOCAL_DISPLAY_FEATURES = 10
SHAP_CACHE_SIZE = 4
SHAP_ADDITIVITY_ATOL = 1e-6
SHAP_ZERO_TOLERANCE = 1e-12
EDA_MIN_SEGMENT_SIZE = 30
EDA_IQR_MULTIPLIER = 1.5
FIGURE_DPI = 150
DATA_REPORT_DIR = PROJECT_ROOT / "reports/data"
EXPLANATION_REPORT_DIR = PROJECT_ROOT / "reports/explain"
EVALUATION_REPORT_DIR = PROJECT_ROOT / "reports/evaluation"
MODEL_MANIFEST_PATH = MODEL_PATH.with_suffix(".manifest.json")
REPRODUCTION_DIR = PROJECT_ROOT / "reports/modeling/reproduction_20261110_reanudada"
REPORT_METRICS_ATOL = 1e-12
DATASET_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
)
DOWNLOAD_TIMEOUT_SECONDS = 45
POSITIVE_DIRECTION = "aumenta"
NEGATIVE_DIRECTION = "disminuye"
AUTOMATIC_PAYMENT_METHODS = (
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)
