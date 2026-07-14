# ==========================================
# Project Paths
# ==========================================

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR

# ==========================================
# Dataset
# ==========================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
HEADER_WIDTH = 60
DATASET_PATH = DATA_DIR / "engineered_data.csv"
MINIMUM_ROWS = 20
MINIMUM_FEATURES = 1
DEFAULT_ENCODING = "utf-8"
REPORT_WIDTH = 50
HEADER_SEPARATOR = "=" * REPORT_WIDTH
SECTION_SEPARATOR = "-" * REPORT_WIDTH
# ==========================================
# Machine Learning
# ==========================================

TEST_SIZE = 0.2

RANDOM_STATE = 42
CLASSIFICATION = "classification"
REGRESSION = "regression"
ONEHOT_DROP = "first"
ONEHOT_HANDLE_UNKNOWN = "ignore"
CLASSIFICATION_AVERAGE = "weighted"
# ==========================================
# Visualization
# ==========================================

FIGURE_DPI = 300
FIGURE_SIZE: tuple[int, int] = (10, 6)

# ==========================================
# Supported Models
# ==========================================

SUPPORTED_CLASSIFICATION_MODELS = {
    "1": {
        "name": "Logistic Regression",
        "model": LogisticRegression,
        "params": {
            "random_state": RANDOM_STATE,
            "max_iter": 1000,
        },
    },
    "2": {
        "name": "Random Forest",
        "model": RandomForestClassifier,
        "params": {
            "random_state": RANDOM_STATE,
        },
    },
    "3": {
        "name": "Support Vector Machine",
        "model": SVC,
        "params": {},
    },
}

SUPPORTED_REGRESSION_MODELS = {
    "1": {
        "name": "Linear Regression",
        "model": LinearRegression,
        "params": {
            "random_state": RANDOM_STATE,
        }
    },
    "2": {
        "name": "Random Forest Regressor",
        "model": RandomForestRegressor,
        "params": {
            "random_state": RANDOM_STATE,
        }
    },
    "3": {
        "name": "Support Vector Regressor",
        "model": SVR,
        "params": {}
    }
}

# ==========================================

# Application Configuration

# ==========================================

APP_NAME = "Model Evaluation Dashboard"

VERSION = "3.0.0"