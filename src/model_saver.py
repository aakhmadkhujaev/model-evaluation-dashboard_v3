from pathlib import Path
from typing import Any

import joblib

from src.config import (
    MODELS_DIR,
)

from src.utils import get_timestamp

def save_model(
    model: Any,
    model_name: str,
) -> Path:
    """
    Save a trained machine learning model.

    Args:
        model: The trained machine learning model.
        model_name: The name of the model.

    Returns:
        The path to the saved model.
    """
    timestamp = get_timestamp()
    filename = f"{model_name}_{timestamp}.joblib"
    model_path = MODELS_DIR / filename

    # Ensure the models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    return model_path