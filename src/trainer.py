from src.config import (
    CLASSIFICATION,
    REGRESSION,
    SUPPORTED_CLASSIFICATION_MODELS,
    SUPPORTED_REGRESSION_MODELS,
)
from typing import Any
import pandas as pd

def _get_model(
    task_type: str,
    model_choice: str,
) -> Any:
    """
    Create a machine learning model based on the selected task type and model choice.

    Args:
        task_type: The machine learning task type
            ('classification' or 'regression').
        model_choice: The identifier of the selected model.

    Returns:
        An instantiated scikit-learn model.
    """
    if task_type == CLASSIFICATION:
        model_info = SUPPORTED_CLASSIFICATION_MODELS.get(model_choice)
    elif task_type == REGRESSION:
        model_info = SUPPORTED_REGRESSION_MODELS.get(model_choice)
    else:
        raise ValueError(
        f"Invalid model choice '{model_choice}'."
        )

    if model_info is None:
        raise ValueError(
            f"Invalid model choice '{model_choice}'."
        )

    model_class = model_info["model"]
    model_params = model_info.get("params", {})

    return model_class(**model_params)

def _train_model(
    model: Any,
    X_train : Any,
    y_train: pd.Series,
) -> Any:
        """
    Train the selected machine learning model.

    Args:
        model: The instantiated machine learning model.
        X_train: The preprocessed training features.
        y_train: The training target values.

    Returns:
        The trained machine learning model.
    """
        model.fit(
            X_train,
            y_train,
        )

        return model
def train_model(
    task_type: str,
    model_choice: str,
    X_train: Any,
    y_train: pd.Series,
) -> Any:
    """
    Train a machine learning model.

    Creates the selected model, trains it using the
    preprocessed training data, and returns the trained model.

    Args:
        task_type: The machine learning task type.
        model_choice: The identifier of the selected model.
        X_train: The preprocessed training features.
        y_train: The training target values.

    Returns:
        The trained machine learning model.
    """
    model = _get_model(
        task_type,
        model_choice,
    )

    trained_model = _train_model(
        model,
        X_train,
        y_train,
    )

    return trained_model