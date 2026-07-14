from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,

)
import math
import pandas as pd

from typing import Any

from src.config import (
    CLASSIFICATION,
    REGRESSION,
    CLASSIFICATION_AVERAGE
)

def _predict(
    model: Any,
    X_test: Any,
) -> Any:
    """
    Generate predictions using the trained machine learning model.

    Args:
        model: The trained machine learning model.
        X_test: The preprocessed testing features.

    Returns:
        The predicted target values.
    """
    predictions = model.predict(X_test)
    
    return predictions

def _calculate_classification_metrics(
    y_test: pd.Series,
    predictions: Any,
) -> dict[str, float]:
    """
    Calculate classification evaluation metrics.

    Args:
        y_test: The true target values.
        predictions: The predicted target values.

    Returns:
        A dictionary containing the classification metrics.
    """
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average=CLASSIFICATION_AVERAGE
    )

    recall = recall_score(
        y_test,
        predictions,
        average=CLASSIFICATION_AVERAGE)

    f1 = f1_score(
        y_test,
        predictions,
        average=CLASSIFICATION_AVERAGE
    )

    metrics = {
         "accuracy": accuracy, 
         "precision": precision, 
         "recall": recall, 
         "f1_score": f1}
    
    return metrics

def _calculate_regression_metrics(
    y_test: pd.Series,
    predictions: Any,
) -> dict[str, float]:
    mae = mean_absolute_error(
        y_test,
        predictions,
    )
    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = math.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )
    metrics = {
        "mae" : mae,
        "mse" : mse,
        "rmse" : rmse,
        "r2_score" : r2
    }
     
    return metrics

def evaluate_model(
    model: Any,
    task_type: str,
    X_test: Any,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a trained machine learning model.

    Args:
        model:
           The trained machine learning model.

        task_type:
            The machine learning task type.

        X_test:
            The preprocessed testing features.

        y_test:
            The true target values.:

    Returns:
        A dictionary containing the evaluation metrics.
    """
    predictions = _predict(model, X_test)

    if task_type == CLASSIFICATION:
        metrics = _calculate_classification_metrics(y_test, predictions)
    elif task_type == REGRESSION:
        metrics = _calculate_regression_metrics(y_test, predictions)
    else:
        raise ValueError("Unsupported task type: classification/regression expected.")
    
    return metrics
