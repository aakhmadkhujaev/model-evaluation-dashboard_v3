import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.metrics import (
    ConfusionMatrixDisplay,
)
import pandas as pd
from pathlib import Path

from typing import Any

from src.config import (
    CLASSIFICATION,
    REGRESSION,
    FIGURES_DIR,
    FIGURE_DPI,
    FIGURE_SIZE,
)

from src.utils import get_timestamp

def _plot_confusion_matrix(
    model: Any,
    X_test: Any,
    y_test: pd.Series,
) -> Figure:
    """
    Create a confusion matrix visualization.

    Args:
        model: The trained classification model.
        X_test: The preprocessed testing features.
        y_test: The true target values.

    Returns:
        The generated Matplotlib figure.
    """
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=axis
    )
    axis.set_title("Confusion Matrix")
    figure.tight_layout()

    return figure

def _save_figure(
    figure: Figure,
    filename: str,
) -> Path:
    """
    Save the generated figure to the specified directory.

    Args:
        figure: The Matplotlib figure to save.
        filename: The name of the file to save the figure as.

    Returns:
        The path to the saved figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    figure_path = FIGURES_DIR / f"{filename}_{timestamp}.png"
    figure.savefig(figure_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return figure_path
    

def visualize_results(
    model: Any,
    task_type: str,
    X_test: Any,
    y_test: pd.Series,
) -> Path:
    """
    Visualize the results of the model based on the task type.

    Args:
        model: The trained model.
        task_type: The type of task (classification or regression).
        X_test: The preprocessed testing features.
        y_test: The true target values.

    Returns:
        The path to the saved figure.
    """
    if task_type == CLASSIFICATION:
        figure = _plot_confusion_matrix(model, X_test, y_test)
        figure_path = _save_figure(figure, "confusion_matrix")
    elif task_type == REGRESSION:
        # Placeholder for regression visualization (e.g., scatter plot)
        # Implement regression visualization logic here
        raise NotImplementedError("Regression visualization is not implemented yet.")
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return figure_path