from src.loader import load_dataset
from src.validator import validate_dataset
from src.preprocessor import preprocess_data
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.reporter import generate_report
from src.visualizer import visualize_results
from src.model_saver import save_model
from typing import Any
from pathlib import Path
from src.utils import (
    print_header,
    print_section,
)
import pandas as pd
from src.config import (
    CLASSIFICATION,
    DATA_DIR,
    REGRESSION,
    SUPPORTED_CLASSIFICATION_MODELS,
    SUPPORTED_REGRESSION_MODELS,
)
def _get_model_name(
    task_type: str,
    model_choice: str,
) -> str:
    """
    Get the display name of the selected machine learning model.

    Args:
        task_type: The machine learning task type.
        model_choice: The identifier of the selected model.

    Returns:
        The name of the selected machine learning model.

    Raises:
        ValueError: If the task type or model choice is invalid.
    """
    if task_type == CLASSIFICATION:
        model_info = SUPPORTED_CLASSIFICATION_MODELS.get(model_choice)
    elif task_type == REGRESSION:
        model_info = SUPPORTED_REGRESSION_MODELS.get(model_choice)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if not model_info:
        raise ValueError(f"Invalid model choice: {model_choice}")

    return model_info["name"]

def _select_task_type() -> str:
        """
        Prompt the user to select a machine learning task.

        Returns:
            The selected machine learning task type.

        Raises:
            ValueError: If the task type is invalid.
        """
        task_type = input("Enter the task type (classification/regression): ").strip().lower()
        if task_type not in (CLASSIFICATION, REGRESSION):
            raise ValueError(f"Invalid task type: {task_type}. Must be 'classification' or 'regression'.")
        return task_type

def _select_target_column(
    df: pd.DataFrame,
) -> str:
        """
        Prompt the user to select the target column.

        Args:
            df: The loaded dataset.

        Returns:
            The selected target column.
        """
        print_section("Available Columns")

        for index, column in enumerate(df.columns, start=1):
            print(f"{index}. {column}")
        target_column = input("Enter the target column name: ").strip()
        if target_column not in df.columns:
            raise ValueError(f"Invalid target column: {target_column}")
        return target_column

def _select_model(
    task_type: str,
) -> str:
    """
    Prompt the user to select a machine learning model.

    Args:
        task_type: The machine learning task type.

    Returns:
        The identifier of the selected model.

    Raises:
        ValueError: If the task type or model choice is invalid.
    """
    if task_type == CLASSIFICATION:
        supported_models = SUPPORTED_CLASSIFICATION_MODELS
        print_section("Supported Classification Models")

    elif task_type == REGRESSION:
        supported_models = SUPPORTED_REGRESSION_MODELS
        print_section("Supported Regression Models")

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    for choice, model_info in supported_models.items():
        print(f"{choice}. {model_info['name']}")

    model_choice = input("Choose a model: ").strip()

    if model_choice not in supported_models:
        raise ValueError(f"Invalid model choice: {model_choice}")

    return model_choice

def run_pipeline() -> dict[str, Any]:
    """
    Run the complete machine learning pipeline.

    Loads the dataset, validates it, preprocesses the data,
    trains the selected model, evaluates its performance,
    and returns the results required for post-processing.

    Returns:
        A dictionary containing the pipeline results.
    """

    print_header("MODEL EVALUATION DASHBOARD")
    
    filename = input("Enter dataset filename: ").strip()
    file_path = DATA_DIR / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    df = load_dataset(file_path)

    task_type = _select_task_type()
    target_column = _select_target_column(df)

    validate_dataset(
        df,
        target_column,
        task_type,
    )
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    df,
    target_column,
    )
    model_choice = _select_model(task_type)
    trained_model = train_model(
        task_type,
        model_choice,
        X_train,
        y_train,
    )
    metrics = evaluate_model(
        trained_model,
        task_type,
        X_test,
        y_test,
    )
    model_name = _get_model_name(
        task_type,
        model_choice,
    )
    return {
        "model": trained_model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "task_type": task_type,
        "model_name": model_name,
    }

def _confirm(
    message: str,
) -> bool:
    """
    Prompt the user for a yes/no confirmation.

    Args:
        message: The confirmation message to display.

    Returns:
        True if the user confirms, False otherwise.
    """
    while True:
        response = input(f"{message} (y/n): ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")

def main() -> None:
    """
    Run the Model Evaluation Dashboard application.

    Executes the machine learning pipeline, displays the
    evaluation metrics, and allows the user to optionally
    generate reports, visualizations, and save the trained model.
    """
    while True:
        results = run_pipeline()
        print_section("Evaluation Metrics")
        for metric, value in results["metrics"].items():
            print(f"{metric:<20}: {value:.4f}")
        report = _confirm("Would you like to generate a report?")
        if report:
            report, report_path = generate_report(
                task_type=results["task_type"],
                model_name=results["model_name"],
                metrics=results["metrics"],
            )
            print(f"Report saved at: {report_path}")
        visualize = _confirm("Would you like to visualize the results?")
        if visualize:
            figure_path = visualize_results(
                model=results["model"],
                task_type=results["task_type"],
                X_test=results["X_test"],
                y_test=results["y_test"],
            )
            print(f"Visualization saved at: {figure_path}")
        model_saved = _confirm("Would you like to save the trained model?")
        if model_saved:
            model_path = save_model(
                model=results["model"],
                model_name=results["model_name"],
            )
            print(f"Model saved at: {model_path}")
        print_section("Pipeline Completed")
        print("Model Evaluation Dashboard finished successfully.")
        if not _confirm("Analyze another dataset?"):
            break

    print("Goodbye!")

if __name__ == "__main__":
    main()