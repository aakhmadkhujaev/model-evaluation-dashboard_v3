import pandas as pd
from src.config import (
    MINIMUM_ROWS,
    MINIMUM_FEATURES,
    CLASSIFICATION,
    REGRESSION,
    REGRESSION
)

def _check_empty_dataset(df: pd.DataFrame) -> None:
    """
    Check whether the dataset is empty.
    """
    if df.empty:
        raise ValueError("The dataset is empty.")

def _check_target_exists(df: pd.DataFrame, target_column: str) -> None:
    """
    Check whether the specified target column exists in the dataset.
    """
    if target_column not in df.columns:
        raise ValueError("The specified target column does not exist in the dataset.")

def _check_duplicate_columns(df: pd.DataFrame) -> None:
    """
    Check whether the dataset contains duplicate column names.
    """
    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

def _check_minimum_rows(df: pd.DataFrame) -> None:
    """
    Check whether the dataset contains at least the minimum number of rows.
    """
    if len(df) < MINIMUM_ROWS:
        raise ValueError(f"The dataset must contain at least {MINIMUM_ROWS} rows.")

def _check_minimum_features(df: pd.DataFrame) -> None:
    """
    Check whether the dataset contains at least the minimum number of features.
    """
    if len(df.columns) < MINIMUM_FEATURES:
        raise ValueError(f"The dataset must contain at least {MINIMUM_FEATURES} features.")

def _check_target_missing_values(target_series: pd.Series) -> None:
    """
    Check whether the target column contains missing values.
    """
    if target_series.isnull().any():
        raise ValueError("The target column contains missing values.")

def _check_target_classes(target_series: pd.Series) -> None:
    """
    Check whether the target column contains at least two unique values.
    """
    if target_series.nunique() < 2:
        raise ValueError("The target column must contain at least two unique values.")

def validate_dataset(df: pd.DataFrame, target_column: str, task_type: str) -> None:
    _check_empty_dataset(df)
    _check_target_exists(df, target_column)
    _check_duplicate_columns(df)
    _check_minimum_rows(df)
    _check_minimum_features(df)
    target_series = df[target_column]
    _check_target_missing_values(target_series)
    if task_type == CLASSIFICATION:
        _check_target_classes(target_series)
    elif task_type == REGRESSION:
        pass
    else:
        raise ValueError("Invalid task type. Please specify either 'classification' or 'regression'.")