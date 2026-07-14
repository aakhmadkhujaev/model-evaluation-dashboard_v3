import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    ONEHOT_DROP,
    ONEHOT_HANDLE_UNKNOWN
)
def _split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into feature matrix (X) and target vector (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def _split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
  
    """
        Split the dataset into training and testing sets.
        """
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )
    return X_train, X_test, y_train, y_test

def _detect_feature_types(
    X_train: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Detect the types of features in the training set.
    """
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    return numeric_columns, categorical_columns

def _build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """
    Build a preprocessor for handling different feature types.
    """
    numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop=ONEHOT_DROP, handle_unknown=ONEHOT_HANDLE_UNKNOWN))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_columns),
            ('cat', categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor

def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
):
    """
    Preprocess the dataset for machine learning.

    Splits the dataset into features and target, creates training
    and testing sets, detects feature types, builds the preprocessing
    pipeline, and applies the required transformations.

    Args:
        df: The input dataset.
        target_column: Name of the target column.

    Returns:
        A tuple containing:
            - X_train_processed
            - X_test_processed
            - y_train
            - y_test
            - Fitted ColumnTransformer.
    """
    # Split features and target
    X, y = _split_features_target(df, target_column)

    # Split training and testing data
    X_train, X_test, y_train, y_test = _split_train_test(X, y)

    # Detect feature types
    numeric_columns, categorical_columns = _detect_feature_types(X_train)

    # Build preprocessing pipeline
    preprocessor = _build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    # Fit on training data and transform both datasets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )