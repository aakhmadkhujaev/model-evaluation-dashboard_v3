import pandas as pd
from pathlib import Path


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file and return it as a DataFrame.
    """
    return pd.read_csv(file_path)
