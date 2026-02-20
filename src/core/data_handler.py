"""
trAIn Health - Data Handler Module
===================================
Functions for loading, validating, and splitting datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Literal
import os

ProblemType = Literal["Classification", "Regression"]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Parquet file into a Pandas DataFrame.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.parquet', '.pqt']:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Please use CSV or Parquet."
        )


def identify_problem_type(target_series: pd.Series) -> ProblemType:
    """
    Identify whether the problem is Classification or Regression.
    
    Args:
        target_series: The target variable series
        
    Returns:
        "Classification" or "Regression"
    """
    n_unique = target_series.nunique()
    
    # Classification if categorical or few unique numeric values
    is_categorical = target_series.dtype in ['object', 'category', 'bool']
    is_few_unique = (
        target_series.dtype in ['int64', 'float64', 'int32'] 
        and n_unique < 20
    )
    
    if is_categorical or is_few_unique:
        return "Classification"
    
    return "Regression"


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float, 
    random_state: int, 
    problem_type: ProblemType
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        problem_type: "Classification" or "Regression"
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify = y if problem_type == "Classification" else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test


def separate_features_target(
    df: pd.DataFrame, 
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate DataFrame into features (X) and target (y).
    
    Args:
        df: Complete dataset
        target_column: Name of the target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y
