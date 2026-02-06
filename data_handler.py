import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Literal, Optional
import os

ProblemType = Literal["Classification", "Regression"]

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV ou Parquet em um DataFrame do Pandas.
    Otimizado para detectar o formato pela extensão.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.parquet', '.pqt']:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}. Use CSV ou Parquet.")

def identify_problem_type(target_series: pd.Series) -> ProblemType:
    """
    Identifica se o problema é de Classificação ou Regressão.
    """
    n_unique = target_series.nunique()
    # Se for categórico ou tiver poucos valores únicos numéricos, assume classificação
    if target_series.dtype in ['object', 'category', 'bool'] or (target_series.dtype in ['int64', 'float64', 'int32'] and n_unique < 20):
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
    Divide os dados em conjuntos de treino e teste.
    """
    stratify = y if problem_type == "Classification" else None
    
    # Garantir que tipos de dados sejam eficientes
    X = X.copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test

def separate_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa o DataFrame em features (X) e target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
