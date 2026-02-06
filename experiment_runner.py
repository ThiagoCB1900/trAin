import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Dict, Any, List, Tuple
import numpy as np
import joblib
import io

def evaluate_model(
    pipeline: ImbPipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    problem_type: str
) -> Dict[str, Any]:
    """
    Treina o pipeline no conjunto de treino e avalia no conjunto de teste intocado.
    Retorna um dicionário com as métricas de avaliação e o objeto do pipeline treinado.
    """
    
    # 1. Treinamento (fit)
    pipeline.fit(X_train, y_train)
    
    # 2. Predição
    y_pred = pipeline.predict(X_test)
    
    metrics = {}
    y_score = None
    
    if problem_type == "Classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        average_type = 'weighted' if y_test.nunique() > 2 else 'binary'
        metrics["precision"] = precision_score(y_test, y_pred, average=average_type, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, average=average_type, zero_division=0)
        metrics["f1_score"] = f1_score(y_test, y_pred, average=average_type, zero_division=0)
        
        # Probabilidades para curvas ROC e PR (se o modelo suportar)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)
        
    elif problem_type == "Regression":
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["r2_score"] = r2_score(y_test, y_pred)
        
    return {
        "metrics": metrics,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_score": y_score
    }

def generate_plots(
    y_test: pd.Series, 
    y_pred: np.ndarray, 
    y_score: np.ndarray = None, 
    problem_type: str = "Classification"
) -> Dict[str, plt.Figure]:
    """Gera os gráficos solicitados: Matriz de Confusão, ROC e Precision-Recall."""
    plots = {}
    
    if problem_type == "Classification":
        # 1. Matriz de Confusão
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title('Matriz de Confusão')
        ax_cm.set_xlabel('Predito')
        ax_cm.set_ylabel('Real')
        plots["confusion_matrix"] = fig_cm
        
        # Gráficos que dependem de probabilidades (apenas para binário neste MVP)
        if y_score is not None and len(np.unique(y_test)) == 2:
            # 2. Curva ROC
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_title('Curva ROC')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            plots["roc_curve"] = fig_roc
            
            # 3. Curva Precision-Recall
            fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
            ax_pr.plot(recall, precision, color='blue', lw=2)
            ax_pr.set_title('Curva Precision-Recall')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            plots["pr_curve"] = fig_pr
            
    return plots

def save_pipeline_to_buffer(pipeline: ImbPipeline) -> io.BytesIO:
    """Serializa o pipeline para um buffer de bytes para download."""
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    return buffer
