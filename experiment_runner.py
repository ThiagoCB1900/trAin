import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score, log_loss
)
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Dict, Any, List, Tuple
import numpy as np
import joblib
import io


def _compute_specificity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fp = cm[0, 1]
        denom = tn + fp
        return float(tn / denom) if denom > 0 else 0.0

    specs = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        denom = tn + fp
        specs.append(float(tn / denom) if denom > 0 else 0.0)
    return float(np.mean(specs)) if specs else 0.0

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
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        metrics["mcc"] = matthews_corrcoef(y_test, y_pred)
        metrics["specificity"] = _compute_specificity(y_test, y_pred)

        average_weighted = "weighted" if y_test.nunique() > 2 else "binary"
        metrics["precision_weighted"] = precision_score(y_test, y_pred, average=average_weighted, zero_division=0)
        metrics["recall_weighted"] = recall_score(y_test, y_pred, average=average_weighted, zero_division=0)
        metrics["f1_weighted"] = f1_score(y_test, y_pred, average=average_weighted, zero_division=0)
        metrics["precision_macro"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_test, y_pred, average="macro", zero_division=0)

        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)
            try:
                metrics["log_loss"] = log_loss(y_test, y_score)
            except ValueError:
                metrics["log_loss"] = None
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)

        if y_score is not None:
            try:
                if y_test.nunique() == 2:
                    if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                        score_vec = y_score[:, 1]
                    else:
                        score_vec = y_score
                    metrics["roc_auc"] = roc_auc_score(y_test, score_vec)
                    metrics["pr_auc"] = average_precision_score(y_test, score_vec)
                else:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
                    metrics["pr_auc"] = None
            except ValueError:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
        
    elif problem_type == "Regression":
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["mse"] = mean_squared_error(y_test, y_pred)
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["mape"] = mean_absolute_percentage_error(y_test, y_pred)
        metrics["median_ae"] = median_absolute_error(y_test, y_pred)
        metrics["r2_score"] = r2_score(y_test, y_pred)
        metrics["explained_variance"] = explained_variance_score(y_test, y_pred)
        
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

        # 1b. Matriz de Confusao Normalizada
        fig_cm_n, ax_cm_n = plt.subplots(figsize=(8, 6))
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm_n)
        ax_cm_n.set_title('Matriz de Confusao (Normalizada)')
        ax_cm_n.set_xlabel('Predito')
        ax_cm_n.set_ylabel('Real')
        plots["confusion_matrix_norm"] = fig_cm_n
        
        # Gráficos que dependem de probabilidades (apenas para binário neste MVP)
        if y_score is not None and len(np.unique(y_test)) == 2:
            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                score_vec = y_score[:, 1]
            else:
                score_vec = y_score
            # 2. Curva ROC
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, score_vec)
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
            precision, recall, _ = precision_recall_curve(y_test, score_vec)
            ax_pr.plot(recall, precision, color='blue', lw=2)
            ax_pr.set_title('Curva Precision-Recall')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            plots["pr_curve"] = fig_pr
    elif problem_type == "Regression":
        # 1. Predito vs Real
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        ax_scatter.scatter(y_test, y_pred, alpha=0.6, color='teal')
        min_val = min(float(y_test.min()), float(np.min(y_pred)))
        max_val = max(float(y_test.max()), float(np.max(y_pred)))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--')
        ax_scatter.set_title('Predito vs Real')
        ax_scatter.set_xlabel('Real')
        ax_scatter.set_ylabel('Predito')
        plots["pred_vs_real"] = fig_scatter

        # 2. Residuos
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        ax_res.scatter(y_pred, residuals, alpha=0.6, color='purple')
        ax_res.axhline(0, color='gray', linestyle='--')
        ax_res.set_title('Residuos vs Predito')
        ax_res.set_xlabel('Predito')
        ax_res.set_ylabel('Residuo')
        plots["residuals"] = fig_res

        # 3. Distribuicao dos Residuos
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        ax_hist.hist(residuals, bins=30, color='slateblue', alpha=0.75)
        ax_hist.set_title('Distribuicao dos Residuos')
        ax_hist.set_xlabel('Residuo')
        ax_hist.set_ylabel('Frequencia')
        plots["residuals_hist"] = fig_hist
            
    return plots

def save_pipeline_to_buffer(pipeline: ImbPipeline) -> io.BytesIO:
    """Serializa o pipeline para um buffer de bytes para download."""
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    return buffer
