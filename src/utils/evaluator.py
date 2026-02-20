"""
trAIn Health - Model Evaluator Module
======================================
Functions for training pipelines and evaluating model performance.
"""

import io
import logging
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)


def _compute_specificity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Compute specificity (True Negative Rate) for classification.
    
    For binary classification, specificity = TN / (TN + FP).
    For multiclass, computes per-class specificity and returns the average.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score as a float
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Binary classification
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fp = cm[0, 1]
        denom = tn + fp
        return float(tn / denom) if denom > 0 else 0.0

    # Multiclass classification
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
    problem_type: str,
) -> Dict[str, Any]:
    """
    Train pipeline on training set and evaluate on test set.
    
    Args:
        pipeline: Scikit-learn/imbalanced-learn pipeline to train
        X_train: Training feature matrix
        y_train: Training target variable
        X_test: Test feature matrix
        y_test: Test target variable
        problem_type: "Classification" or "Regression"
        
    Returns:
        Dictionary containing:
            - metrics: Dict of evaluation metrics
            - pipeline: Trained pipeline object
            - y_pred: Predictions on test set
            - y_score: Predicted probabilities/scores (if available)
    """
    logger.info(f"Training pipeline for {problem_type} problem")
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline training completed")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    metrics: Dict[str, Any] = {}
    y_score: Optional[np.ndarray] = None
    
    if problem_type == "Classification":
        logger.info("Computing classification metrics")
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        metrics["mcc"] = matthews_corrcoef(y_test, y_pred)
        metrics["specificity"] = _compute_specificity(y_test, y_pred)

        # Precision, recall, F1 metrics
        average_weighted = "weighted" if y_test.nunique() > 2 else "binary"
        metrics["precision_weighted"] = precision_score(
            y_test, y_pred, average=average_weighted, zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            y_test, y_pred, average=average_weighted, zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            y_test, y_pred, average=average_weighted, zero_division=0
        )
        metrics["precision_macro"] = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        # Probability-based metrics
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)
            try:
                metrics["log_loss"] = log_loss(y_test, y_score)
            except ValueError:
                metrics["log_loss"] = None
                logger.warning("Could not compute log loss")
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)

        # ROC AUC and PR AUC
        if y_score is not None:
            try:
                if y_test.nunique() == 2:
                    # Binary classification
                    if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                        score_vec = y_score[:, 1]
                    else:
                        score_vec = y_score
                    metrics["roc_auc"] = roc_auc_score(y_test, score_vec)
                    metrics["pr_auc"] = average_precision_score(y_test, score_vec)
                else:
                    # Multiclass classification
                    metrics["roc_auc"] = roc_auc_score(
                        y_test, y_score, multi_class="ovr", average="weighted"
                    )
                    metrics["pr_auc"] = None
            except ValueError as e:
                logger.warning(f"Could not compute ROC/PR AUC: {e}")
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None

    elif problem_type == "Regression":
        logger.info("Computing regression metrics")
        
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["mse"] = mean_squared_error(y_test, y_pred)
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["mape"] = mean_absolute_percentage_error(y_test, y_pred)
        metrics["median_ae"] = median_absolute_error(y_test, y_pred)
        metrics["r2_score"] = r2_score(y_test, y_pred)
        metrics["explained_variance"] = explained_variance_score(y_test, y_pred)

    logger.info(f"Evaluation completed with {len(metrics)} metrics")
    
    return {
        "metrics": metrics,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def generate_plots(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    problem_type: str = "Classification",
) -> Dict[str, plt.Figure]:
    """
    Generate evaluation plots for the model predictions.
    
    For classification:
        - Confusion Matrix
        - Normalized Confusion Matrix
        - ROC Curve (binary only)
        - Precision-Recall Curve (binary only)
        
    For regression:
        - Predicted vs Actual scatter plot
        - Residuals vs Predicted plot
        - Residuals distribution histogram
    
    Args:
        y_test: True test labels
        y_pred: Predicted labels
        y_score: Predicted probabilities/scores (optional, for classification)
        problem_type: "Classification" or "Regression"
        
    Returns:
        Dictionary mapping plot names to matplotlib Figure objects
    """
    logger.info(f"Generating plots for {problem_type}")
    plots: Dict[str, plt.Figure] = {}

    if problem_type == "Classification":
        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Matriz de Confusão")
        ax_cm.set_xlabel("Predito")
        ax_cm.set_ylabel("Real")
        plots["confusion_matrix"] = fig_cm

        # Normalized Confusion Matrix
        fig_cm_n, ax_cm_n = plt.subplots(figsize=(8, 6))
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax_cm_n)
        ax_cm_n.set_title("Matriz de Confusão (Normalizada)")
        ax_cm_n.set_xlabel("Predito")
        ax_cm_n.set_ylabel("Real")
        plots["confusion_matrix_norm"] = fig_cm_n

        # Probability-based plots (binary classification only)
        if y_score is not None and len(np.unique(y_test)) == 2:
            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                score_vec = y_score[:, 1]
            else:
                score_vec = y_score

            # ROC Curve
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, score_vec)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc:.2f})",
            )
            ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax_roc.set_title("Curva ROC")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            plots["roc_curve"] = fig_roc

            # Precision-Recall Curve
            fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, score_vec)
            ax_pr.plot(recall, precision, color="blue", lw=2)
            ax_pr.set_title("Curva Precision-Recall")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            plots["pr_curve"] = fig_pr

    elif problem_type == "Regression":
        # Predicted vs Actual
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        ax_scatter.scatter(y_test, y_pred, alpha=0.6, color="teal")
        min_val = min(float(y_test.min()), float(np.min(y_pred)))
        max_val = max(float(y_test.max()), float(np.max(y_pred)))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--")
        ax_scatter.set_title("Predito vs Real")
        ax_scatter.set_xlabel("Real")
        ax_scatter.set_ylabel("Predito")
        plots["pred_vs_real"] = fig_scatter

        # Residuals vs Predicted
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        ax_res.scatter(y_pred, residuals, alpha=0.6, color="purple")
        ax_res.axhline(0, color="gray", linestyle="--")
        ax_res.set_title("Residuos vs Predito")
        ax_res.set_xlabel("Predito")
        ax_res.set_ylabel("Residuo")
        plots["residuals"] = fig_res

        # Residuals Distribution
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        ax_hist.hist(residuals, bins=30, color="slateblue", alpha=0.75)
        ax_hist.set_title("Distribuicao dos Residuos")
        ax_hist.set_xlabel("Residuo")
        ax_hist.set_ylabel("Frequencia")
        plots["residuals_hist"] = fig_hist

    logger.info(f"Generated {len(plots)} plots")
    return plots


def save_pipeline_to_buffer(pipeline: ImbPipeline) -> io.BytesIO:
    """
    Serialize a trained pipeline to a BytesIO buffer for download.
    
    Args:
        pipeline: Trained scikit-learn/imbalanced-learn pipeline
        
    Returns:
        BytesIO buffer containing the serialized pipeline
    """
    logger.info("Serializing pipeline to buffer")
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    logger.info("Pipeline serialization completed")
    return buffer
