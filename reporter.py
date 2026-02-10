import pandas as pd
from datetime import datetime
from typing import Dict, Any

def generate_report(
    dataset_info: Dict[str, Any],
    user_params: Dict[str, Any],
    model_params: Dict[str, Any],
    metrics: Dict[str, float],
    problem_type: str,
    random_state: int
) -> str:
    """
    Gera um relatório detalhado em formato TXT com todas as informações do experimento.
    """
    
    report_content = []
    
    report_content.append("=====================================================")
    report_content.append(" RELATORIO DE EXPERIMENTO DE MACHINE LEARNING (TCC) ")
    report_content.append("=====================================================")
    report_content.append(f"Timestamp da Execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Seed (Semente) Fixa: {random_state}")
    report_content.append("-----------------------------------------------------")
    
    # 1. Informações do Dataset
    report_content.append("\n[1] INFORMACOES DO DATASET")
    report_content.append(f"  Nome do Arquivo: {dataset_info.get('filename', 'N/A')}")
    report_content.append(f"  Variável Target: {dataset_info.get('target_column', 'N/A')}")
    report_content.append(f"  Tipo de Problema: {problem_type}")
    report_content.append(f"  Total de Amostras: {dataset_info.get('total_samples', 'N/A')}")
    report_content.append(f"  Total de Features: {dataset_info.get('total_features', 'N/A')}")
    if problem_type == "Classification":
        train_dist = dataset_info.get("class_distribution_train")
        test_dist = dataset_info.get("class_distribution_test")
        if train_dist:
            report_content.append(f"  Distribuicao Classes (Treino): {train_dist}")
        if test_dist:
            report_content.append(f"  Distribuicao Classes (Teste): {test_dist}")
    elif problem_type == "Regression":
        target_stats = dataset_info.get("target_stats")
        if target_stats:
            report_content.append("  Estatisticas Target (Treino):")
            report_content.append(f"    Media: {target_stats.get('mean', 'N/A')}")
            report_content.append(f"    Desvio Padrao: {target_stats.get('std', 'N/A')}")
            report_content.append(f"    Min: {target_stats.get('min', 'N/A')}")
            report_content.append(f"    Max: {target_stats.get('max', 'N/A')}")
    
    # 2. Parâmetros e Técnicas Escolhidas
    report_content.append("\n[2] PARAMETROS E TECNICAS ESCOLHIDAS")
    report_content.append(f"  Proporção Treino/Teste: {1.0 - user_params['test_size']:.0%} / {user_params['test_size']:.0%}")
    report_content.append(f"  Técnica de Normalização (Scaler): {user_params['scaler_name']}")
    report_content.append(f"  Técnica de Balanceamento (Sampler): {user_params['sampler_name']} (Aplicado apenas no Treino)")
    report_content.append(f"  Algoritmo de Treinamento: {user_params['model_name']}")
    if model_params:
        report_content.append("  Parametros do Modelo:")
        for key, value in model_params.items():
            report_content.append(f"    {key}: {value}")
    
    # 3. Avaliação de Desempenho
    report_content.append("\n[3] AVALIACAO DE DESEMPENHO (Conjunto de Teste Intocado)")
    
    if problem_type == "Classification":
        report_content.append("  Metricas de Classificacao:")
        report_content.append(f"    Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        report_content.append(f"    Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
        report_content.append(f"    Specificity (TNR): {metrics.get('specificity', 0.0):.4f}")
        report_content.append(f"    Precision (Weighted): {metrics.get('precision_weighted', 0.0):.4f}")
        report_content.append(f"    Recall (Weighted): {metrics.get('recall_weighted', 0.0):.4f}")
        report_content.append(f"    F1-Score (Weighted): {metrics.get('f1_weighted', 0.0):.4f}")
        report_content.append(f"    Precision (Macro): {metrics.get('precision_macro', 0.0):.4f}")
        report_content.append(f"    Recall (Macro): {metrics.get('recall_macro', 0.0):.4f}")
        report_content.append(f"    F1-Score (Macro): {metrics.get('f1_macro', 0.0):.4f}")
        if metrics.get("roc_auc") is not None:
            report_content.append(f"    ROC AUC: {metrics.get('roc_auc'):.4f}")
        if metrics.get("pr_auc") is not None:
            report_content.append(f"    PR AUC: {metrics.get('pr_auc'):.4f}")
        if metrics.get("log_loss") is not None:
            report_content.append(f"    Log Loss: {metrics.get('log_loss'):.4f}")
        if metrics.get("mcc") is not None:
            report_content.append(f"    MCC: {metrics.get('mcc'):.4f}")
    elif problem_type == "Regression":
        report_content.append("  Metricas de Regressao:")
        report_content.append(f"    RMSE (Root Mean Squared Error): {metrics.get('rmse', 0.0):.4f}")
        report_content.append(f"    MSE (Mean Squared Error): {metrics.get('mse', 0.0):.4f}")
        report_content.append(f"    MAE (Mean Absolute Error): {metrics.get('mae', 0.0):.4f}")
        report_content.append(f"    MAPE (Mean Absolute Percentage Error): {metrics.get('mape', 0.0):.4f}")
        report_content.append(f"    MedAE (Median Absolute Error): {metrics.get('median_ae', 0.0):.4f}")
        report_content.append(f"    R2 (Coefficient of Determination): {metrics.get('r2_score', 0.0):.4f}")
        report_content.append(f"    Explained Variance: {metrics.get('explained_variance', 0.0):.4f}")
        
    report_content.append("\n=====================================================")
    report_content.append(" FIM DO RELATÓRIO")
    report_content.append("=====================================================")
    
    return "\n".join(report_content)
