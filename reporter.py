import pandas as pd
from datetime import datetime
from typing import Dict, Any

def generate_report(
    dataset_info: Dict[str, Any],
    user_params: Dict[str, Any],
    metrics: Dict[str, float],
    problem_type: str,
    random_state: int
) -> str:
    """
    Gera um relatório detalhado em formato TXT com todas as informações do experimento.
    """
    
    report_content = []
    
    report_content.append("=====================================================")
    report_content.append(" RELATÓRIO DE EXPERIMENTO DE MACHINE LEARNING (TCC) ")
    report_content.append("=====================================================")
    report_content.append(f"Timestamp da Execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Seed (Semente) Fixa: {random_state}")
    report_content.append("-----------------------------------------------------")
    
    # 1. Informações do Dataset
    report_content.append("\n[1] INFORMAÇÕES DO DATASET")
    report_content.append(f"  Nome do Arquivo: {dataset_info.get('filename', 'N/A')}")
    report_content.append(f"  Variável Target: {dataset_info.get('target_column', 'N/A')}")
    report_content.append(f"  Tipo de Problema: {problem_type}")
    report_content.append(f"  Total de Amostras: {dataset_info.get('total_samples', 'N/A')}")
    report_content.append(f"  Total de Features: {dataset_info.get('total_features', 'N/A')}")
    
    # 2. Parâmetros e Técnicas Escolhidas
    report_content.append("\n[2] PARÂMETROS E TÉCNICAS ESCOLHIDAS")
    report_content.append(f"  Proporção Treino/Teste: {1.0 - user_params['test_size']:.0%} / {user_params['test_size']:.0%}")
    report_content.append(f"  Técnica de Normalização (Scaler): {user_params['scaler_name']}")
    report_content.append(f"  Técnica de Balanceamento (Sampler): {user_params['sampler_name']} (Aplicado apenas no Treino)")
    report_content.append(f"  Algoritmo de Treinamento: {user_params['model_name']}")
    
    # 3. Avaliação de Desempenho
    report_content.append("\n[3] AVALIAÇÃO DE DESEMPENHO (Conjunto de Teste Intocado)")
    
    if problem_type == "Classification":
        report_content.append("  Métricas de Classificação:")
        report_content.append(f"    Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        report_content.append(f"    Precision (Weighted): {metrics.get('precision', 0.0):.4f}")
        report_content.append(f"    Recall (Weighted): {metrics.get('recall', 0.0):.4f}")
        report_content.append(f"    F1-Score (Weighted): {metrics.get('f1_score', 0.0):.4f}")
    elif problem_type == "Regression":
        report_content.append("  Métricas de Regressão:")
        report_content.append(f"    RMSE (Root Mean Squared Error): {metrics.get('rmse', 0.0):.4f}")
        report_content.append(f"    MAE (Mean Absolute Error): {metrics.get('mae', 0.0):.4f}")
        report_content.append(f"    R² (Coefficient of Determination): {metrics.get('r2_score', 0.0):.4f}")
        
    report_content.append("\n=====================================================")
    report_content.append(" FIM DO RELATÓRIO")
    report_content.append("=====================================================")
    
    return "\n".join(report_content)
