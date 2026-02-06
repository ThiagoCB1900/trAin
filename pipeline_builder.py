from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Any, Dict, List, Literal

from models.registry import build_model, get_model_specs_by_problem

ProblemType = Literal["Classification", "Regression"]
ScalerName = Literal["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
SamplerName = Literal["None", "RandomOverSampler", "RandomUnderSampler", "SMOTE"]
ModelName = str

def get_scaler(scaler_name: ScalerName) -> Any:
    """Retorna a instância do scaler scikit-learn selecionado."""
    if scaler_name == "StandardScaler":
        return StandardScaler()
    elif scaler_name == "MinMaxScaler":
        return MinMaxScaler()
    elif scaler_name == "RobustScaler":
        return RobustScaler()
    return "passthrough" # scikit-learn/imblearn pipeline aceita "passthrough" para pular a etapa

def get_sampler(sampler_name: SamplerName, random_state: int) -> Any:
    """Retorna a instância do sampler imbalanced-learn selecionado."""
    if sampler_name == "RandomOverSampler":
        return RandomOverSampler(random_state=random_state)
    elif sampler_name == "RandomUnderSampler":
        return RandomUnderSampler(random_state=random_state)
    elif sampler_name == "SMOTE":
        return SMOTE(random_state=random_state)
    return "passthrough"

def get_model(model_name: ModelName, params: Dict[str, Any], random_state: int) -> Any:
    """Retorna a instância do modelo scikit-learn selecionado com seed fixa."""
    return build_model(model_name, params, random_state)

def build_pipeline(
    scaler_name: ScalerName, 
    sampler_name: SamplerName, 
    model_name: ModelName,
    model_params: Dict[str, Any],
    problem_type: ProblemType, 
    random_state: int
) -> ImbPipeline:
    """
    Constrói o pipeline de ML, garantindo que o sampler seja usado apenas para Classificação.
    """
    scaler = get_scaler(scaler_name)
    model = get_model(model_name, model_params, random_state)
    
    steps = [
        ('scaler', scaler),
    ]
    
    # O balanceamento é aplicado APENAS se for Classificação
    if problem_type == "Classification":
        sampler = get_sampler(sampler_name, random_state)
        steps.append(('sampler', sampler))
    elif sampler_name != "None":
        # Aviso metodológico: se o usuário escolheu um sampler para Regressão, ele será ignorado
        # A UI deve impedir isso, mas o backend garante a corretude.
        print(f"Aviso: Sampler '{sampler_name}' ignorado para problema de Regressão.")
    
    steps.append(('model', model))
    
    # Usa ImbPipeline para suportar samplers
    return ImbPipeline(steps)

# Mapeamento de modelos por tipo de problema para uso na UI
MODEL_MAP: Dict[ProblemType, List[ModelName]] = {
    "Classification": [spec.name for spec in get_model_specs_by_problem("Classification")],
    "Regression": [spec.name for spec in get_model_specs_by_problem("Regression")]
}

SCALER_OPTIONS: list[ScalerName] = ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
SAMPLER_OPTIONS: list[SamplerName] = ["None", "RandomOverSampler", "RandomUnderSampler", "SMOTE"]
