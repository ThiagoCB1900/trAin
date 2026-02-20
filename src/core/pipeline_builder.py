"""
trAIn Health - Pipeline Builder Module
=======================================
Functions for building ML pipelines with preprocessing and models.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Any, Dict, List, Literal, Union
import logging

from src.models.registry import build_model

logger = logging.getLogger(__name__)

ProblemType = Literal["Classification", "Regression"]
ScalerName = Literal["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
SamplerName = Literal["None", "RandomOverSampler", "RandomUnderSampler", "SMOTE"]
ModelName = str


def get_scaler(scaler_name: ScalerName) -> Union[Any, str]:
    """
    Get the selected scikit-learn scaler instance.
    
    Args:
        scaler_name: Name of the scaler to use
        
    Returns:
        Scaler instance or "passthrough" to skip this step
    """
    scalers = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
    }
    
    if scaler_name in scalers:
        return scalers[scaler_name]()
    
    return "passthrough"


def get_sampler(sampler_name: SamplerName, random_state: int) -> Union[Any, str]:
    """
    Get the selected imbalanced-learn sampler instance.
    
    Args:
        sampler_name: Name of the sampler to use
        random_state: Random seed for reproducibility
        
    Returns:
        Sampler instance or "passthrough" to skip this step
    """
    samplers = {
        "RandomOverSampler": RandomOverSampler,
        "RandomUnderSampler": RandomUnderSampler,
        "SMOTE": SMOTE,
    }
    
    if sampler_name in samplers:
        return samplers[sampler_name](random_state=random_state)
    
    return "passthrough"


def build_pipeline(
    scaler_name: ScalerName, 
    sampler_name: SamplerName, 
    model_name: ModelName,
    model_params: Dict[str, Any],
    problem_type: ProblemType, 
    random_state: int
) -> ImbPipeline:
    """
    Build a complete ML pipeline with preprocessing and model.
    
    Args:
        scaler_name: Scaler to use for normalization
        sampler_name: Sampler for class balancing (Classification only)
        model_name: Name of the ML model
        model_params: Hyperparameters for the model
        problem_type: "Classification" or "Regression"
        random_state: Random seed for reproducibility
        
    Returns:
        Complete imbalanced-learn Pipeline
    """
    scaler = get_scaler(scaler_name)
    model = build_model(model_name, model_params, random_state)
    
    steps = [('scaler', scaler)]
    
    # Balancing is ONLY applied for Classification
    if problem_type == "Classification":
        sampler = get_sampler(sampler_name, random_state)
        steps.append(('sampler', sampler))
    elif sampler_name != "None":
        logger.warning(
            f"Sampler '{sampler_name}' ignored for Regression problem."
        )
    
    steps.append(('model', model))
    
    return ImbPipeline(steps)
