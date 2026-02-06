from typing import Dict, List
from .specs import ModelSpec, ProblemType
from .logistic_regression import get_spec as get_logistic_regression
from .random_forest_classifier import get_spec as get_rf_classifier
from .svm import get_spec as get_svm
from .linear_regression import get_spec as get_linear_regression
from .random_forest_regressor import get_spec as get_rf_regressor

MODEL_SPECS: List[ModelSpec] = [
    get_logistic_regression(),
    get_rf_classifier(),
    get_svm(),
    get_linear_regression(),
    get_rf_regressor()
]

MODEL_SPEC_MAP: Dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_SPECS}


def get_model_specs_by_problem(problem_type: ProblemType) -> List[ModelSpec]:
    return [spec for spec in MODEL_SPECS if spec.problem_type == problem_type]


def build_model(model_name: str, params: Dict[str, object], random_state: int):
    spec = MODEL_SPEC_MAP.get(model_name)
    if spec is None:
        raise ValueError(f"Modelo nao suportado: {model_name}")
    return spec.build_fn(params, random_state)
