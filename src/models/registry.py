from typing import Dict, List
from .specs import ModelSpec, ProblemType
from .logistic_regression import get_spec as get_logistic_regression
from .knn_classifier import get_spec as get_knn_classifier
from .naive_bayes import get_spec as get_naive_bayes
from .gradient_boosting_classifier import get_spec as get_gradient_boosting_classifier
from .decision_tree_classifier import get_spec as get_decision_tree_classifier
from .random_forest_classifier import get_spec as get_rf_classifier
from .svm import get_spec as get_svm
from .xgboost_classifier import get_spec as get_xgboost_classifier
from .linear_regression import get_spec as get_linear_regression
from .ridge_regression import get_spec as get_ridge_regression
from .decision_tree_regressor import get_spec as get_decision_tree_regressor
from .gradient_boosting_regressor import get_spec as get_gradient_boosting_regressor
from .random_forest_regressor import get_spec as get_rf_regressor
from .svr_regressor import get_spec as get_svr_regressor
from .xgboost_regressor import get_spec as get_xgboost_regressor

MODEL_SPECS: List[ModelSpec] = [
    get_logistic_regression(),
    get_knn_classifier(),
    get_naive_bayes(),
    get_gradient_boosting_classifier(),
    get_decision_tree_classifier(),
    get_rf_classifier(),
    get_svm(),
    get_xgboost_classifier(),
    get_linear_regression(),
    get_ridge_regression(),
    get_decision_tree_regressor(),
    get_gradient_boosting_regressor(),
    get_rf_regressor(),
    get_svr_regressor(),
    get_xgboost_regressor()
]

MODEL_SPEC_MAP: Dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_SPECS}


def get_model_specs_by_problem(problem_type: ProblemType) -> List[ModelSpec]:
    return [spec for spec in MODEL_SPECS if spec.problem_type == problem_type]


def build_model(model_name: str, params: Dict[str, object], random_state: int):
    spec = MODEL_SPEC_MAP.get(model_name)
    if spec is None:
        raise ValueError(f"Modelo nao suportado: {model_name}")
    return spec.build_fn(params, random_state)
