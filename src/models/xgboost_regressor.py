from typing import Any, Dict
from .specs import ModelSpec, ParamSpec

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - dependencia opcional
    XGBRegressor = None


def build_model(params: Dict[str, Any], random_state: int):
    if XGBRegressor is None:
        raise ImportError("XGBoost nao instalado. Instale com: pip install xgboost")

    return XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 6),
        min_child_weight=params.get("min_child_weight", 1.0),
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0),
        gamma=params.get("gamma", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0)
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="XGBoost Regressor",
        problem_type="Regression",
        params=[
            ParamSpec(name="n_estimators", label="Estimators", kind="int", default=300, min=50, max=2000, step=50),
            ParamSpec(name="learning_rate", label="Learning Rate", kind="float", default=0.1, min=0.001, max=1.0, step=0.001),
            ParamSpec(name="max_depth", label="Max Profundidade", kind="int", default=6, min=1, max=20, step=1),
            ParamSpec(name="min_child_weight", label="Min Child Weight", kind="float", default=1.0, min=0.1, max=10.0, step=0.1),
            ParamSpec(name="subsample", label="Subsample", kind="float", default=1.0, min=0.5, max=1.0, step=0.05),
            ParamSpec(name="colsample_bytree", label="Colsample Bytree", kind="float", default=1.0, min=0.5, max=1.0, step=0.05),
            ParamSpec(name="gamma", label="Gamma", kind="float", default=0.0, min=0.0, max=10.0, step=0.1),
            ParamSpec(name="reg_lambda", label="Reg Lambda", kind="float", default=1.0, min=0.0, max=10.0, step=0.1),
            ParamSpec(name="reg_alpha", label="Reg Alpha", kind="float", default=0.0, min=0.0, max=10.0, step=0.1)
        ],
        build_fn=build_model
    )
