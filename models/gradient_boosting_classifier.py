from typing import Any, Dict
from sklearn.ensemble import GradientBoostingClassifier
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> GradientBoostingClassifier:
    max_features = params.get("max_features", "None")
    if max_features == "None":
        max_features = None

    return GradientBoostingClassifier(
        random_state=random_state,
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        subsample=params.get("subsample", 1.0),
        max_features=max_features
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="Gradient Boosting",
        problem_type="Classification",
        params=[
            ParamSpec(name="n_estimators", label="Estimators", kind="int", default=100, min=50, max=2000, step=50),
            ParamSpec(name="learning_rate", label="Learning Rate", kind="float", default=0.1, min=0.001, max=1.0, step=0.001),
            ParamSpec(name="max_depth", label="Max Profundidade", kind="int", default=3, min=1, max=20, step=1),
            ParamSpec(name="min_samples_split", label="Min Samples Split", kind="int", default=2, min=2, max=50, step=1),
            ParamSpec(name="min_samples_leaf", label="Min Samples Leaf", kind="int", default=1, min=1, max=50, step=1),
            ParamSpec(name="subsample", label="Subsample", kind="float", default=1.0, min=0.5, max=1.0, step=0.05),
            ParamSpec(name="max_features", label="Max Features", kind="choice", default="None", choices=["None", "auto", "sqrt", "log2"])
        ],
        build_fn=build_model
    )
