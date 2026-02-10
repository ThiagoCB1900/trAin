from typing import Any, Dict
from sklearn.tree import DecisionTreeRegressor
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> DecisionTreeRegressor:
    max_depth = params.get("max_depth", 0)
    max_features = params.get("max_features", "None")
    if max_features == "None":
        max_features = None

    return DecisionTreeRegressor(
        random_state=random_state,
        criterion=params.get("criterion", "squared_error"),
        max_depth=None if max_depth in (0, None) else int(max_depth),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=max_features
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="Decision Tree Regressor",
        problem_type="Regression",
        params=[
            ParamSpec(name="criterion", label="Criterion", kind="choice", default="squared_error", choices=["squared_error", "friedman_mse"]),
            ParamSpec(name="max_depth", label="Max Profundidade (0 = None)", kind="int", default=0, min=0, max=200, step=1),
            ParamSpec(name="min_samples_split", label="Min Samples Split", kind="int", default=2, min=2, max=50, step=1),
            ParamSpec(name="min_samples_leaf", label="Min Samples Leaf", kind="int", default=1, min=1, max=50, step=1),
            ParamSpec(name="max_features", label="Max Features", kind="choice", default="None", choices=["None", "sqrt", "log2"])
        ],
        build_fn=build_model
    )
