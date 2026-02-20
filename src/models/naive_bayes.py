from typing import Any, Dict
from sklearn.naive_bayes import GaussianNB
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> GaussianNB:
    return GaussianNB(
        var_smoothing=params.get("var_smoothing", 1e-9)
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="Naive Bayes",
        problem_type="Classification",
        params=[
            ParamSpec(name="var_smoothing", label="Var Smoothing", kind="float", default=1e-9, min=1e-12, max=1e-6, step=1e-12)
        ],
        build_fn=build_model
    )
