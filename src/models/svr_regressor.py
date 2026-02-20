from typing import Any, Dict
from sklearn.svm import SVR
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> SVR:
    gamma = params.get("gamma", "scale")
    kernel = params.get("kernel", "rbf")

    return SVR(
        C=params.get("C", 1.0),
        kernel=kernel,
        gamma=gamma,
        degree=params.get("degree", 3),
        coef0=params.get("coef0", 0.0),
        epsilon=params.get("epsilon", 0.1)
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="SVR",
        problem_type="Regression",
        params=[
            ParamSpec(name="C", label="C", kind="float", default=1.0, min=0.001, max=100.0, step=0.001),
            ParamSpec(name="kernel", label="Kernel", kind="choice", default="rbf", choices=["rbf", "linear", "poly", "sigmoid"]),
            ParamSpec(name="gamma", label="Gamma", kind="choice", default="scale", choices=["scale", "auto"]),
            ParamSpec(name="degree", label="Degree (Poly)", kind="int", default=3, min=2, max=8, step=1),
            ParamSpec(name="coef0", label="Coef0 (Poly/Sigmoid)", kind="float", default=0.0, min=-1.0, max=1.0, step=0.1),
            ParamSpec(name="epsilon", label="Epsilon", kind="float", default=0.1, min=0.0, max=1.0, step=0.01)
        ],
        build_fn=build_model
    )
