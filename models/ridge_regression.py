from typing import Any, Dict
from sklearn.linear_model import Ridge
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> Ridge:
    return Ridge(
        alpha=params.get("alpha", 1.0),
        fit_intercept=params.get("fit_intercept", True),
        solver=params.get("solver", "auto"),
        max_iter=params.get("max_iter", 1000),
        tol=params.get("tol", 1e-3),
        positive=params.get("positive", False),
        random_state=random_state
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="Ridge Regression",
        problem_type="Regression",
        params=[
            ParamSpec(name="alpha", label="Alpha", kind="float", default=1.0, min=0.0, max=100.0, step=0.1),
            ParamSpec(name="fit_intercept", label="Fit Intercept", kind="bool", default=True),
            ParamSpec(name="solver", label="Solver", kind="choice", default="auto", choices=["auto", "svd", "cholesky", "lsqr", "sag", "saga", "lbfgs"]),
            ParamSpec(name="max_iter", label="Max Iteracoes", kind="int", default=1000, min=100, max=10000, step=100),
            ParamSpec(name="tol", label="Tol", kind="float", default=1e-3, min=1e-6, max=1e-1, step=1e-4),
            ParamSpec(name="positive", label="Positive", kind="bool", default=False)
        ],
        build_fn=build_model
    )
