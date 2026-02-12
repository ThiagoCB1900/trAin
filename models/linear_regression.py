from typing import Any, Dict
from sklearn.linear_model import LinearRegression
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> LinearRegression:
    n_jobs = params.get("n_jobs", 0)
    if n_jobs == 0:
        n_jobs = None
    return LinearRegression(
        fit_intercept=params.get("fit_intercept", True),
        n_jobs=n_jobs,
        positive=params.get("positive", False)
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="Linear Regression",
        problem_type="Regression",
        params=[
            ParamSpec(name="fit_intercept", label="Fit Intercept", kind="bool", default=True),
            ParamSpec(name="positive", label="Positive", kind="bool", default=False),
            ParamSpec(name="n_jobs", label="N Jobs (0 = None)", kind="int", default=0, min=0, max=32, step=1)
        ],
        build_fn=build_model
    )
