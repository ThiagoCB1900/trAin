from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> LogisticRegression:
    penalty = params.get("penalty", "l2")
    solver = params.get("solver", "lbfgs")
    l1_ratio = params.get("l1_ratio", None)

    if penalty == "none":
        penalty = None

    # Ajustes simples para combinacoes invalidadas
    if penalty in ("l1", "elasticnet") and solver not in ("liblinear", "saga"):
        solver = "saga"
    if penalty == "elasticnet":
        if solver != "saga":
            solver = "saga"
        if l1_ratio is None:
            l1_ratio = 0.5
    else:
        l1_ratio = None

    if penalty is None and solver == "liblinear":
        solver = "lbfgs"

    return LogisticRegression(
        random_state=random_state,
        C=params.get("C", 1.0),
        max_iter=params.get("max_iter", 1000),
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        class_weight=None if params.get("class_weight", "None") == "None" else "balanced",
        fit_intercept=params.get("fit_intercept", True)
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="Logistic Regression",
        problem_type="Classification",
        params=[
            ParamSpec(name="C", label="C (Regularizacao)", kind="float", default=1.0, min=0.001, max=100.0, step=0.001),
            ParamSpec(name="penalty", label="Penalty", kind="choice", default="l2", choices=["l2", "l1", "elasticnet", "none"]),
            ParamSpec(name="solver", label="Solver", kind="choice", default="lbfgs", choices=["lbfgs", "liblinear", "saga"]),
            ParamSpec(name="l1_ratio", label="L1 Ratio (ElasticNet)", kind="float", default=0.5, min=0.0, max=1.0, step=0.01),
            ParamSpec(name="class_weight", label="Class Weight", kind="choice", default="None", choices=["None", "balanced"]),
            ParamSpec(name="fit_intercept", label="Fit Intercept", kind="bool", default=True),
            ParamSpec(name="max_iter", label="Max Iteracoes", kind="int", default=1000, min=100, max=10000, step=100)
        ],
        build_fn=build_model
    )
