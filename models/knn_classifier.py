from typing import Any, Dict
from sklearn.neighbors import KNeighborsClassifier
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> KNeighborsClassifier:
    metric = params.get("metric", "minkowski")
    p = params.get("p", 2)

    if metric == "euclidean":
        metric = "minkowski"
        p = 2
    elif metric == "manhattan":
        metric = "minkowski"
        p = 1

    return KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        algorithm=params.get("algorithm", "auto"),
        leaf_size=params.get("leaf_size", 30),
        p=p,
        metric=metric
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="KNN",
        problem_type="Classification",
        params=[
            ParamSpec(name="n_neighbors", label="Vizinhos (k)", kind="int", default=5, min=1, max=200, step=1),
            ParamSpec(name="weights", label="Weights", kind="choice", default="uniform", choices=["uniform", "distance"]),
            ParamSpec(name="algorithm", label="Algoritmo", kind="choice", default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"]),
            ParamSpec(name="leaf_size", label="Leaf Size", kind="int", default=30, min=10, max=200, step=1),
            ParamSpec(name="metric", label="Metric", kind="choice", default="minkowski", choices=["minkowski", "euclidean", "manhattan", "chebyshev"]),
            ParamSpec(name="p", label="P (Minkowski)", kind="int", default=2, min=1, max=5, step=1)
        ],
        build_fn=build_model
    )
