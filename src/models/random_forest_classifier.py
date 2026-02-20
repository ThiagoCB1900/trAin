from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from .specs import ModelSpec, ParamSpec


def build_model(params: Dict[str, Any], random_state: int) -> RandomForestClassifier:
    max_depth = params.get("max_depth", 0)
    max_features = params.get("max_features", "sqrt")
    if max_features == "None":
        max_features = None
    class_weight = params.get("class_weight", "None")
    if class_weight == "None":
        class_weight = None
    return RandomForestClassifier(
        random_state=random_state,
        n_estimators=params.get("n_estimators", 200),
        max_depth=None if max_depth in (0, None) else int(max_depth),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=max_features,
        bootstrap=params.get("bootstrap", True),
        class_weight=class_weight
    )


def get_spec() -> ModelSpec:
    return ModelSpec(
        name="RandomForestClassifier",
        problem_type="Classification",
        params=[
            ParamSpec(name="n_estimators", label="Arvores", kind="int", default=200, min=50, max=2000, step=50),
            ParamSpec(name="max_depth", label="Max Profundidade (0 = None)", kind="int", default=0, min=0, max=200, step=1),
            ParamSpec(name="min_samples_split", label="Min Samples Split", kind="int", default=2, min=2, max=50, step=1),
            ParamSpec(name="min_samples_leaf", label="Min Samples Leaf", kind="int", default=1, min=1, max=50, step=1),
            ParamSpec(name="max_features", label="Max Features", kind="choice", default="sqrt", choices=["sqrt", "log2", "None"]),
            ParamSpec(name="bootstrap", label="Bootstrap", kind="bool", default=True),
            ParamSpec(name="class_weight", label="Class Weight", kind="choice", default="None", choices=["None", "balanced", "balanced_subsample"])
        ],
        build_fn=build_model
    )
