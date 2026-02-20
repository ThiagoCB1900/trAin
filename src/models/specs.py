from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

ProblemType = Literal["Classification", "Regression"]
ParamKind = Literal["int", "float", "bool", "choice"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: ParamKind
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    problem_type: ProblemType
    params: List[ParamSpec]
    build_fn: Callable[[Dict[str, Any], int], Any]
