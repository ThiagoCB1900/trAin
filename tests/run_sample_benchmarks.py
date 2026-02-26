"""
Run reproducible baseline benchmarks on bundled sample datasets.

Outputs:
- docs/paper_sbcas2026/results/classification_heart_results.csv
- docs/paper_sbcas2026/results/regression_insurance_results.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.data_handler import identify_problem_type, load_data, separate_features_target, split_data
from src.core.pipeline_builder import build_pipeline
from src.models.registry import get_model_specs_by_problem
from src.utils.evaluator import evaluate_model

RESULTS_DIR = ROOT / "docs" / "paper_sbcas2026" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _default_params_from_spec(spec: Any) -> dict[str, Any]:
    return {param.name: param.default for param in spec.params}


def run_classification_benchmark() -> pd.DataFrame:
    data_path = ROOT / "sample_data" / "heart_statlog_cleveland_hungary_final.csv"
    df = load_data(str(data_path))

    target_col = "target"
    X, y = separate_features_target(df, target_col)

    problem_type = identify_problem_type(y)
    assert problem_type == "Classification", f"Expected Classification, got {problem_type}"

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, problem_type=problem_type
    )

    rows: list[dict[str, Any]] = []

    for spec in get_model_specs_by_problem("Classification"):
        model_name = spec.name
        params = _default_params_from_spec(spec)

        try:
            pipeline = build_pipeline(
                scaler_name="StandardScaler",
                sampler_name="None",
                model_name=model_name,
                model_params=params,
                problem_type="Classification",
                random_state=42,
            )
            result = evaluate_model(
                pipeline, X_train, y_train, X_test, y_test, "Classification"
            )
            metrics = result["metrics"]

            rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics.get("accuracy"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "f1_weighted": metrics.get("f1_weighted"),
                    "mcc": metrics.get("mcc"),
                    "specificity": metrics.get("specificity"),
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "model": model_name,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "f1_weighted": None,
                    "mcc": None,
                    "specificity": None,
                    "roc_auc": None,
                    "pr_auc": None,
                    "status": "error",
                    "error": str(exc),
                }
            )

    df_results = pd.DataFrame(rows).sort_values(by=["accuracy"], ascending=False, na_position="last")
    out_path = RESULTS_DIR / "classification_heart_results.csv"
    df_results.to_csv(out_path, index=False)
    return df_results


def run_regression_benchmark() -> pd.DataFrame:
    data_path = ROOT / "sample_data" / "insurance_encoded.csv"
    df = load_data(str(data_path))

    target_col = "charges"
    X, y = separate_features_target(df, target_col)

    problem_type = identify_problem_type(y)
    assert problem_type == "Regression", f"Expected Regression, got {problem_type}"

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, problem_type=problem_type
    )

    rows: list[dict[str, Any]] = []

    for spec in get_model_specs_by_problem("Regression"):
        model_name = spec.name
        params = _default_params_from_spec(spec)

        try:
            pipeline = build_pipeline(
                scaler_name="StandardScaler",
                sampler_name="None",
                model_name=model_name,
                model_params=params,
                problem_type="Regression",
                random_state=42,
            )
            result = evaluate_model(
                pipeline, X_train, y_train, X_test, y_test, "Regression"
            )
            metrics = result["metrics"]

            rows.append(
                {
                    "model": model_name,
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "mse": metrics.get("mse"),
                    "r2_score": metrics.get("r2_score"),
                    "explained_variance": metrics.get("explained_variance"),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "model": model_name,
                    "rmse": None,
                    "mae": None,
                    "mse": None,
                    "r2_score": None,
                    "explained_variance": None,
                    "status": "error",
                    "error": str(exc),
                }
            )

    df_results = pd.DataFrame(rows).sort_values(by=["rmse"], ascending=True, na_position="last")
    out_path = RESULTS_DIR / "regression_insurance_results.csv"
    df_results.to_csv(out_path, index=False)
    return df_results


def main() -> None:
    print("=" * 70)
    print("Sample benchmarks for trAIn Health")
    print("=" * 70)

    cls_df = run_classification_benchmark()
    reg_df = run_regression_benchmark()

    print("\nTop 3 (classification by accuracy):")
    print(cls_df[["model", "accuracy", "f1_weighted", "roc_auc", "status"]].head(3).to_string(index=False))

    print("\nTop 3 (regression by RMSE):")
    print(reg_df[["model", "rmse", "r2_score", "status"]].head(3).to_string(index=False))

    print("\nSaved result files in docs/paper_sbcas2026/results/")


if __name__ == "__main__":
    main()
