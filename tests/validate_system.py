"""
Quick validation script to test all core functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

print("=" * 60)
print("trAIn Health - System Validation")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from src.core.data_handler import load_data, identify_problem_type, split_data, separate_features_target
    from src.core.pipeline_builder import build_pipeline, SCALER_OPTIONS, SAMPLER_OPTIONS
    from src.utils.evaluator import evaluate_model, generate_plots
    from src.utils.reporter import generate_report
    from src.models.registry import get_model_specs_by_problem, build_model
    from src.ui.literature import get_literature_html
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic data
print("\n[2/6] Creating synthetic dataset...")
try:
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
    })
    y_classification = pd.Series(np.random.choice([0, 1], 100))
    y_regression = pd.Series(np.random.randn(100) * 10 + 50)
    print("✓ Synthetic data created")
except Exception as e:
    print(f"✗ Data creation failed: {e}")
    sys.exit(1)

# Test 3: Test problem type identification
print("\n[3/6] Testing problem type identification...")
try:
    prob_type_class = identify_problem_type(y_classification)
    prob_type_reg = identify_problem_type(y_regression)
    assert prob_type_class == "Classification", "Classification detection failed"
    assert prob_type_reg == "Regression", "Regression detection failed"
    print(f"✓ Classification: {prob_type_class}")
    print(f"✓ Regression: {prob_type_reg}")
except Exception as e:
    print(f"✗ Problem type identification failed: {e}")
    sys.exit(1)

# Test 4: Test data splitting
print("\n[4/6] Testing data splitting...")
try:
    X_train, X_test, y_train, y_test = split_data(
        X, y_classification, test_size=0.2, random_state=42, 
        problem_type="Classification"
    )
    assert len(X_train) == 80, f"Train size wrong: {len(X_train)}"
    assert len(X_test) == 20, f"Test size wrong: {len(X_test)}"
    print(f"✓ Split successful: {len(X_train)} train, {len(X_test)} test")
except Exception as e:
    print(f"✗ Data splitting failed: {e}")
    sys.exit(1)

# Test 5: Test pipeline building
print("\n[5/6] Testing pipeline construction...")
try:
    pipeline = build_pipeline(
        scaler_name="StandardScaler",
        sampler_name="None",
        model_name="Logistic Regression",
        model_params={"C": 1.0, "max_iter": 100},
        problem_type="Classification",
        random_state=42
    )
    print("✓ Pipeline created successfully")
    print(f"  Steps: {[name for name, _ in pipeline.steps]}")
except Exception as e:
    print(f"✗ Pipeline construction failed: {e}")
    sys.exit(1)

# Test 6: Test model training and evaluation
print("\n[6/6] Testing model training...")
try:
    results = evaluate_model(
        pipeline, X_train, y_train, X_test, y_test, "Classification"
    )
    assert "metrics" in results, "No metrics in results"
    assert "accuracy" in results["metrics"], "No accuracy metric"
    print(f"✓ Model trained successfully")
    print(f"  Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"  Metrics: {list(results['metrics'].keys())[:5]}...")
except Exception as e:
    print(f"✗ Model training failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("=" * 60)
print("\nThe system is fully functional and ready for use!")
