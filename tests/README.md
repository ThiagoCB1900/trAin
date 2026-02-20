# trAIn Health - Test Suite

Automated tests to validate system functionality and literature quality.

## 🧪 Test Files

### `validate_system.py`
**Complete system validation**

Tests all core functionality:
- Module imports
- Data handling (loading, splitting)
- Problem type identification
- Pipeline construction
- Model training and evaluation

**Run:**
```bash
python tests/validate_system.py
```

**Expected output:** ✓✓✓ ALL TESTS PASSED ✓✓✓

---

### `test_linear_regression_literature.py`
**Literature quality validation for Linear Regression**

Validates that the Linear Regression documentation contains:
- Portuguese content
- Mathematical formulations (OLS)
- Normalization requirements
- Multicolinearity discussion
- Clinical study references (Framingham, APACHE)

**Run:**
```bash
python tests/test_linear_regression_literature.py
```

---

### `test_xgboost_literature.py`
**Literature quality validation for XGBoost**

Validates that XGBoost documentation contains:
- Complete mathematical foundations
- Regularization explanations
- Hyperparameter descriptions
- Clinical studies
- Separate documentation for XGBoost Regressor

**Run:**
```bash
python tests/test_xgboost_literature.py
```

---

## 🚀 Running All Tests

To run all tests at once:

```bash
# Run validation
python tests/validate_system.py

# Run literature tests
python tests/test_linear_regression_literature.py
python tests/test_xgboost_literature.py
```

All tests should pass with ✓ marks.

---

## 📝 Adding New Tests

To add tests for new models:

1. Create `test_{model_name}_literature.py`
2. Follow the pattern from existing tests
3. Import from `src.ui.literature`
4. Add assertions for key content requirements

Example:
```python
from src.ui.literature import get_literature_html

def test_my_model_literature():
    html = get_literature_html('My Model', False)
    assert 'key concept' in html
    # ... more assertions
```

---

**Note:** These tests ensure documentation quality and system reliability for production use.
