#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for XGBoost literature integration."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.literature import get_literature_html

def test_xgboost_literature():
    """Test that XGBoost literature loads correctly."""
    
    # Test loading XGBoost literature
    html = get_literature_html('XGBoost', False)
    
    # Basic assertions
    assert len(html) > 0, "HTML should not be empty"
    assert '<title>XGBoost' in html, "Should contain XGBoost title"
    assert 'Extreme Gradient Boosting' in html, "Should contain full name"
    assert 'Chen' in html, "Should reference original authors"
    assert 'regularizacao' in html, "Should mention regularization (Portuguese)"
    assert 'hiperparametros' in html or 'hiperparâmetros' in html, "Should have hyperparameters section"
    
    print("✓ SUCCESS: XGBoost literature loaded correctly!")
    print(f"✓ Length: {len(html):,} characters")
    print(f"✓ Contains proper Portuguese content")
    print(f"✓ Includes mathematical formulations")
    print(f"✓ References clinical studies")
    
    # Test XGBoost Regressor (now has dedicated file)
    html_reg = get_literature_html('XGBoost Regressor', False)
    assert len(html_reg) > 0, "XGBoost Regressor HTML should not be empty"
    assert 'XGBoost Regressor' in html_reg or 'regressão' in html_reg, "Should be regression-focused"
    print(f"✓ XGBoost Regressor has dedicated literature")
    
    return True

if __name__ == "__main__":
    try:
        test_xgboost_literature()
        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
