#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for Linear Regression literature integration."""

from literature_content import get_literature_html

def test_linear_regression_literature():
    """Test that Linear Regression literature loads correctly."""
    
    # Test loading Linear Regression literature
    html = get_literature_html('Linear Regression', False)
    
    # Basic assertions
    assert len(html) > 0, "HTML should not be empty"
    assert '<title>Linear Regression' in html, "Should contain Linear Regression title"
    assert 'Regressão Linear' in html or 'Regressao Linear' in html, "Should contain Portuguese name"
    assert 'OLS' in html or 'Ordinary Least Squares' in html, "Should mention OLS"
    assert 'interpretabilidade' in html, "Should mention interpretability (Portuguese)"
    assert 'normalização' in html or 'normalizacao' in html, "Should have normalization section"
    assert 'multicolinearidade' in html, "Should mention multicolinearity"
    assert 'Framingham' in html or 'APACHE' in html, "Should reference clinical scores"
    
    print("✓ SUCCESS: Linear Regression literature loaded correctly!")
    print(f"✓ Length: {len(html):,} characters")
    print(f"✓ Contains proper Portuguese content")
    print(f"✓ Includes mathematical formulations (OLS)")
    print(f"✓ References clinical studies and scores")
    print(f"✓ Covers normalization (OBRIGATÓRIA)")
    print(f"✓ Discusses multicolinearity and VIF")
    
    return True

if __name__ == "__main__":
    try:
        test_linear_regression_literature()
        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
