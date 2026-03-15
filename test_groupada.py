"""
Test script for GroupAdaUniLasso implementation.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/uni-lasso')

from unilasso.uni_lasso import fit_uni, cv_uni

# Test 1: Basic backward compatibility test
print("=" * 60)
print("Test 1: Basic backward compatibility (no new features)")
print("=" * 60)

np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
y = np.random.randn(n)

try:
    result = fit_uni(X, y, family="gaussian", adaptive_weighting=False, enable_group_constraint=False)
    print(f"✓ fit_uni backward compatibility: PASSED")
    print(f"  - Coefficients shape: {result.coefs.shape}")
    print(f"  - Lambdas shape: {result.lmdas.shape}")
except Exception as e:
    print(f"✗ fit_uni backward compatibility: FAILED")
    print(f"  Error: {e}")

try:
    cv_result = cv_uni(X, y, family="gaussian", adaptive_weighting=False, enable_group_constraint=False, n_folds=3)
    print(f"✓ cv_uni backward compatibility: PASSED")
    print(f"  - Best lambda: {cv_result.best_lmda:.4f}")
except Exception as e:
    print(f"✗ cv_uni backward compatibility: FAILED")
    print(f"  Error: {e}")

# Test 2: Test with adaptive weighting
print("\n" + "=" * 60)
print("Test 2: With adaptive weighting enabled")
print("=" * 60)

try:
    result = fit_uni(X, y, family="gaussian", adaptive_weighting=True, weight_method="correlation", weight_max_scale=5.0)
    print(f"✓ fit_uni with adaptive weighting: PASSED")
    print(f"  - Coefficients shape: {result.coefs.shape}")
except Exception as e:
    print(f"✗ fit_uni with adaptive weighting: FAILED")
    print(f"  Error: {e}")

# Test 3: Test with group constraint
print("\n" + "=" * 60)
print("Test 3: With group constraint enabled")
print("=" * 60)

try:
    result = fit_uni(X, y, family="gaussian", enable_group_constraint=True, corr_threshold=0.3, group_penalty=2.0)
    print(f"✓ fit_uni with group constraint: PASSED")
    print(f"  - Coefficients shape: {result.coefs.shape}")
    if hasattr(result, 'groups') and result.groups is not None:
        print(f"  - Number of groups: {len(result.groups)}")
except Exception as e:
    print(f"✗ fit_uni with group constraint: FAILED")
    print(f"  Error: {e}")

# Test 4: Test with both features enabled
print("\n" + "=" * 60)
print("Test 4: With both adaptive weighting and group constraint")
print("=" * 60)

try:
    result = fit_uni(X, y, family="gaussian",
                     adaptive_weighting=True,
                     enable_group_constraint=True,
                     backend="numba")
    print(f"✓ fit_uni with both features: PASSED")
except Exception as e:
    print(f"✗ fit_uni with both features: FAILED")
    print(f"  Error: {e}")

# Test 5: Test cv_uni with new features
print("\n" + "=" * 60)
print("Test 5: cv_uni with new features")
print("=" * 60)

try:
    cv_result = cv_uni(X, y, family="gaussian",
                        adaptive_weighting=True,
                        enable_group_constraint=True,
                        n_folds=3)
    print(f"✓ cv_uni with new features: PASSED")
    print(f"  - Best lambda: {cv_result.best_lmda:.4f}")
except Exception as e:
    print(f"✗ cv_uni with new features: FAILED")
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
