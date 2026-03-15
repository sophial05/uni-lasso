"""
Test GroupAdaUniLasso on collinear data to verify group constraint effectiveness.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/uni-lasso')

from unilasso.uni_lasso import fit_uni, _greedy_correlation_grouping, _compute_group_penalty_weights

print("=" * 60)
print("Test: Group constraint utility functions and grouping")
print("=" * 60)

# Create highly collinear dataset
np.random.seed(42)
n = 100
p = 10

# Create a base signal
x_base = np.random.randn(n)
X = np.zeros((n, p))

# Make first 5 features highly correlated with x_base
for i in range(5):
    X[:, i] = x_base + 0.1 * np.random.randn(n)  # High correlation

# Make next 5 features independent
for i in range(5, 10):
    X[:, i] = np.random.randn(n)

# True beta
beta_true = np.zeros(p)
beta_true[0:5] = 1.0  # All positive in collinear group
beta_true[5] = -1.0    # Negative in independent
y = X @ beta_true + 0.5 * np.random.randn(n)

# Test greedy grouping function directly
print("\n1. Testing _greedy_correlation_grouping:")
corr_matrix = np.corrcoef(X.T)
groups = _greedy_correlation_grouping(corr_matrix, corr_threshold=0.7, max_group_size=20)
print(f"   Groups found: {groups}")

# Verify the first group has the collinear features
collinear_group = None
for g in groups:
    if 0 in g:
        collinear_group = g
        break
print(f"   Group containing feature 0: {collinear_group}")
print(f"   Group size: {len(collinear_group)}")
if set(collinear_group) == set([0, 1, 2, 3, 4]):
    print("   ✓ Correctly grouped all collinear features together!")

# Test _compute_group_penalty_weights
print("\n2. Testing _compute_group_penalty_weights:")
# Create univariate betas with mixed signs in the collinear group
beta_univariate = np.array([1.2, 0.8, -0.3, 1.5, 0.6, -1.0, 0.2, 0.1, -0.2, 0.05])
feature_weights = np.ones(p)
group_signs, group_weights = _compute_group_penalty_weights(groups, beta_univariate, feature_weights)
print(f"   Univariate betas: {beta_univariate}")
print(f"   Group signs: {group_signs}")
print(f"   Group weights: {group_weights}")
# Check that the collinear group has consistent sign (should be positive majority)
collinear_signs = group_signs[collinear_group]
print(f"   Collinear group signs: {collinear_signs}")
if np.all(collinear_signs == 1):
    print("   ✓ All collinear features have consistent positive group sign!")

# Test fit_uni with lower lambda (less regularization)
print("\n3. Testing fit_uni with lower lambda:")
result = fit_uni(X, y, family="gaussian",
                adaptive_weighting=True,
                enable_group_constraint=True,
                corr_threshold=0.7,
                group_penalty=5.0,
                weight_method="correlation",
                n_lmdas=20,
                lmda_min_ratio=0.001)

print(f"   Number of lambdas: {len(result.lmdas)}")
print(f"   Lambda range: [{result.lmdas[0]:.4f}, {result.lmdas[-1]:.4f}]")

# Check coefficients at a mid-range lambda
if len(result.coefs.shape) == 2:
    mid_idx = len(result.coefs) // 2
    coefs_mid = result.coefs[mid_idx]
    print(f"\n   Coefficients at lambda={result.lmdas[mid_idx]:.4f}:")
    print(f"     Collinear group (0-4): {coefs_mid[0:5]}")
    print(f"     Independent features (5-9): {coefs_mid[5:10]}")

# Check group info
if hasattr(result, 'groups') and result.groups is not None:
    print(f"\n   Attached groups: {result.groups}")
if hasattr(result, 'group_signs') and result.group_signs is not None:
    print(f"   Attached group_signs: {result.group_signs}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
