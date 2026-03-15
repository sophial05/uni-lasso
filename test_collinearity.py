"""
Test GroupAdaUniLasso on collinear data to verify group constraint effectiveness.
"""

import numpy as np
import sys
sys.path.insert(0, '/workspaces/uni-lasso')

from unilasso.uni_lasso import fit_uni

print("=" * 60)
print("Test: Group constraint on collinear data")
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

# True beta: first 5 features should all have positive sign (collinear group)
beta_true = np.zeros(p)
beta_true[0:5] = 1.0  # All positive in collinear group
beta_true[5] = -1.0    # Negative in independent
y = X @ beta_true + 0.5 * np.random.randn(n)

print(f"Dataset created: {n} samples, {p} features")
print(f"  - Features 0-4: Highly collinear with each other")
print(f"  - Features 5-9: Independent")

# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)
print(f"\nCorrelation between feature 0 and 1: {corr_matrix[0, 1]:.4f}")
print(f"Correlation between feature 0 and 4: {corr_matrix[0, 4]:.4f}")
print(f"Correlation between feature 0 and 5: {corr_matrix[0, 5]:.4f}")

# Test without group constraint
print("\n" + "-" * 60)
print("1. Without group constraint:")
print("-" * 60)
result_no_group = fit_uni(X, y, family="gaussian",
                          adaptive_weighting=False,
                          enable_group_constraint=False,
                          n_lmdas=10)
coefs_no_group = result_no_group.coefs
if len(coefs_no_group.shape) == 1:
    coefs_no_group = coefs_no_group.reshape(1, -1)
# Pick middle lambda
idx = min(5, len(coefs_no_group) - 1)
coefs_no_group_mid = coefs_no_group[idx]
print(f"Coefficients for collinear group (0-4): {coefs_no_group_mid[0:5]}")
signs_no_group = np.sign(coefs_no_group_mid[0:5])
print(f"Signs: {signs_no_group}")
print(f"Sign agreement in group: {np.mean(signs_no_group == signs_no_group[0]) * 100:.1f}%")

# Test with group constraint
print("\n" + "-" * 60)
print("2. With group constraint:")
print("-" * 60)
result_with_group = fit_uni(X, y, family="gaussian",
                            adaptive_weighting=False,
                            enable_group_constraint=True,
                            corr_threshold=0.7,
                            group_penalty=10.0,
                            n_lmdas=10)
coefs_with_group = result_with_group.coefs
if len(coefs_with_group.shape) == 1:
    coefs_with_group = coefs_with_group.reshape(1, -1)
idx = min(5, len(coefs_with_group) - 1)
coefs_with_group_mid = coefs_with_group[idx]
print(f"Coefficients for collinear group (0-4): {coefs_with_group_mid[0:5]}")
signs_with_group = np.sign(coefs_with_group_mid[0:5])
print(f"Signs: {signs_with_group}")
print(f"Sign agreement in group: {np.mean(signs_with_group == signs_with_group[0]) * 100:.1f}%")

# Show groups found
if hasattr(result_with_group, 'groups') and result_with_group.groups is not None:
    print(f"\nGroups found: {result_with_group.groups}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
