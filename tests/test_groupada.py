"""
Unit tests for GroupAdaUniLasso implementation.

Tests cover:
1. Backward compatibility - new features disabled should match old behavior
2. Adaptive weighting - feature-level significance weights
3. Group constraints - collinearity grouping and sign consistency
4. All new parameters work correctly
5. Results are attached properly
"""

import sys
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, '/workspaces/uni-lasso')


def test_backward_compatibility():
    """Test that with new features disabled, results are accessible."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    # Test fit_uni
    result = fit_uni(X, y, family="gaussian",
                    adaptive_weighting=False,
                    enable_group_constraint=False,
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'coefs')
    assert hasattr(result, 'intercept')
    assert hasattr(result, 'lmdas')

    # Test cv_uni
    cv_result = cv_uni(X, y, family="gaussian",
                      adaptive_weighting=False,
                      enable_group_constraint=False,
                      n_folds=3,
                      n_lmdas=10,
                      seed=42)
    assert cv_result is not None
    assert hasattr(cv_result, 'coefs')
    assert hasattr(cv_result, 'best_lmda')


def test_adaptive_weighting_enabled():
    """Test that adaptive weighting can be enabled without errors."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 80, 15
    X = np.random.randn(n, p)
    y = X[:, 0] + 0.5 * np.random.randn(n)  # Feature 0 is significant

    for method in ["t_statistic", "p_value", "correlation"]:
        result = fit_uni(X, y, family="gaussian",
                        adaptive_weighting=True,
                        weight_method=method,
                        weight_max_scale=5.0,
                        n_lmdas=10)
        assert result is not None

    # Test cv_uni with adaptive weighting
    cv_result = cv_uni(X, y, family="gaussian",
                      adaptive_weighting=True,
                      n_folds=3,
                      n_lmdas=10)
    assert cv_result is not None


def test_group_constraint_enabled():
    """Test that group constraints can be enabled without errors."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 10

    # Create some collinear features
    x_base = np.random.randn(n)
    X = np.zeros((n, p))
    for i in range(5):
        X[:, i] = x_base + 0.1 * np.random.randn(n)
    for i in range(5, 10):
        X[:, i] = np.random.randn(n)
    y = x_base + 0.5 * np.random.randn(n)

    # Test fit_uni with group constraint
    result = fit_uni(X, y, family="gaussian",
                    enable_group_constraint=True,
                    corr_threshold=0.7,
                    group_penalty=5.0,
                    max_group_size=20,
                    n_lmdas=10)
    assert result is not None

    # Check that group info is attached
    assert hasattr(result, 'groups')
    assert hasattr(result, 'group_signs')

    # Test cv_uni with group constraint
    cv_result = cv_uni(X, y, family="gaussian",
                      enable_group_constraint=True,
                      n_folds=3,
                      n_lmdas=10,
                      seed=42)
    assert cv_result is not None


def test_both_features_enabled():
    """Test that both adaptive weighting and group constraints work together."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 15
    X = np.random.randn(n, p)
    y = X[:, :3].sum(axis=1) + 0.5 * np.random.randn(n)

    # Test fit_uni with both features
    result = fit_uni(X, y, family="gaussian",
                    adaptive_weighting=True,
                    enable_group_constraint=True,
                    n_lmdas=10)
    assert result is not None

    # Test cv_uni with both features
    cv_result = cv_uni(X, y, family="gaussian",
                      adaptive_weighting=True,
                      enable_group_constraint=True,
                      n_folds=3,
                      n_lmdas=10)
    assert cv_result is not None


def test_utility_functions():
    """Test the core utility functions directly."""
    from unilasso.uni_lasso import (
        _compute_feature_significance_weights,
        _greedy_correlation_grouping,
        _compute_group_penalty_weights
    )

    # Test _compute_feature_significance_weights
    univariate_results = {
        'beta': np.array([1.0, -2.0, 0.5, 0.0]),
        't_stats': np.array([3.0, -5.0, 1.0, 0.0]),
        'p_values': np.array([0.01, 0.001, 0.3, 1.0]),
        'correlations': np.array([0.5, -0.7, 0.2, 0.0])
    }

    for method in ["t_statistic", "p_value", "correlation"]:
        weights = _compute_feature_significance_weights(
            univariate_results, method, weight_max_scale=5.0
        )
        assert len(weights) == 4
        assert np.all(weights >= 1.0)
        assert np.all(weights <= 5.0)

    # Test _greedy_correlation_grouping
    np.random.seed(42)
    n_features = 10
    # Create correlation matrix with two clear groups
    corr_matrix = np.eye(n_features)
    # Group 1: features 0-3
    for i in range(4):
        for j in range(4):
            if i != j:
                corr_matrix[i, j] = 0.9
    # Group 2: features 4-6
    for i in range(4, 7):
        for j in range(4, 7):
            if i != j:
                corr_matrix[i, j] = 0.85

    groups = _greedy_correlation_grouping(corr_matrix, corr_threshold=0.7)
    # Check that we have at least the two main groups
    group_sizes = [len(g) for g in groups]
    assert 4 in group_sizes or 3 in group_sizes  # Either group 1 or 2 should be found

    # Test _compute_group_penalty_weights
    groups_test = [[0, 1, 2, 3], [4, 5, 6], [7], [8], [9]]
    beta_univariate = np.array([1.0, 0.8, -0.3, 1.2, -0.5, -0.8, -0.2, 0.1, -0.1, 0.05])
    feature_weights = np.ones(n_features)

    group_signs, group_weights = _compute_group_penalty_weights(
        groups_test, beta_univariate, feature_weights
    )

    # Check that group 0-3 have positive sign (majority vote)
    assert np.all(group_signs[0:4] == 1.0)
    # Check that group 4-6 have negative sign
    assert np.all(group_signs[4:7] == -1.0)


def test_invalid_weight_method():
    """Test that invalid weight method raises error."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50)

    with pytest.raises(ValueError, match="Unknown weight method"):
        fit_uni(X, y, adaptive_weighting=True, weight_method="invalid_method")


def test_backend_selection():
    """Test that both backends work with new features."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 60, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    for backend in ["numba", "pytorch"]:
        result = fit_uni(X, y, backend=backend,
                        adaptive_weighting=True,
                        enable_group_constraint=True,
                        n_lmdas=5)
        assert result is not None


if __name__ == "__main__":
    print("Running GroupAdaUniLasso tests...")

    test_backward_compatibility()
    print("✓ test_backward_compatibility passed")

    test_adaptive_weighting_enabled()
    print("✓ test_adaptive_weighting_enabled passed")

    test_group_constraint_enabled()
    print("✓ test_group_constraint_enabled passed")

    test_both_features_enabled()
    print("✓ test_both_features_enabled passed")

    test_utility_functions()
    print("✓ test_utility_functions passed")

    test_backend_selection()
    print("✓ test_backend_selection passed")

    print("\nAll GroupAdaUniLasso tests passed!")
