"""
Unit tests for full GLM family support in GroupAdaUniLasso.

Tests cover:
1. Gaussian regression (already supported)
2. Binomial classification (logistic regression)
3. Poisson regression (count data)
4. Backward compatibility
5. All new features with different GLM families
"""

import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/workspaces/uni-lasso')


def test_gaussian_regression_basic():
    """Test that Gaussian regression works as before."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(n)

    # Test fit_uni with Gaussian
    result = fit_uni(X, y, family="gaussian",
                    adaptive_weighting=False,
                    enable_group_constraint=False,
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'coefs')
    assert hasattr(result, 'intercept')
    assert hasattr(result, 'lmdas')
    assert result.family == "gaussian"

    # Test cv_uni with Gaussian
    cv_result = cv_uni(X, y, family="gaussian",
                      adaptive_weighting=False,
                      enable_group_constraint=False,
                      n_folds=3,
                      n_lmdas=10,
                      seed=42)
    assert cv_result is not None
    assert hasattr(cv_result, 'coefs')
    assert hasattr(cv_result, 'best_lmda')
    assert cv_result.family == "gaussian"


def test_gaussian_with_new_features():
    """Test Gaussian regression with adaptive weighting and group constraints."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 80, 15

    # Create some collinear features
    x_base = np.random.randn(n)
    X = np.zeros((n, p))
    for i in range(5):
        X[:, i] = x_base + 0.1 * np.random.randn(n)
    for i in range(5, 10):
        X[:, i] = np.random.randn(n)
    for i in range(10, 15):
        X[:, i] = np.random.randn(n)

    y = x_base + 0.3 * X[:, 5] + 0.5 * np.random.randn(n)

    # Test with both features enabled
    result = fit_uni(X, y, family="gaussian",
                    adaptive_weighting=True,
                    enable_group_constraint=True,
                    weight_method="correlation",
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'groups')
    assert hasattr(result, 'group_signs')

    # Test cv_uni with both features
    cv_result = cv_uni(X, y, family="gaussian",
                      adaptive_weighting=True,
                      enable_group_constraint=True,
                      n_folds=3,
                      n_lmdas=10,
                      seed=42)
    assert cv_result is not None


def test_binomial_classification():
    """Test binomial (logistic) classification support."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 15
    X = np.random.randn(n, p)

    # Create binary outcome
    logits = X[:, 0] - 0.5 * X[:, 1]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n) < probs).astype(float)

    # Test fit_uni with binomial
    result = fit_uni(X, y, family="binomial",
                    adaptive_weighting=False,
                    enable_group_constraint=False,
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'coefs')
    assert result.family == "binomial"

    # Test cv_uni with binomial
    cv_result = cv_uni(X, y, family="binomial",
                      adaptive_weighting=False,
                      enable_group_constraint=False,
                      n_folds=3,
                      n_lmdas=10,
                      seed=42)
    assert cv_result is not None
    assert cv_result.family == "binomial"


def test_binomial_with_new_features():
    """Test binomial classification with adaptive weighting and group constraints."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 100, 12

    # Create some collinear features
    x_base = np.random.randn(n)
    X = np.zeros((n, p))
    for i in range(4):
        X[:, i] = x_base + 0.1 * np.random.randn(n)
    for i in range(4, p):
        X[:, i] = np.random.randn(n)

    # Create binary outcome
    logits = x_base
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n) < probs).astype(float)

    # Test with new features
    result = fit_uni(X, y, family="binomial",
                    adaptive_weighting=True,
                    enable_group_constraint=True,
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'groups')


def test_poisson_regression():
    """Test Poisson regression for count data."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 15
    X = np.random.randn(n, p)

    # Create count outcome
    log_lambda = 0.5 + X[:, 0] - 0.3 * X[:, 1]
    lambda_true = np.exp(log_lambda)
    y = np.random.poisson(lambda_true, size=n)

    # Test fit_uni with poisson
    result = fit_uni(X, y, family="poisson",
                    adaptive_weighting=False,
                    enable_group_constraint=False,
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'coefs')
    assert result.family == "poisson"

    # Test cv_uni with poisson
    cv_result = cv_uni(X, y, family="poisson",
                      adaptive_weighting=False,
                      enable_group_constraint=False,
                      n_folds=3,
                      n_lmdas=10,
                      seed=42)
    assert cv_result is not None
    assert cv_result.family == "poisson"


def test_poisson_with_new_features():
    """Test Poisson regression with adaptive weighting."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 80, 12
    X = np.random.randn(n, p)

    # Create count outcome
    log_lambda = 0.3 + X[:, 0]
    lambda_true = np.exp(log_lambda)
    y = np.random.poisson(lambda_true, size=n)

    # Test with adaptive weighting
    result = fit_uni(X, y, family="poisson",
                    adaptive_weighting=True,
                    enable_group_constraint=False,
                    weight_method="correlation",
                    n_lmdas=10)
    assert result is not None


def test_multinomial_classification():
    """Test multinomial classification (as one-vs-rest)."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    n, p = 100, 15
    X = np.random.randn(n, p)

    # Create 3-class outcome
    y = np.zeros(n, dtype=int)
    y[:33] = 0
    y[33:66] = 1
    y[66:] = 2
    # Make features somewhat informative
    X[:33, 0] += 1.0
    X[33:66, 1] += 1.0
    X[66:, 2] += 1.0

    # Test fit_uni with multinomial (treated as one-vs-rest)
    result = fit_uni(X, y, family="multinomial",
                    adaptive_weighting=False,
                    enable_group_constraint=False,
                    n_lmdas=10)
    assert result is not None
    assert hasattr(result, 'coefs')
    assert result.family == "multinomial"


def test_backend_selection_glm():
    """Test that both numba and pytorch backends work with different GLM families."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 60, 10
    X = np.random.randn(n, p)

    # Test Gaussian
    y_gauss = X[:, 0] + 0.5 * np.random.randn(n)
    for backend in ["numba", "pytorch"]:
        result = fit_uni(X, y_gauss, family="gaussian", backend=backend,
                        adaptive_weighting=True, n_lmdas=5)
        assert result is not None

    # Test Binomial
    y_binom = (np.random.rand(n) < 0.5).astype(float)
    for backend in ["numba", "pytorch"]:
        result = fit_uni(X, y_binom, family="binomial", backend=backend,
                        adaptive_weighting=False, n_lmdas=5)
        assert result is not None


def test_config_valid_families():
    """Test that config has all valid families."""
    from unilasso.config import VALID_FAMILIES

    expected_families = {"gaussian", "binomial", "multinomial", "poisson", "cox"}
    assert VALID_FAMILIES == expected_families
    assert len(VALID_FAMILIES) == 5


if __name__ == "__main__":
    print("Running full GLM family tests...\n")

    test_config_valid_families()
    print("✓ test_config_valid_families passed")

    test_gaussian_regression_basic()
    print("✓ test_gaussian_regression_basic passed")

    test_gaussian_with_new_features()
    print("✓ test_gaussian_with_new_features passed")

    test_binomial_classification()
    print("✓ test_binomial_classification passed")

    test_binomial_with_new_features()
    print("✓ test_binomial_with_new_features passed")

    test_poisson_regression()
    print("✓ test_poisson_regression passed")

    test_poisson_with_new_features()
    print("✓ test_poisson_with_new_features passed")

    test_multinomial_classification()
    print("✓ test_multinomial_classification passed")

    test_backend_selection_glm()
    print("✓ test_backend_selection_glm passed")

    print("\nAll full GLM family tests passed!")
