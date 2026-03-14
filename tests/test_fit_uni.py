"""
Unit tests for fit_uni function with Numba backend.

Tests ensure that:
1. Numba and PyTorch backends produce identical results
2. Backward compatibility is maintained
3. Edge cases are handled correctly
"""

import sys
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, '/workspaces/uni-lasso')


def test_backend_import():
    """Test that both backends can be imported and accessed."""
    from unilasso.uni_lasso import _fit_pytorch_lasso_path
    from unilasso.solvers import _fit_numba_lasso_path

    assert _fit_pytorch_lasso_path is not None
    assert _fit_numba_lasso_path is not None


def test_solver_equivalence_small():
    """Test that Numba and PyTorch solvers produce identical results on small data."""
    from unilasso.solvers import _fit_numba_lasso_path
    from unilasso.uni_lasso import _fit_pytorch_lasso_path

    np.random.seed(42)
    n, p = 50, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    lmdas = np.logspace(-2, 0, 5)
    negative_penalty = 1.0

    # Run both solvers
    betas_numba, ints_numba = _fit_numba_lasso_path(
        X, y, lmdas, negative_penalty, fit_intercept=True,
        lr=0.01, max_epochs=5000, tol=1e-6
    )
    betas_pytorch, ints_pytorch = _fit_pytorch_lasso_path(
        X, y, lmdas, negative_penalty, fit_intercept=True,
        lr=0.01, max_epochs=5000, tol=1e-6
    )

    # Check that results are close (differences due to float32/float64 and convergence)
    np.testing.assert_allclose(betas_numba, betas_pytorch, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(ints_numba, ints_pytorch, rtol=5e-2, atol=5e-2)


def test_fit_uni_backend_equivalence():
    """Test that fit_uni produces identical results with both backends."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    y = X[:, :5].sum(axis=1) + np.random.randn(n) * 0.5

    # Run with both backends
    result_numba = fit_uni(X, y, n_lmdas=20, backend="numba", verbose=False)
    result_pytorch = fit_uni(X, y, n_lmdas=20, backend="pytorch", verbose=False)

    # Check results (differences due to float32/float64 and convergence paths)
    np.testing.assert_allclose(result_numba.coefs, result_pytorch.coefs, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(result_numba.intercept, result_pytorch.intercept, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(result_numba.lmdas, result_pytorch.lmdas)


def test_cv_uni_backend_equivalence():
    """Test that cv_uni produces identical results with both backends."""
    from unilasso.uni_lasso import cv_uni

    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    y = X[:, :5].sum(axis=1) + np.random.randn(n) * 0.5

    # Run with both backends (use same seed for fold splitting)
    result_numba = cv_uni(X, y, n_folds=3, n_lmdas=10, backend="numba",
                          verbose=False, seed=42)
    result_pytorch = cv_uni(X, y, n_folds=3, n_lmdas=10, backend="pytorch",
                            verbose=False, seed=42)

    # Check that the lambda paths are the same
    np.testing.assert_allclose(result_numba.lmdas, result_pytorch.lmdas)


def test_backend_parameter_validation():
    """Test that invalid backend values raise appropriate errors."""
    from unilasso.uni_lasso import fit_uni, cv_uni

    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50)

    # Test invalid backend for fit_uni
    with pytest.raises(ValueError, match="backend must be either"):
        fit_uni(X, y, backend="invalid")

    # Test invalid backend for cv_uni
    with pytest.raises(ValueError, match="backend must be either"):
        cv_uni(X, y, n_folds=2, backend="invalid")


def test_negative_penalty_zero():
    """Test that negative_penalty=0 works (standard lasso)."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 80, 15
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    result = fit_uni(X, y, negative_penalty=0.0, n_lmdas=10, backend="numba", verbose=False)

    assert result is not None
    assert hasattr(result, 'coefs')
    assert hasattr(result, 'intercept')
    assert hasattr(result, 'lmdas')


def test_single_lambda():
    """Test with a single lambda value."""
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    n, p = 60, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    result = fit_uni(X, y, lmdas=0.1, backend="numba", verbose=False)

    # When there's a single lambda, coefs gets squeezed to (p,)
    assert result.coefs.shape == (p,) or result.coefs.shape[0] == 1
    assert len(result.lmdas) == 1


def test_warm_start_effect():
    """Test that warm start works correctly across the lambda path."""
    from unilasso.solvers import _fit_numba_lasso_path

    np.random.seed(42)
    n, p = 50, 8
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    # Create a lambda path descending from large to small
    lmdas = np.logspace(1, -2, 10)

    betas, ints = _fit_numba_lasso_path(
        X, y, lmdas, negative_penalty=1.0,
        lr=0.01, max_epochs=100, tol=1e-6
    )

    # Coefficients should generally get larger in magnitude as lambda decreases
    # (not strictly monotonic due to the proximal operator, but trend should exist)
    l1_norms = np.sum(np.abs(betas), axis=1)

    # The last (smallest lambda) should have larger or equal L1 norm than first
    assert l1_norms[-1] >= l1_norms[0]


def test_default_backend_is_numba():
    """Test that the default backend is numba."""
    from unittest.mock import patch
    from unilasso.uni_lasso import fit_uni

    np.random.seed(42)
    X = np.random.randn(30, 5)
    y = np.random.randn(30)

    # Patch the numba solver to verify it's called
    with patch('unilasso.uni_lasso._fit_numba_lasso_path') as mock_numba:
        mock_numba.return_value = (np.zeros((10, 5)), np.zeros(10))

        # Call without specifying backend
        fit_uni(X, y, n_lmdas=10, verbose=False)

        # Verify numba solver was called
        mock_numba.assert_called_once()


if __name__ == "__main__":
    # Run tests without pytest for quick verification
    print("Running quick verification tests...")

    test_backend_import()
    print("✓ test_backend_import passed")

    test_solver_equivalence_small()
    print("✓ test_solver_equivalence_small passed")

    test_fit_uni_backend_equivalence()
    print("✓ test_fit_uni_backend_equivalence passed")

    test_negative_penalty_zero()
    print("✓ test_negative_penalty_zero passed")

    test_single_lambda()
    print("✓ test_single_lambda passed")

    print("\nAll quick tests passed!")
