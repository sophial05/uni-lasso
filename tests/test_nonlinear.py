"""
Test nonlinear univariate models: spline and tree regression.
"""
import numpy as np
import pytest
from unilasso.uni_lasso import fit_uni, cv_uni
from unilasso.univariate_regression import (
    leave_one_out_spline,
    leave_one_out_tree,
    fit_loo_univariate_models
)


def simulate_nonlinear_data(n=100, p=10, nonlinear_type="sine"):
    """Simulate data with nonlinear relationships."""
    np.random.seed(42)
    X = np.random.randn(n, p)

    if nonlinear_type == "sine":
        # Sine wave relationship
        y = np.sin(X[:, 0] * 2) + 0.5 * np.sin(X[:, 1] * 3) + np.random.randn(n) * 0.2
    elif nonlinear_type == "quadratic":
        # Quadratic relationship
        y = X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2 + np.random.randn(n) * 0.2
    elif nonlinear_type == "step":
        # Step function relationship
        y = np.where(X[:, 0] > 0, 1.0, -1.0) + np.where(X[:, 1] > 0, 0.5, -0.5) + np.random.randn(n) * 0.1
    else:
        # Linear for comparison
        y = X[:, 0] + X[:, 1] + np.random.randn(n) * 0.2

    return X, y


class TestSplineRegression:
    """Tests for spline univariate regression."""

    def test_leave_one_out_spline_basic(self):
        """Test basic spline LOO functionality."""
        np.random.seed(42)
        n = 50
        p = 3
        X = np.random.randn(n, p)
        y = np.sin(X[:, 0]) + np.random.randn(n) * 0.1

        result = leave_one_out_spline(X, y, spline_df=5, degree=3)

        assert "fit" in result
        assert "beta" in result
        assert "beta0" in result
        assert result["fit"].shape == (n, p)
        assert result["beta"].shape == (p,)
        assert result["beta0"].shape == (p,)

    def test_leave_one_out_spline_different_df(self):
        """Test spline with different degrees of freedom."""
        np.random.seed(42)
        n = 30
        p = 2
        X = np.random.randn(n, p)
        y = X[:, 0] ** 2 + np.random.randn(n) * 0.1

        # Test with different df values
        for df in [3, 5, 7]:
            result = leave_one_out_spline(X, y, spline_df=df, degree=3)
            assert result["fit"].shape == (n, p)

    def test_fit_loo_univariate_models_spline(self):
        """Test spline through the dispatch function."""
        np.random.seed(42)
        n = 50
        p = 5
        X = np.random.randn(n, p)
        y = np.sin(X[:, 0] * 2) + np.random.randn(n) * 0.1

        result = fit_loo_univariate_models(
            X, y,
            family="gaussian",
            univariate_model="spline",
            spline_df=5,
            spline_degree=3
        )

        assert "fit" in result
        assert "beta" in result
        assert "beta0" in result


class TestTreeRegression:
    """Tests for decision tree univariate regression."""

    def test_leave_one_out_tree_basic(self):
        """Test basic tree LOO functionality."""
        np.random.seed(42)
        n = 50
        p = 3
        X = np.random.randn(n, p)
        y = np.where(X[:, 0] > 0, 1.0, -1.0) + np.random.randn(n) * 0.1

        result = leave_one_out_tree(X, y, tree_max_depth=2)

        assert "fit" in result
        assert "beta" in result
        assert "beta0" in result
        assert result["fit"].shape == (n, p)
        assert result["beta"].shape == (p,)
        assert result["beta0"].shape == (p,)

    def test_leave_one_out_tree_different_depth(self):
        """Test tree with different depths."""
        np.random.seed(42)
        n = 30
        p = 2
        X = np.random.randn(n, p)
        y = np.where(X[:, 0] > 0, 1.0, -1.0) + np.random.randn(n) * 0.1

        # Test with different depths
        for depth in [1, 2, 3]:
            result = leave_one_out_tree(X, y, tree_max_depth=depth)
            assert result["fit"].shape == (n, p)

    def test_fit_loo_univariate_models_tree(self):
        """Test tree through the dispatch function."""
        np.random.seed(42)
        n = 50
        p = 5
        X = np.random.randn(n, p)
        y = np.where(X[:, 0] > 0, 1.0, -1.0) + np.random.randn(n) * 0.1

        result = fit_loo_univariate_models(
            X, y,
            family="gaussian",
            univariate_model="tree",
            tree_max_depth=2
        )

        assert "fit" in result
        assert "beta" in result
        assert "beta0" in result


class TestFitUniNonlinear:
    """Tests for fit_uni with nonlinear models."""

    def test_fit_uni_spline(self):
        """Test fit_uni with spline model."""
        X, y = simulate_nonlinear_data(n=80, p=8, nonlinear_type="sine")

        result = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=20,
            univariate_model="spline",
            spline_df=5,
            spline_degree=3
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "intercept")
        assert hasattr(result, "lmdas")
        assert result.coefs.shape[0] == 20
        assert result.coefs.shape[1] == 8

    def test_fit_uni_tree(self):
        """Test fit_uni with tree model."""
        X, y = simulate_nonlinear_data(n=80, p=8, nonlinear_type="step")

        result = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=20,
            univariate_model="tree",
            tree_max_depth=2
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "intercept")
        assert hasattr(result, "lmdas")
        assert result.coefs.shape[0] == 20
        assert result.coefs.shape[1] == 8

    def test_fit_uni_spline_with_adaptive(self):
        """Test fit_uni with spline and adaptive weighting."""
        X, y = simulate_nonlinear_data(n=80, p=8, nonlinear_type="quadratic")

        result = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=20,
            univariate_model="spline",
            spline_df=5,
            adaptive_weighting=True,
            weight_method="correlation"
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "intercept")

    def test_fit_uni_tree_with_group_constraint(self):
        """Test fit_uni with tree and group constraint."""
        X, y = simulate_nonlinear_data(n=80, p=8, nonlinear_type="step")

        result = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=20,
            univariate_model="tree",
            tree_max_depth=2,
            enable_group_constraint=True,
            corr_threshold=0.5
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "groups")
        assert hasattr(result, "group_signs")

    def test_fit_uni_invalid_model(self):
        """Test that invalid univariate_model raises error."""
        X, y = simulate_nonlinear_data(n=50, p=5)

        with pytest.raises(ValueError, match="univariate_model must be one of"):
            fit_uni(
                X, y,
                family="gaussian",
                univariate_model="invalid"
            )


class TestCVUniNonlinear:
    """Tests for cv_uni with nonlinear models."""

    def test_cv_uni_spline(self):
        """Test cv_uni with spline model."""
        X, y = simulate_nonlinear_data(n=100, p=8, nonlinear_type="sine")

        result = cv_uni(
            X, y,
            family="gaussian",
            n_folds=3,
            n_lmdas=15,
            univariate_model="spline",
            spline_df=5
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "intercept")
        assert hasattr(result, "avg_losses")
        assert hasattr(result, "best_lmda")

    def test_cv_uni_tree(self):
        """Test cv_uni with tree model."""
        X, y = simulate_nonlinear_data(n=100, p=8, nonlinear_type="step")

        result = cv_uni(
            X, y,
            family="gaussian",
            n_folds=3,
            n_lmdas=15,
            univariate_model="tree",
            tree_max_depth=2
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "intercept")
        assert hasattr(result, "avg_losses")
        assert hasattr(result, "best_lmda")

    def test_cv_uni_backward_compatibility(self):
        """Test that cv_uni still works with linear model (default)."""
        X, y = simulate_nonlinear_data(n=100, p=8, nonlinear_type="linear")

        # Default should be linear
        result = cv_uni(
            X, y,
            family="gaussian",
            n_folds=3,
            n_lmdas=10
        )

        assert hasattr(result, "coefs")
        assert hasattr(result, "best_lmda")


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_fit_uni_default_linear(self):
        """Test that fit_uni defaults to linear model."""
        X, y = simulate_nonlinear_data(n=50, p=5, nonlinear_type="linear")

        # Without specifying univariate_model
        result = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=10
        )

        assert hasattr(result, "coefs")
        assert result.coefs.shape[0] == 10

    def test_linear_vs_spline_comparison(self):
        """Compare linear and spline on nonlinear data."""
        X, y = simulate_nonlinear_data(n=100, p=5, nonlinear_type="sine")

        # Linear model
        result_linear = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=20,
            univariate_model="linear"
        )

        # Spline model
        result_spline = fit_uni(
            X, y,
            family="gaussian",
            n_lmdas=20,
            univariate_model="spline",
            spline_df=5
        )

        # Both should return valid results
        assert hasattr(result_linear, "coefs")
        assert hasattr(result_spline, "coefs")

    def test_fit_loo_univariate_models_default_linear(self):
        """Test that fit_loo_univariate_models defaults to linear."""
        np.random.seed(42)
        n = 30
        p = 3
        X = np.random.randn(n, p)
        y = X[:, 0] + np.random.randn(n) * 0.1

        # Default should be linear
        result_default = fit_loo_univariate_models(X, y, family="gaussian")

        # Explicit linear
        result_linear = fit_loo_univariate_models(
            X, y,
            family="gaussian",
            univariate_model="linear"
        )

        # Results should be similar
        np.testing.assert_allclose(result_default["fit"], result_linear["fit"], rtol=1e-5)


if __name__ == "__main__":
    # Run tests manually
    test_spline = TestSplineRegression()
    test_spline.test_leave_one_out_spline_basic()
    print("✓ test_leave_one_out_spline_basic passed")

    test_tree = TestTreeRegression()
    test_tree.test_leave_one_out_tree_basic()
    print("✓ test_leave_one_out_tree_basic passed")

    test_fit = TestFitUniNonlinear()
    test_fit.test_fit_uni_spline()
    print("✓ test_fit_uni_spline passed")

    test_fit.test_fit_uni_tree()
    print("✓ test_fit_uni_tree passed")

    test_cv = TestCVUniNonlinear()
    test_cv.test_cv_uni_spline()
    print("✓ test_cv_uni_spline passed")

    print("\nAll manual tests passed!")
