"""
Numba-optimized solvers for UniLasso.

This module provides high-performance implementations of the lasso path solver
using Numba JIT compilation.
"""

import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True, cache=True)
def _fit_numba_lasso_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lmdas: np.ndarray,
    negative_penalty: float,
    fit_intercept: bool = True,
    lr: float = 0.01,
    max_epochs: int = 5000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a lasso path using Numba-optimized gradient descent with warm start.

    Uses asymmetric soft-thresholding for negative coefficient penalty.
    This implementation exactly matches the PyTorch version's behavior.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, n_features)
    y_train : np.ndarray
        Target values of shape (n_samples,)
    lmdas : np.ndarray
        Regularization path, sorted in descending order for warm start
    negative_penalty : float
        Additional penalty strength for negative coefficients
    fit_intercept : bool
        Whether to fit an intercept term
    lr : float
        Learning rate for gradient descent
    max_epochs : int
        Maximum number of epochs per lambda
    tol : float
        Convergence tolerance

    Returns
    -------
    betas_matrix : np.ndarray
        Coefficients of shape (n_lmdas, n_features)
    intercepts : np.ndarray
        Intercepts of shape (n_lmdas,)
    """
    n_samples, n_features = X_train.shape
    n_lmdas = len(lmdas)

    # Pre-allocate result storage
    betas_matrix = np.zeros((n_lmdas, n_features))
    intercepts = np.zeros(n_lmdas)

    # Initialize parameters (warm start across lambdas)
    weights = np.zeros(n_features)
    bias = 0.0

    for i, lmda in enumerate(lmdas):
        tau_pos = lr * lmda
        tau_neg = lr * (lmda + negative_penalty)

        for epoch in range(max_epochs):
            # Step 1: Gradient descent on MSE only
            y_pred = np.dot(X_train, weights) + bias
            error = y_pred - y_train

            # Compute gradients (mean over samples)
            # Note: derivative of mean((y_pred - y)^2) is 2 * mean(error * X)
            grad_weights = 2.0 * np.dot(X_train.T, error) / n_samples
            grad_bias = 2.0 * np.sum(error) / n_samples

            # Take gradient step
            weights_gd = weights - lr * grad_weights
            bias_gd = bias - lr * grad_bias

            # Step 2: Asymmetric proximal operator
            w_prox = np.zeros_like(weights_gd)

            # Positive direction shrinkage: w > tau_pos -> w - tau_pos
            mask_pos = weights_gd > tau_pos
            w_prox[mask_pos] = weights_gd[mask_pos] - tau_pos

            # Negative direction shrinkage: w < -tau_neg -> w + tau_neg
            mask_neg = weights_gd < -tau_neg
            w_prox[mask_neg] = weights_gd[mask_neg] + tau_neg

            # Check convergence based on weight change
            max_change = np.max(np.abs(w_prox - weights))
            converged = epoch > 0 and max_change < tol

            # Update weights for next iteration
            weights = w_prox
            bias = bias_gd

            if converged:
                break

        # Store results for this lambda
        betas_matrix[i, :] = weights
        intercepts[i] = bias

    return betas_matrix, intercepts


@jit(nopython=True, cache=True)
def _fit_numba_lasso_path_accelerated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lmdas: np.ndarray,
    negative_penalty: float,
    fit_intercept: bool = True,
    lr: float = 0.01,
    max_epochs: int = 5000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accelerated version with precomputed XtX and Xty for faster gradient computation.

    This uses direct matrix computations instead of per-sample operations.
    """
    n_samples, n_features = X_train.shape
    n_lmdas = len(lmdas)

    # Pre-allocate result storage
    betas_matrix = np.zeros((n_lmdas, n_features))
    intercepts = np.zeros(n_lmdas)

    # Precompute matrices for faster gradient computation
    XtX = np.dot(X_train.T, X_train) / n_samples
    Xty = np.dot(X_train.T, y_train) / n_samples
    y_mean = np.mean(y_train)
    X_mean = np.mean(X_train, axis=0)

    # Initialize parameters
    weights = np.zeros(n_features)
    bias = 0.0

    for i, lmda in enumerate(lmdas):
        tau_pos = lr * lmda
        tau_neg = lr * (lmda + negative_penalty)

        for epoch in range(max_epochs):
            # Fast gradient computation using precomputed matrices
            # grad = XtX @ w + bias * X_mean - Xty
            grad_weights = np.dot(XtX, weights) + bias * X_mean - Xty
            grad_bias = np.dot(X_mean, weights) + bias - y_mean

            # Gradient descent step
            weights_new = weights - lr * grad_weights
            bias_new = bias - lr * grad_bias

            # Asymmetric proximal operator
            w_prox = np.zeros_like(weights_new)

            # Positive direction shrinkage
            mask_pos = weights_new > tau_pos
            w_prox[mask_pos] = weights_new[mask_pos] - tau_pos

            # Negative direction shrinkage
            mask_neg = weights_new < -tau_neg
            w_prox[mask_neg] = weights_new[mask_neg] + tau_neg

            # Check convergence
            max_change = np.max(np.abs(w_prox - weights))
            if epoch > 0 and max_change < tol:
                break

            weights = w_prox
            bias = bias_new

        betas_matrix[i, :] = weights
        intercepts[i] = bias

    return betas_matrix, intercepts
