"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""



import numpy as np
import pandas as pd
from numba import jit
import adelie as ad


from typing import TypedDict, List, Optional, Tuple, Union, Callable
import logging

from .univariate_regression import fit_loo_univariate_models
from .config import VALID_FAMILIES
from .utils import warn_zero_variance



logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

class UniLassoResult(TypedDict):
    gamma: np.ndarray
    gamma_intercept: np.ndarray
    beta: np.ndarray
    beta_intercepts: np.ndarray
    lasso_model: ad.grpnet
    regularizers: List[float]
    cv_plot: Optional[Callable]
    best_idx: Optional[int]


@jit(nopython=True, cache=True)
def _fit_univariate_regression_gaussian_numba(
            X: np.ndarray, 
            y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit univariate Gaussian regression for each feature in X."""
    n, p = X.shape
    beta_intercepts = np.zeros(p)
    beta_coefs = np.zeros(p)

    for j in range(p):
        xj = np.expand_dims(X[:, j], axis=1)
        xj_mean = np.mean(xj)
        y_mean = np.mean(y)
        sxy = np.sum(xj[:, 0] * y) - n * xj_mean * y_mean
        sxx = np.sum(xj[:, 0] ** 2) - n * xj_mean ** 2
        slope = sxy / sxx
        beta_intercepts[j] = y_mean - slope * xj_mean
        beta_coefs[j] = slope

    return beta_intercepts, beta_coefs


def fit_univariate_regression(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit univariate regression model for each feature in X.

    Args:
        X: Feature matrix of shape (n, p).
        y: Target vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').

    Returns:
        Tuple containing:
            - beta_intercepts: Intercepts of the regression model.
            - beta_coefs: Coefficients of the regression model.
    """
    n, p = X.shape

    if family == "gaussian":
        beta_intercepts, beta_coefs = _fit_univariate_regression_gaussian_numba(X, y)
    elif family in {"binomial", "cox"}:
        if family == "binomial":
            glm_y = ad.glm.binomial(y)
        elif family == "cox":
            glm_y = ad.glm.cox(start=np.zeros(n), stop=y[:, 0], status=y[:, 1])

        beta_intercepts = np.zeros(p)
        beta_coefs = np.zeros(p)

        for j in range(p):
            X_j = np.column_stack([np.ones(n), X[:, j]])
            X_j = np.asfortranarray(X_j)
            glm_fit = ad.grpnet(X_j, 
                                glm_y, 
                                intercept=False, 
                                lmda_path=[0.0])
            coefs = glm_fit.betas.A
            beta_intercepts[j] = coefs[0][0]
            beta_coefs[j] = coefs[0][1]
    else:
        raise ValueError(f"Unsupported family type: {family}")

    return beta_intercepts, beta_coefs


def fit_univariate_models(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit univariate least squares regression for each feature in X and compute
    leave-one-out (LOO) fitted values.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').

    Returns:
        Tuple containing:
            - loo_fits: Leave-one-out fitted values.
            - beta_intercepts: Intercepts from univariate regressions.
            - beta_coefs: Slopes from univariate regressions.
    """
    beta_intercepts, beta_coefs = fit_univariate_regression(X, y, family)
    loo_fits = fit_loo_univariate_models(X, y, family=family)["fit"]
    return loo_fits, beta_intercepts, beta_coefs


def _format_unilasso_input(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str, 
            regularizers: Optional[Union[float, List[float]]]
) -> Tuple[np.ndarray, np.ndarray, str, Optional[List[float]], Optional[np.ndarray]]:
    """Format and validate input for UniLasso."""
    if family not in VALID_FAMILIES:
        raise ValueError(f"Family must be one of {VALID_FAMILIES}")
    
    X = np.array(X, dtype=float)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("X must be a 1D or 2D NumPy array.")

    zero_var_idx = np.where(np.var(X, axis=0) == 0)[0]
    if len(zero_var_idx) > 0:
        warn_zero_variance(len(zero_var_idx), X.shape[1])
        X = np.delete(X, zero_var_idx, axis=1)
        if X.shape[1] == 0:
            raise ValueError("All features have zero variance.")

    y = _format_y(y, family)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows (samples).")

    regularizers = _format_regularizers(regularizers)

    return X, y, family, regularizers, zero_var_idx


def _format_y(
        y: Union[np.ndarray, pd.DataFrame], 
        family: str) -> np.ndarray:
    """Format and validate y based on the family."""
    if family in {"gaussian", "binomial"}:
        y = np.array(y, dtype=float).flatten()
        if family == "binomial" and not np.all(np.isin(y, [0, 1])):
            raise ValueError("For `binomial` family, y must be binary with values 0 and 1.")
    elif family == "cox":
        if isinstance(y, (pd.DataFrame, dict)):
            if not 'time' in y.columns or not 'status' in y.columns:
                raise ValueError("For `cox` family, y must be a DataFrame with columns 'time' and 'status'.")
            y = np.column_stack((y["time"], y["status"]))
        if y.shape[1] != 2:
            raise ValueError("For `cox` family, y must have two columns corresponding to time and status.")
        if not np.all(y[:, 0] >= 0):
            raise ValueError("For `cox` family, time values must be nonnegative.")
        if not np.all(np.isin(y[:, 1], [0, 1])):
            raise ValueError("For `cox` family, status values must be binary with values 0 and 1.")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")
    
    return y


def _format_regularizers(regularizers: Optional[Union[float, List[float]]]) -> Optional[List[float]]:
    """Format and validate regularizers."""
    if regularizers is None:
        return None
    if isinstance(regularizers, (float, int)):
        regularizers = [float(regularizers)]
    if not isinstance(regularizers, list) or any(r < 0 for r in regularizers):
        raise ValueError("regularizers must be a nonnegative float or list of floats.")
    return regularizers


def _prepare_unilasso_input(
                X: np.ndarray, 
                y: np.ndarray, 
                family: str, 
                regularizers: Optional[Union[float, List[float]]]
) -> Tuple[np.ndarray, 
           np.ndarray, 
           np.ndarray, 
           ad.glm.GlmBase64, 
           List[ad.constraint.ConstraintBase64], 
           Optional[List[float]], Optional[np.ndarray]]:
    """Prepare input for UniLasso."""
    X, y, family, regularizers, zero_var_idx = _format_unilasso_input(X, y, family, regularizers)

    loo_fits, beta_intercepts, beta_coefs_fit = fit_univariate_models(X, y, family=family)
    loo_fits = np.asfortranarray(loo_fits)

    glm_family = _get_glm_family(family, y)
    constraints = [ad.constraint.lower(b=np.zeros(1)) for _ in range(X.shape[1])]

    return loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, regularizers, zero_var_idx




def _get_glm_family(family: str, 
                    y: np.ndarray) -> ad.glm.GlmBase64:
    """Get the appropriate GLM family."""
    if family == "gaussian":
        return ad.glm.gaussian(y)
    elif family == "binomial":
        return ad.glm.binomial(y)
    elif family == "cox":
        return ad.glm.cox(start=np.zeros(len(y)), stop=y[:, 0], status=y[:, 1])
    else:
        raise ValueError(f"Unsupported family: {family}")



def _handle_zero_variance(
            gamma_hat_fit: np.ndarray,
            beta_coefs_fit: np.ndarray,
            zero_var_idx: Optional[np.ndarray],
            total_num_var: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle zero variance features."""
    if zero_var_idx is not None and len(zero_var_idx) > 0:
        num_regs = gamma_hat_fit.shape[0]
        gamma_hat = np.zeros((num_regs, total_num_var))
        beta_coefs = np.zeros((num_regs, total_num_var))
        pos_var_idx = np.setdiff1d(np.arange(total_num_var), zero_var_idx)
        gamma_hat[:, pos_var_idx] = gamma_hat_fit
        beta_coefs[:, pos_var_idx] = beta_coefs_fit
    else:
        gamma_hat = gamma_hat_fit
        beta_coefs = beta_coefs_fit
    return gamma_hat, beta_coefs



def _print_unilasso_results(
            theta_hat: np.ndarray, 
            regularizers: List[float], 
            best_idx: Optional[int] = None
) -> None:
    """Log UniLasso results."""
    num_selected = np.sum(theta_hat != 0, axis=1)
    logger.info("\n--- UniLasso Results ---")
    logger.info(f"Number of Selected Features: {num_selected}")
    logger.info(f"Regularization path (rounded to 3 decimal places): {np.round(regularizers, 3)}")
    if best_idx is not None:
        logger.info(f"Best Regularization Parameter: {regularizers[best_idx]}")




# ------------------------------------------------------------------------------
# Perform cross-validation UniLasso
# ------------------------------------------------------------------------------

def cv_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            verbose: bool = False,
            seed: Optional[int] = None
) ->  UniLassoResult:
    """
    Perform cross-validation UniLasso.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        verbose: Whether to print results.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing UniLasso results.
    """
    loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X, y, family, None)

    cv_lasso = ad.cv_grpnet(
        X=loo_fits,
        glm=glm_family,
        seed=seed,
        groups=None,
        intercept=True,
        constraints=constraints,
    )

    lasso_model = cv_lasso.fit(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=True,
        constraints=constraints,
    )

    theta_hat = lasso_model.betas.toarray()
    theta_0 = lasso_model.intercepts

    gamma_hat_fit = theta_hat * beta_coefs_fit
    gamma_0 = theta_0 + np.sum(theta_hat * beta_intercepts, axis=1)

    gamma_hat, beta_coefs = _handle_zero_variance(gamma_hat_fit, beta_coefs_fit, zero_var_idx, X.shape[1])

    if verbose:
        _print_unilasso_results(theta_hat, cv_lasso.lmdas, int(cv_lasso.best_idx))

    return {
        "gamma": gamma_hat,
        "gamma_intercept": gamma_0,
        "beta": beta_coefs,
        "beta_intercepts": beta_intercepts,
        "lasso_model": lasso_model,
        "cv_plot": cv_lasso.plot_loss,
        "best_idx": int(cv_lasso.best_idx),
        "regularizers": cv_lasso.lmdas
    }




# ------------------------------------------------------------------------------
# Fit UniLasso for a specified regularization path
# ------------------------------------------------------------------------------
def fit_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            regularizers: Optional[Union[float, List[float]]] = None,
            verbose: bool = False
) -> UniLassoResult:
    """
    Perform UniLasso with specified regularization parameters.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        regularizers: Lasso regularization parameter(s).
        verbose: Whether to print results.

    Returns:
        Dictionary containing UniLasso results.
    """
    loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, regularizers, zero_var_idx = _prepare_unilasso_input(X, y, family, regularizers)
    assert regularizers is not None

    lasso_model = ad.grpnet(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=True,
        lmda_path=regularizers,
        constraints=constraints,
    )

    sorted_indices = np.argsort(-np.array(regularizers))
    reverse_indices = np.argsort(sorted_indices)

    theta_hat = lasso_model.betas.toarray()[reverse_indices, :]
    theta_0 = lasso_model.intercepts[reverse_indices]

    gamma_hat_fit = theta_hat * beta_coefs_fit
    gamma_0 = theta_0 + np.sum(theta_hat * beta_intercepts, axis=1)

    gamma_hat, beta_coefs = _handle_zero_variance(gamma_hat_fit, beta_coefs_fit, zero_var_idx, X.shape[1])

    if verbose:
        _print_unilasso_results(theta_hat, regularizers)

    return {
        "gamma": gamma_hat.squeeze(),
        "gamma_intercept": gamma_0.squeeze(),
        "beta": beta_coefs.squeeze(),
        "beta_intercepts": beta_intercepts.squeeze(),
        "lasso_model": lasso_model,
        "regularizers": regularizers
    }