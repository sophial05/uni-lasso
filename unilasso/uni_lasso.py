"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""



import numpy as np
import pandas as pd
from numba import jit, prange
import matplotlib.pyplot as plt
import adelie as ad


from typing import List, Optional, Tuple, Union, Callable
import logging

from .univariate_regression import fit_loo_univariate_models
from .config import VALID_FAMILIES, VALID_UNIVARIATE_MODELS
from .utils import warn_zero_variance, warn_removed_lmdas
from .solvers import _fit_numba_lasso_path


# Configure logger
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def _compute_feature_significance_weights(
    univariate_results: dict,
    weight_method: str = "t_statistic",
    weight_max_scale: float = 10.0
) -> np.ndarray:
    """
    Compute feature-level significance weights for adaptive penalty.

    Parameters
    ----------
    univariate_results : dict
        Dictionary containing univariate regression results with keys:
        't_stats' (for t-statistic), 'p_values' (for p-value), 'correlations' (for correlation)
    weight_method : str
        Method to compute weights: 't_statistic', 'p_value', or 'correlation'
    weight_max_scale : float
        Maximum scale for weights, normalized to [1, weight_max_scale]

    Returns
    -------
    weights : np.ndarray
        Array of significance weights of shape (n_features,)
    """
    n_features = len(univariate_results['beta'])

    if weight_method == "t_statistic":
        stats = np.abs(univariate_results.get('t_stats', np.ones(n_features)))
    elif weight_method == "p_value":
        p_vals = univariate_results.get('p_values', np.ones(n_features))
        stats = -np.log10(np.clip(p_vals, 1e-10, 1.0))
    elif weight_method == "correlation":
        stats = np.abs(univariate_results.get('correlations', np.ones(n_features)))
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    # Normalize to [1, weight_max_scale]
    min_stat = np.min(stats)
    max_stat = np.max(stats)
    if max_stat == min_stat:
        return np.ones(n_features)

    normalized = (stats - min_stat) / (max_stat - min_stat)
    weights = 1.0 + normalized * (weight_max_scale - 1.0)

    return weights


@jit(nopython=True, parallel=True, cache=True)
def _parallel_corr_matrix(X: np.ndarray, min_var: float = 1e-8) -> np.ndarray:
    """
    Parallelized correlation matrix computation with low variance feature filtering.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    min_var : float
        Minimum variance threshold, features with variance below this will have
        correlation set to 0 to avoid numerical issues

    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix of shape (n_features, n_features)
    """
    n_samples, n_features = X.shape
    corr_matrix = np.zeros((n_features, n_features), dtype=np.float64)

    # Precompute means and standard deviations
    means = np.zeros(n_features, dtype=np.float64)
    stds = np.zeros(n_features, dtype=np.float64)

    for j in prange(n_features):
        x = X[:, j]
        mean_x = np.mean(x)
        var_x = np.var(x)
        means[j] = mean_x
        if var_x < min_var:
            stds[j] = 0.0
        else:
            stds[j] = np.sqrt(var_x)

    # Compute correlation matrix in parallel (upper triangle only)
    for i in prange(n_features):
        if stds[i] == 0.0:
            continue
        x_centered = X[:, i] - means[i]
        for j in range(i, n_features):
            if stds[j] == 0.0:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
                continue
            y_centered = X[:, j] - means[j]
            cov = np.dot(x_centered, y_centered) / n_samples
            corr = cov / (stds[i] * stds[j])
            # Clip to [-1, 1] to avoid numerical errors
            corr = max(min(corr, 1.0), -1.0)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def _greedy_correlation_grouping(
    corr_matrix: np.ndarray,
    corr_threshold: float = 0.7,
    max_group_size: int = 20
) -> List[List[int]]:
    """
    Greedy non-overlapping correlation grouping of features.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Absolute correlation matrix of shape (n_features, n_features)
    corr_threshold : float
        Threshold for grouping correlated features
    max_group_size : int
        Maximum size of each group

    Returns
    -------
    groups : List[List[int]]
        List of feature groups, each group is a list of feature indices
    """
    n_features = corr_matrix.shape[0]
    assigned = np.zeros(n_features, dtype=bool)
    groups = []

    # Compute feature importance (sum of absolute correlations) to prioritize grouping
    feature_importance = np.sum(np.abs(corr_matrix), axis=1)

    while not np.all(assigned):
        # Pick the most important unassigned feature as group leader
        unassigned_idx = np.where(~assigned)[0]
        leader = unassigned_idx[np.argmax(feature_importance[unassigned_idx])]

        # Find all unassigned features correlated with leader
        correlated = np.where(
            (~assigned) & (np.abs(corr_matrix[leader]) >= corr_threshold)
        )[0]

        # Limit group size, keep most correlated first
        if len(correlated) > max_group_size:
            # Sort by correlation with leader
            corr_values = corr_matrix[leader, correlated]
            sorted_idx = np.argsort(-np.abs(corr_values))
            correlated = correlated[sorted_idx[:max_group_size]]

        # Add group
        groups.append(correlated.tolist())
        assigned[correlated] = True

    return groups


def _compute_group_penalty_weights(
    groups: List[List[int]],
    beta_univariate: np.ndarray,
    feature_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute group-level penalty weights and dominant signs.

    Parameters
    ----------
    groups : List[List[int]]
        List of feature groups
    beta_univariate : np.ndarray
        Univariate regression coefficients of shape (n_features,)
    feature_weights : np.ndarray
        Feature-level significance weights of shape (n_features,)

    Returns
    -------
    group_signs : np.ndarray
        Dominant sign for each group (+1 or -1) of shape (n_features,)
    group_weights : np.ndarray
        Group penalty weight for each feature of shape (n_features,)
    """
    n_features = len(beta_univariate)
    group_signs = np.ones(n_features)
    group_weights = np.ones(n_features)

    for group in groups:
        if len(group) == 0:
            continue

        # Compute weighted sign vote (weighted by absolute beta * feature weight)
        betas = beta_univariate[group]
        weights = feature_weights[group]
        votes = np.sign(betas) * np.abs(betas) * weights
        total_vote = np.sum(votes)

        # Dominant sign is the sign of the total vote
        dominant_sign = np.sign(total_vote) if total_vote != 0 else 1.0

        # Group weight is proportional to the confidence of the dominant sign
        confidence = np.abs(total_vote) / (np.sum(np.abs(votes)) + 1e-10)
        group_weight = 1.0 + confidence * 4.0  # Scale from 1 to 5

        # Assign to all features in the group
        group_signs[group] = dominant_sign
        group_weights[group] = group_weight

    return group_signs, group_weights


import numpy as np
from typing import Optional, Callable
import adelie as ad

class UniLassoResultBase:
    """
    Base class for UniLasso results, encapsulating model outputs.
    """

    def __init__(self,
                 coefs: np.ndarray,
                 intercept: np.ndarray,
                 family: str,
                 gamma: np.ndarray,
                 gamma_intercept: np.ndarray,
                 beta: np.ndarray,
                 beta_intercepts: np.ndarray,
                 lasso_model: ad.grpnet,
                 lmdas: np.ndarray):
        """
        Initializes the base UniLasso result object.

        Parameters:
        - coefs (np.ndarray): Coefficients of the univariate-guided lasso.
        - intercept (np.ndarray): Intercept of the univariate-guided lasso.
        - family (str): Family of the response variable ('gaussian', 'binomial', 'cox').
        - gamma (np.ndarray): Hidden gamma coefficients.
        - gamma_intercept (np.ndarray): Hidden gamma intercept.
        - beta (np.ndarray): Hidden beta coefficients.
        - beta_intercepts (np.ndarray): Hidden beta intercepts.
        - lasso_model (ad.grpnet): The fitted Lasso model.
        - lmdas (np.ndarray): Regularization path.
        """
        self.coefs = coefs
        self.intercept = intercept
        self.family = family
        self._gamma = gamma
        self._gamma_intercept = gamma_intercept
        self._beta = beta
        self._beta_intercepts = beta_intercepts
        self.lasso_model = lasso_model
        self.lmdas = lmdas
        # 新增：分组信息（可选）
        self.groups = None
        self.group_signs = None

    def get_gamma(self) -> np.ndarray:
        """Returns the hidden gamma coefficients."""
        return self._gamma

    def get_gamma_intercept(self) -> np.ndarray:
        """Returns the hidden gamma intercept."""
        return self._gamma_intercept

    def get_beta(self) -> np.ndarray:
        """Returns the hidden beta coefficients."""
        return self._beta

    def get_beta_intercepts(self) -> np.ndarray:
        """Returns the hidden beta intercepts."""
        return self._beta_intercepts

    def __repr__(self):
        """Custom string representation of the result object."""
        return (f"{self.__class__.__name__}(coefs={self.coefs.shape}, "
                f"intercept={self.intercept.shape}, "
                f"lasso_model={type(self.lasso_model).__name__}, "
                f"lmdas={self.lmdas.shape})")


class UniLassoResult(UniLassoResultBase):
    """
    Class for storing standard UniLasso results.
    """
    pass


class UniLassoCVResult(UniLassoResultBase):
    """
    Class for storing cross-validation UniLasso results.
    """

    def __init__(self, 
                 coefs: np.ndarray, 
                 intercept: np.ndarray, 
                 family: str,
                 gamma: np.ndarray, 
                 gamma_intercept: np.ndarray, 
                 beta: np.ndarray, 
                 beta_intercepts: np.ndarray, 
                 lasso_model: ad.grpnet, 
                 lmdas: np.ndarray,
                 avg_losses: np.ndarray, 
                 cv_plot: Optional[Callable] = None, 
                 best_idx: Optional[int] = None, 
                 best_lmda: Optional[float] = None):
        """
        Initializes the cross-validation result object.

        Additional Parameters:
        - avg_losses (np.ndarray): Average cross-validation losses.
        - cv_plot (Optional[Callable]): Function to generate cross-validation plot.
        - best_idx (Optional[int]): Index of the best-performing regularization parameter.
        - best_lmda (Optional[float]): Best regularization parameter.
        """
        super().__init__(coefs, intercept, family, gamma, gamma_intercept, beta, beta_intercepts, lasso_model, lmdas)
        self.avg_losses = avg_losses
        self.cv_plot = cv_plot
        self.best_idx = best_idx
        self.best_lmda = best_lmda

    def __repr__(self):
        base_repr = super().__repr__()
        return (f"{base_repr}, best_lmda={self.best_lmda}, "
                f"best_idx={self.best_idx}, avg_losses={self.avg_losses.shape})")
    


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
        family: Family of the response variable ('gaussian', 'binomial', 'multinomial', 'poisson', 'cox').

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
            if family == "binomial":
                X_j = np.column_stack([np.ones(n), X[:, j]])
            else:
                # Cox model requires no intercept term
                X_j = np.column_stack([np.zeros(n), X[:, j]])
            X_j = np.asfortranarray(X_j)
            glm_fit = ad.grpnet(X_j,
                                glm_y,
                                intercept=False,
                                lmda_path=[0.0])
            coefs = glm_fit.betas.toarray()

            if family == "binomial":
                beta_intercepts[j] = coefs[0][0]

            beta_coefs[j] = coefs[0][1]
    elif family in {"poisson", "multinomial"}:
        # For Poisson and Multinomial: use simple estimation
        beta_intercepts = np.zeros(p)
        beta_coefs = np.zeros(p)

        # Use Gaussian as a simple fallback for quick estimation
        # This is used for feature importance estimation
        for j in range(p):
            xj = X[:, j]
            # Simple correlation-based beta for Poisson
            if family == "poisson":
                y_mean = np.mean(y)
                x_mean = np.mean(xj)
                cov = np.mean((xj - x_mean) * (y - y_mean))
                var_x = np.mean((xj - x_mean)**2)
                if var_x > 1e-10:
                    beta_coefs[j] = cov / var_x
                beta_intercepts[j] = np.log(y_mean + 1e-10) - beta_coefs[j] * x_mean
            else:  # multinomial
                # For multinomial, use one-vs-rest approach
                y_bin = (y != np.min(y)).astype(float)
                y_mean = np.mean(y_bin)
                x_mean = np.mean(xj)
                cov = np.mean((xj - x_mean) * (y_bin - y_mean))
                var_x = np.mean((xj - x_mean)**2)
                if var_x > 1e-10:
                    beta_coefs[j] = cov / var_x
                beta_intercepts[j] = y_mean - beta_coefs[j] * x_mean
    else:
        raise ValueError(f"Unsupported family type: {family}")

    return beta_intercepts, beta_coefs


def fit_univariate_models(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            univariate_model: str = "linear",
            spline_df: int = 5,
            spline_degree: int = 3,
            tree_max_depth: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit univariate least squares regression for each feature in X and compute
    leave-one-out (LOO) fitted values.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'multinomial', 'poisson', 'cox').
        univariate_model: Type of univariate model to use: 'linear', 'spline', or 'tree'.
        spline_df: Degrees of freedom for spline regression (if univariate_model="spline").
        spline_degree: Degree of polynomial for spline regression (default: 3 for cubic).
        tree_max_depth: Maximum depth for decision tree regression (if univariate_model="tree").

    Returns:
        Tuple containing:
            - loo_fits: Leave-one-out fitted values.
            - beta_intercepts: Intercepts from univariate regressions.
            - beta_coefs: Slopes from univariate regressions.
    """
    # Ensure y is float for all families
    y_float = np.asarray(y, dtype=float)

    # For nonlinear models, use the model-specific LOO fits directly
    if univariate_model != "linear":
        loo_result = fit_loo_univariate_models(
            X, y_float,
            family=family,
            univariate_model=univariate_model,
            spline_df=spline_df,
            spline_degree=spline_degree,
            tree_max_depth=tree_max_depth
        )
        loo_fits = loo_result["fit"]
        beta_coefs = loo_result["beta"]
        beta_intercepts = loo_result["beta0"]
        return loo_fits, beta_intercepts, beta_coefs

    # Linear model - original behavior
    beta_intercepts, beta_coefs = fit_univariate_regression(X, y_float, family)

    # For Poisson and Multinomial: use Gaussian LOO as approximation
    if family in {"poisson", "multinomial"}:
        loo_fits = fit_loo_univariate_models(X, y_float, family="gaussian")["fit"]
    else:
        loo_fits = fit_loo_univariate_models(X, y_float, family=family)["fit"]

    return loo_fits, beta_intercepts, beta_coefs


def _format_unilasso_feature_matrix(X: np.ndarray,
                                    remove_zero_var: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Format and validate feature matrix for UniLasso."""

    X = np.array(X, dtype=float)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("X must be a 1D or 2D NumPy array.")

    if remove_zero_var:
        zero_var_idx = np.where(np.var(X, axis=0) == 0)[0]
        if len(zero_var_idx) > 0:
            warn_zero_variance(len(zero_var_idx), X.shape[1])
            X = np.delete(X, zero_var_idx, axis=1)
            if X.shape[1] == 0:
                raise ValueError("All features have zero variance.")
    else:
        zero_var_idx = None
    
    return X, zero_var_idx



def _format_unilasso_input(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str, 
            lmdas: Optional[Union[float, List[float], np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Format and validate input for UniLasso."""
    if family not in VALID_FAMILIES:
        raise ValueError(f"Family must be one of {VALID_FAMILIES}")
    
    X, zero_var_idx = _format_unilasso_feature_matrix(X, True)
    y = _format_y(y, family)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows (samples).")

    lmdas = _format_lmdas(lmdas)

    return X, y, family, lmdas, zero_var_idx


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


def _format_lmdas(lmdas: Optional[Union[float, List[float], np.ndarray]]) -> Optional[np.ndarray]:
    """Format and validate lmdas."""
    if lmdas is None:
        return None
    if isinstance(lmdas, (float, int)):
        lmdas = [float(lmdas)]

    if not isinstance(lmdas, list) and not isinstance(lmdas, np.ndarray):
        raise ValueError("lmdas must be a nonnegative float, list of floats, or NumPy array of floats.")
    
    lmdas = np.array(lmdas, dtype=float)

    if np.any(np.isnan(lmdas)) or np.any(np.isinf(lmdas)):
        raise ValueError("Regularizers contain NaN or infinite values.")
    
    if np.any(lmdas < 0):
        raise ValueError("Regularizers must be nonnegative.")

    return lmdas


def _prepare_unilasso_input(
                X: np.ndarray,
                y: np.ndarray,
                family: str,
                lmdas: Optional[Union[float, List[float], np.ndarray]],
                univariate_model: str = "linear",
                spline_df: int = 5,
                spline_degree: int = 3,
                tree_max_depth: int = 2
) -> Tuple[np.ndarray,
           np.ndarray,
           np.ndarray,
           np.ndarray,
           np.ndarray,
           Optional[ad.glm.GlmBase64],
           Optional[List[ad.constraint.ConstraintBase64]],
           Optional[np.ndarray]]:
    """Prepare input for UniLasso."""
    X, y, family, lmdas, zero_var_idx = _format_unilasso_input(X, y, family, lmdas)

    loo_fits, beta_intercepts, beta_coefs_fit = fit_univariate_models(
        X, y,
        family=family,
        univariate_model=univariate_model,
        spline_df=spline_df,
        spline_degree=spline_degree,
        tree_max_depth=tree_max_depth
    )
    loo_fits = np.asfortranarray(loo_fits)

    # Only get glm_family and constraints for families supported by adelie
    # For poisson and multinomial, we use our custom solver and don't need these
    # For nonlinear models (spline/tree), we also skip adelie initialization
    if family in {"gaussian", "binomial", "cox"} and univariate_model == "linear":
        glm_family = _get_glm_family(family, y)
        constraints = [ad.constraint.lower(b=np.zeros(1)) for _ in range(X.shape[1])]
    else:
        glm_family = None
        constraints = None

    return X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx




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
            cur_num_var: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle zero variance features."""
    if zero_var_idx is not None and len(zero_var_idx) > 0:
        total_num_var = cur_num_var + len(zero_var_idx)
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
            gamma_hat: np.ndarray, 
            lmdas: np.ndarray, 
            best_idx: Optional[int] = None
) -> None:
    """Print UniLasso results."""

    if gamma_hat.ndim == 1:
        num_selected = np.sum(gamma_hat != 0)
    else:
        num_selected = np.sum(gamma_hat != 0, axis=1)

    # check if interactive environment
    try:
        get_ipython()

        from IPython.core.display import display, HTML
        display(HTML("\n\n<b> --- UniLasso Results --- </b>"))
    except NameError:
        print("\n\n\033[1m --- UniLasso Results --- \033[0m")

    print(f"Number of Selected Features: {num_selected}")
    print(f"Regularization path (rounded to 3 decimal places): {np.round(lmdas, 3)}")
    if best_idx is not None:
        print(f"Best Regularization Parameter: {lmdas[best_idx]}")



def _format_output(lasso_model: ad.grpnet,
                   beta_coefs_fit: np.ndarray,
                   beta_intercepts: np.ndarray,
                   zero_var_idx: Optional[np.ndarray],
                   X: np.ndarray,
                   fit_intercept: bool,
                   reverse_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format UniLasso output."""
    theta_hat = lasso_model.betas.toarray()
    theta_0 = lasso_model.intercepts

    beta_coefs_fit = beta_coefs_fit.squeeze()
    beta_intercepts = beta_intercepts.squeeze()

    if reverse_indices is not None:
        theta_hat = theta_hat[reverse_indices]
        theta_0 = theta_0[reverse_indices]


    gamma_hat_fit = theta_hat * beta_coefs_fit
    gamma_hat, beta_coefs = _handle_zero_variance(gamma_hat_fit, beta_coefs_fit, zero_var_idx, X.shape[1])
    gamma_hat = gamma_hat.squeeze()
    beta_coefs = beta_coefs.squeeze()

    if fit_intercept:
        gamma_0 = theta_0 + np.sum(theta_hat * beta_intercepts, axis=1)
        gamma_0 = gamma_0.squeeze()
    else:
        gamma_0 = np.zeros(len(theta_0))
   
    return gamma_hat, gamma_0, beta_coefs





def _configure_lmda_min_ratio(n: int,
                              p: int) -> np.ndarray:
    """Configure lambda min ratio for UniLasso."""
    return 0.01 if n < p else 1e-4




def _check_lmda_min_ratio(lmda_min_ratio: float) -> float:
    """Check lambda min ratio for UniLasso."""
    if lmda_min_ratio <= 0:
        raise ValueError("Minimum regularization ratio must be positive.")
    if lmda_min_ratio > 1:
        raise ValueError("Minimum regularization ratio must be less than 1.")
    return lmda_min_ratio
    

def _configure_lmda_path(X: np.ndarray, 
                         y: np.ndarray,
                         family: str,
                         n_lmdas: Optional[int], 
                         lmda_min_ratio: Optional[float]) -> np.ndarray:
    """Configure the regularization path for UniLasso."""

    n, p = X.shape
    if n_lmdas is None:
        n_lmdas = 100
    
    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(n, p)

    assert n_lmdas > 0, "Number of regularization parameters must be positive."
    _check_lmda_min_ratio(lmda_min_ratio)
    
    if family == "cox":
        y = y[:, 0]

    # Define function to standardize columns using n (not n-1)
    def moment_sd(z):
        return np.sqrt(np.sum((z - np.mean(z))**2) / len(z))

    X_standardized = (X - np.mean(X, axis=0)) / np.apply_along_axis(moment_sd, 0, X)
    X_standardized = np.array(X_standardized)  

    # Standardize y (centering only)
    y = y - np.mean(y)

    n = X_standardized.shape[0]
    lambda_max = np.max(np.abs(X_standardized.T @ y)) / n

    lambda_path = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * lmda_min_ratio), n_lmdas))

    return lambda_path





# ------------------------------------------------------------------------------
# Perform cross-validation UniLasso
# ------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt

def plot(unilasso_fit) -> None:
    """
    Plots the Lasso coefficient paths as a function of the regularization parameter (lambda),
    with the number of active (nonzero) coefficients labeled at the top.

    Parameters:
    - unilasso_fit: UniLassoResult object containing fitted coefficients and lambda values.
    """
    
    assert hasattr(unilasso_fit, "coefs") and hasattr(unilasso_fit, "lmdas"), \
        "Input must have 'coefs' and 'lmdas' attributes."

    coefs, lambdas = unilasso_fit.coefs, unilasso_fit.lmdas
    if coefs.ndim == 1 or len(lambdas) == 1:
        print("Only one regularization parameter was used. No path to plot.")
        return
    
    plt.figure(figsize=(8, 6))
    neg_log_lambdas = -np.log(lambdas)  # lambdas are already in descending order

    # Compute the number of nonzero coefficients at each lambda
    n_nonzero = np.sum(coefs != 0, axis=1)

    # Plot coefficient paths
    for i in range(coefs.shape[1]):  
        plt.plot(neg_log_lambdas, coefs[:, i], lw=2)

    # Labels and formatting
    plt.xlabel(r"$-\log(\lambda)$", fontsize=12)
    plt.ylabel("Coefficients", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Add secondary x-axis for the number of active coefficients
    ax1 = plt.gca()  
    ax2 = ax1.twiny()  
    ax2.set_xlim(ax1.get_xlim())  
    # Dynamic tick calculation for better readability
    tick_indices = np.linspace(0, len(neg_log_lambdas) - 1, min(6, len(neg_log_lambdas)), dtype=int)
    ax2.set_xticks(neg_log_lambdas[tick_indices])  
    ax2.set_xticklabels(n_nonzero[tick_indices]) 
    ax2.set_xlabel("Number of Active Coefficients", fontsize=12)

    plt.show()


def plot_cv(cv_result: UniLassoCVResult) -> None:
    """
    Plots the cross-validation
    curve as a function of the regularization parameter (lambda).
    """
    cv_result.cv_plot()



def extract_cv(cv_result: UniLassoCVResult) -> UniLassoResult:
    """
    Extract the best coefficients and intercept from a cross-validated UniLasso result.

    Args:
        - cv_result: UniLassoCVResult object.
    
    Returns:
        - UniLassoResult object with the best coefficients and intercept.
    """

    best_coef = cv_result.coefs[cv_result.best_idx].squeeze()
    best_intercept = cv_result.intercept[cv_result.best_idx].squeeze()

    extracted_fit = UniLassoResult(
        coefs=best_coef,
        intercept=best_intercept,
        family=cv_result.family,
        gamma=best_coef,
        gamma_intercept=best_intercept,
        beta=cv_result._beta,
        beta_intercepts=cv_result._beta_intercepts,
        lasso_model=cv_result.lasso_model,
        lmdas=cv_result.lmdas
    )

    return extracted_fit



def cv_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            n_folds: int = 5,
            lmda_min_ratio: float = None,
            verbose: bool = False,
            seed: Optional[int] = None
) ->  UniLassoCVResult:
    """
    Perform cross-validation UniLasso.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        n_folds: Number of cross-validation folds.
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter.
        verbose: Whether to print results.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing UniLasso results.
    """
    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(X.shape[0], X.shape[1])

    assert n_folds > 1, "Number of folds must be greater than 1."
    _check_lmda_min_ratio(lmda_min_ratio)

    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X, y, family, None)
    fit_intercept = False if family == "cox" else True

    cv_lasso = ad.cv_grpnet(
        X=loo_fits,
        glm=glm_family,
        seed=seed,
        n_folds=n_folds,
        groups=None,
        min_ratio=lmda_min_ratio,
        intercept=fit_intercept,
        constraints=constraints,
        tol=1e-7
    )

    # refit lasso along a regularization path that stops at the best chosen lambda
    lasso_model = cv_lasso.fit(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=fit_intercept,
        constraints=constraints,
    )

    gamma_hat, gamma_0, beta_coefs = _format_output(lasso_model,
                                                    beta_coefs_fit,
                                                    beta_intercepts,
                                                    zero_var_idx,
                                                    X,
                                                    fit_intercept)

    

    cv_plot = cv_lasso.plot_loss
    if verbose:
        _print_unilasso_results(gamma_hat, cv_lasso.lmdas, int(cv_lasso.best_idx))
        cv_plot()
    

    unilasso_result = UniLassoCVResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=cv_lasso.lmdas,
        avg_losses=cv_lasso.avg_losses,
        cv_plot=cv_plot,
        best_idx=int(cv_lasso.best_idx),
        best_lmda=cv_lasso.lmdas[cv_lasso.best_idx]
    )

    return unilasso_result


# ------------------------------------------------------------------------------
# Fit UniLasso for a specified regularization path
# ------------------------------------------------------------------------------
def fit_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            lmdas: Optional[Union[float, List[float], np.ndarray]] = None,
            n_lmdas: Optional[int] = 100,
            lmda_min_ratio: Optional[float] = 1e-2,
            verbose: bool = False
) -> UniLassoResult:
    """
    Perform UniLasso with specified regularization parameters.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        lmdas: Lasso regularization parameter(s).
        n_lmdas: Number of regularization parameters to use if `lmdas` is None.
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter. 
        verbose: Whether to print results.

    Returns:
        Dictionary containing UniLasso results.
    """
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx = _prepare_unilasso_input(X, y, family, lmdas)

    fit_intercept = False if family == "cox" else True


    lasso_model = ad.grpnet(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=fit_intercept,
        lmda_path=lmdas, # Regularization path, if unspecified, will be generated
        constraints=constraints,
        lmda_path_size=n_lmdas,
        min_ratio=lmda_min_ratio,
        tol=1e-7
    )

    glm_lmdas = np.array(lasso_model.lmdas)

    if lmdas is not None:
        if not np.all(np.isin(lmdas, glm_lmdas)):
            removed_lmdas = np.setdiff1d(lmdas, glm_lmdas)
            removed_lmdas = np.round(removed_lmdas, 3)
            warn_removed_lmdas(removed_lmdas)

        matching_idx = np.where(np.isin(lmdas, glm_lmdas))[0]
        lmdas = lmdas[matching_idx]
    else:
        lmdas = glm_lmdas

    if len(lmdas) == 0:
        raise ValueError("No regularization strengths remain after removing invalid values")

    reverse_indices = np.arange(len(glm_lmdas))
    reverse_indices = reverse_indices[::-1]

    # Apply the same reversal to lmdas to maintain correspondence with coefs
    lmdas = lmdas[reverse_indices]

    gamma_hat, gamma_0, beta_coefs = _format_output(lasso_model,
                                                    beta_coefs_fit,
                                                    beta_intercepts,
                                                    zero_var_idx,
                                                    X,
                                                    fit_intercept,
                                                    reverse_indices)

    if verbose:
        _print_unilasso_results(gamma_hat, lmdas)

    unilasso_result = UniLassoResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=lmdas
    )

    return unilasso_result


def predict(result: UniLassoResult,
            X: np.ndarray, 
            lmda_idx: Optional[int] = None) -> np.ndarray:
    """
    Predict response variable using UniLasso model.

    Args:
        result: UniLasso result object.
        X: Feature matrix of shape (n, p).
        lmda_idx: Index of the regularization parameter to use for prediction.

    Returns:
        Predicted response variable.
    """

    if not type(result) == UniLassoResult:
        raise ValueError("`result` must be a UniLassoResult object.")
    
    if len(result.coefs.shape) == 1:
        result.coefs = np.expand_dims(result.coefs, axis=0)
    assert result.coefs.shape[1] == X.shape[1], "Feature matrix must have the same number of columns as the fitted model."

    X, _ = _format_unilasso_feature_matrix(X, remove_zero_var=False)
    
    if lmda_idx is not None:
        assert lmda_idx >= 0 and lmda_idx < len(result.lmdas), "Invalid regularization parameter index."
        y_hat = X @ result.coefs[lmda_idx] + result.intercept[lmda_idx]
    else:
        y_hat = X @ result.coefs.T + result.intercept 

    y_hat = y_hat.squeeze()
          
    return y_hat




import numpy as np
import torch
from typing import Optional, Callable
# 假设已经导入了 _format_output, _prepare_unilasso_input, UniLassoCVResult 等原有函数

class PyTorchGrpnetAdapter:
    """
    严肃工程实现：适配器模式。
    作用：将 PyTorch 计算出的权重矩阵和截距数组，包装成类似 ad.grpnet 的对象接口。
    """
    def __init__(self, betas_matrix: np.ndarray, intercepts_array: np.ndarray, lmdas: np.ndarray):
        # 参数 betas_matrix: 形状为 (n_lmdas, n_features) 的特征系数矩阵
        # 参数 intercepts_array: 形状为 (n_lmdas,) 的截距数组
        self._betas = betas_matrix
        self.intercepts = intercepts_array
        self.lmdas = lmdas

    @property
    def betas(self):
        """
        模拟 adelie 中稀疏矩阵的 .toarray() 方法。
        返回一个具有 toarray 方法的内部匿名类，巧妙满足 _format_output 的调用链。
        """
        class MockSparseMatrix:
            def __init__(self, mat):
                self.mat = mat
            def toarray(self):
                return self.mat
                
        return MockSparseMatrix(self._betas)
    
    
def _fit_pytorch_lasso_path(
X_train: np.ndarray,
y_train: np.ndarray,
lmdas: np.ndarray,
negative_penalty: float,
fit_intercept: bool = True,
lr: float = 0.01,
max_epochs: int = 5000,
tol: float = 1e-6,
feature_weights: Optional[np.ndarray] = None,
group_signs: Optional[np.ndarray] = None,
group_penalty: float = 0.0,
group_weights: Optional[np.ndarray] = None,
family: str = "gaussian",
momentum: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    沿着正则化路径 (lambda_path) 训练模型，利用 Warm Start 加速收敛。

    使用双层非对称软阈值算子 (Double Asymmetric Soft-Thresholding)，支持：
    1. 特征级自适应权重惩罚
    2. 组级符号一致性约束

    Parameters
    ----------
    negative_penalty : float
        对负系数的额外惩罚强度。当 negative_penalty -> inf 时，
        模型趋近于硬约束 theta >= 0（但不完全等价）。
        当 negative_penalty = 0 时，退化为标准 Lasso。
    feature_weights : np.ndarray, optional
        特征级显著性权重，默认全为 1
    group_signs : np.ndarray, optional
        每个特征所在组的主导符号，默认全为 1
    group_penalty : float
        全局组惩罚强度，用于符号不一致惩罚
    group_weights : np.ndarray, optional
        组级惩罚权重，默认全为 1
    """
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    n_features = X_train.shape[1]
    n_lmdas = len(lmdas)
    n_samples = X_train.shape[0]

    # Default values for optional parameters
    if feature_weights is None:
        feature_weights = np.ones(n_features)
    if group_signs is None:
        group_signs = np.ones(n_features)
    if group_weights is None:
        group_weights = np.ones(n_features)

    # Convert to torch tensors
    fw_t = torch.tensor(feature_weights, dtype=torch.float32)
    gs_t = torch.tensor(group_signs, dtype=torch.float32)
    gw_t = torch.tensor(group_weights, dtype=torch.float32)

    # 预分配结果存储空间
    betas_matrix = np.zeros((n_lmdas, n_features))
    intercepts = np.zeros(n_lmdas)

    # 初始化可训练参数
    weights = torch.zeros((n_features, 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    # Use Nesterov only when momentum > 0
    use_nesterov = momentum > 0
    optimizer = torch.optim.SGD([weights, bias], lr=lr, momentum=momentum, nesterov=use_nesterov)

    # Initialize momentum buffers
    v_weights = torch.zeros_like(weights)
    v_bias = torch.zeros_like(bias)

    for i, lmda in enumerate(lmdas):
        # Reset momentum when lambda changes (warm start) if momentum is enabled
        if momentum > 0:
            optimizer.param_groups[0]['momentum'] = 0.5
            for param_group in optimizer.param_groups:
                param_group['momentum_buffer'] = None

        # For early stopping: track last 3 changes
        last_changes = torch.zeros(3)
        change_idx = 0

        # 针对当前的 lambda 进行梯度下降更新 (利用了上一轮的 weights 作为起点)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            # 第一步：根据 family 计算对应的损失
            y_pred = torch.matmul(X_t, weights) + bias

            if family == "gaussian":
                # Gaussian: MSE loss
                loss = torch.mean((y_pred - y_t) ** 2)
            elif family == "binomial":
                # Binomial: Logistic loss
                mu = torch.sigmoid(torch.clamp(y_pred, -50, 50))
                loss = -torch.mean(y_t * torch.log(mu + 1e-15) + (1 - y_t) * torch.log(1 - mu + 1e-15))
            elif family == "poisson":
                # Poisson: log-linear loss
                mu = torch.exp(torch.clamp(y_pred, -50, 50))
                loss = torch.mean(mu - y_t * torch.log(mu + 1e-15))
            elif family == "multinomial":
                # Multinomial: treat as binomial
                mu = torch.sigmoid(torch.clamp(y_pred, -50, 50))
                loss = -torch.mean(y_t * torch.log(mu + 1e-15) + (1 - y_t) * torch.log(1 - mu + 1e-15))
            else:
                # Default to Gaussian
                loss = torch.mean((y_pred - y_t) ** 2)

            loss.backward()

            # 走普通梯度下降的一步
            optimizer.step()

            # 第二步：双层非对称近端算子 (Double Asymmetric Proximal Operator)
            with torch.no_grad():
                w_new = torch.zeros_like(weights)

                for j in range(n_features):
                    w = weights[j, 0]
                    fw = fw_t[j]
                    gs = gs_t[j]
                    gw = gw_t[j]

                    # Compute base thresholds (feature adaptive)
                    tau_pos_base = lr * lmda * fw
                    tau_neg_base = lr * (lmda + negative_penalty) * fw

                    # Compute group penalty adjustment
                    if gs > 0:
                        # Group sign is positive: extra penalty for negative values
                        tau_neg = tau_neg_base + lr * group_penalty * gw
                        tau_pos = tau_pos_base
                    else:
                        # Group sign is negative: extra penalty for positive values
                        tau_pos = tau_pos_base + lr * group_penalty * gw
                        tau_neg = tau_neg_base

                    # Apply soft thresholding
                    if w > tau_pos:
                        w_new[j, 0] = w - tau_pos
                    elif w < -tau_neg:
                        w_new[j, 0] = w + tau_neg
                    else:
                        w_new[j, 0] = 0.0

                # 检查收敛
                max_change = torch.max(torch.abs(w_new - weights))
                last_changes[change_idx] = max_change
                change_idx = (change_idx + 1) % 3

                # Converge only if last 3 changes are all below tol
                converged = epoch > 3 and torch.all(last_changes < tol)

                if converged:
                    break

                weights.copy_(w_new)

        betas_matrix[i, :] = weights.detach().numpy().flatten()
        intercepts[i] = bias.detach().item()
    return betas_matrix, intercepts


from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def cv_uni(
    X: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    n_folds: int = 5,
    lmdas: Optional[np.ndarray] = None,
    n_lmdas: int = 100,
    lmda_min_ratio: Optional[float] = None,
    negative_penalty: float = 10.0,
    verbose: bool = False,
    seed: Optional[int] = None,
    backend: str = "numba",
    lr: Optional[float] = None,
    # 新增：自适应惩罚参数 (与 fit_uni 一致)
    adaptive_weighting: bool = False,
    weight_method: str = "t_statistic",
    weight_max_scale: float = 10.0,
    # 新增：组约束参数 (与 fit_uni 一致)
    enable_group_constraint: bool = False,
    corr_threshold: float = 0.7,
    group_penalty: float = 5.0,
    max_group_size: int = 20,
    # 新增：非线性单变量模型参数
    univariate_model: str = "linear",
    spline_df: int = 5,
    spline_degree: int = 3,
    tree_max_depth: int = 2
) -> UniLassoCVResult:
    """
    GroupAdaUniLasso: 交叉验证版的全GLM家族单变量引导 Lasso 回归。

    支持所有 fit_uni 的新功能：特征级自适应惩罚、组级符号一致性约束、
    非线性单变量模型（样条、决策树）。

    Parameters
    ----------
    family : str, optional
        GLM家族: 'gaussian' (线性回归), 'binomial' (二分类),
        'multinomial' (多分类), 'poisson' (泊松回归), 'cox' (生存分析)
    backend : str, optional
        Solver backend to use: 'numba' (default, fast) or 'pytorch' (original).
    adaptive_weighting : bool, optional
        开启特征级自适应惩罚 (默认关闭，与原版行为一致)
    weight_method : str, optional
        显著性权重计算方法: 't_statistic', 'p_value', 'correlation'
    weight_max_scale : float, optional
        权重最大上限 (默认10.0)
    enable_group_constraint : bool, optional
        开启组级一致性约束 (默认关闭，与原版行为一致)
    corr_threshold : float, optional
        共线性分组阈值 (默认0.7)
    group_penalty : float, optional
        全局组惩罚强度 (默认5.0)
    max_group_size : int, optional
        最大组大小 (默认20)
    univariate_model : str, optional
        单变量模型类型: 'linear' (线性，默认), 'spline' (样条), 'tree' (决策树)
    spline_df : int, optional
        样条回归自由度 (univariate_model='spline' 时使用，默认5)
    spline_degree : int, optional
        样条多项式次数 (默认3为三次样条)
    tree_max_depth : int, optional
        决策树最大深度 (univariate_model='tree' 时使用，默认2)
    """
    # 校验 univariate_model 参数
    if univariate_model not in VALID_UNIVARIATE_MODELS:
        raise ValueError(f"univariate_model must be one of {VALID_UNIVARIATE_MODELS}")

    # 向后兼容：当新功能都关闭时，回退到原有行为
    use_original = (not adaptive_weighting) and (not enable_group_constraint) and (univariate_model == "linear")

    # 1. 前置校验
    if backend not in ["numba", "pytorch"]:
        raise ValueError("backend must be either 'numba' or 'pytorch'")

    # 如果新功能关闭且需要用原始adelie（保持向后兼容）
    # 注意：cox模型仍然使用原始adelie，因为需要特殊处理
    if use_original and family == "cox":
        # 回退到原始adelie实现（cox在原始实现中有完整支持）
        return cv_unilasso(X, y, family, n_folds, lmda_min_ratio, verbose, seed)

    # Select solver based on backend
    if backend == "numba":
        _fit_lasso_path = _fit_numba_lasso_path
    else:
        _fit_lasso_path = _fit_pytorch_lasso_path

    # 2. 严格复用原版的数据准备逻辑，获取至关重要的 loo_fits
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(
        X, y, family, lmdas,
        univariate_model=univariate_model,
        spline_df=spline_df,
        spline_degree=spline_degree,
        tree_max_depth=tree_max_depth
    )

    fit_intercept = False if family == "cox" else True

    # 设置默认学习率（根据GLM家族）
    if lr is None:
        if family == "gaussian":
            lr = 0.01
        elif family == "binomial":
            lr = 0.1
        elif family == "poisson":
            lr = 0.05
        elif family == "multinomial":
            lr = 0.1
        else:
            lr = 0.01

    # 3. 计算自适应权重和分组约束 (新功能，在全量数据上计算一次)
    feature_weights = None
    group_signs = None
    group_weights_arr = None
    groups = None

    if adaptive_weighting or enable_group_constraint:
        # 构建单变量结果字典
        univariate_results = {
            'beta': beta_coefs_fit,
            't_stats': np.ones_like(beta_coefs_fit),
            'p_values': np.ones_like(beta_coefs_fit),
            'correlations': np.zeros_like(beta_coefs_fit)
        }

        # 计算特征相关性
        if X.shape[1] > 0:
            if X.shape[1] > 1:
                univariate_results['correlations'] = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
            else:
                univariate_results['correlations'] = np.array([np.corrcoef(X[:, 0], y)[0, 1]])

    if adaptive_weighting:
        feature_weights = _compute_feature_significance_weights(
            univariate_results, weight_method, weight_max_scale
        )
    else:
        feature_weights = np.ones(len(beta_coefs_fit))

    if enable_group_constraint:
        # 计算特征相关矩阵
        if len(beta_coefs_fit) > 1:
            corr_matrix = _parallel_corr_matrix(X)
        else:
            corr_matrix = np.array([[1.0]])

        # 贪心分组
        groups = _greedy_correlation_grouping(
            corr_matrix, corr_threshold, max_group_size
        )

        # 计算组惩罚权重和符号
        group_signs, group_weights_arr = _compute_group_penalty_weights(
            groups, beta_coefs_fit, feature_weights
        )
    else:
        group_signs = np.ones(len(beta_coefs_fit))
        group_weights_arr = np.ones(len(beta_coefs_fit))

    # 4. 确定正则化路径
    if original_lmdas is not None:
        lambda_path = np.sort(np.array(original_lmdas))[::-1]
    else:
        lambda_path = _configure_lmda_path(
            X=loo_fits,
            y=y,
            family=family,
            n_lmdas=n_lmdas,
            lmda_min_ratio=lmda_min_ratio
        )

    # 5. 交叉验证过程
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    avg_losses = np.zeros(len(lambda_path))

    for train_idx, val_idx in kf.split(loo_fits):
        X_train, X_val = loo_fits[train_idx], loo_fits[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        b_mat, int_arr = _fit_lasso_path(
            X_train, y_train, lambda_path, negative_penalty, fit_intercept,
            lr=lr,
            feature_weights=feature_weights,
            group_signs=group_signs,
            group_penalty=group_penalty,
            group_weights=group_weights_arr,
            family=family
        )

        for i in range(len(lambda_path)):
            preds = X_val @ b_mat[i] + int_arr[i]
            if family == "gaussian":
                avg_losses[i] += np.mean((preds - y_val) ** 2) / n_folds
            elif family == "binomial":
                # Logistic loss
                mu = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
                log_loss = -np.mean(y_val * np.log(mu + 1e-15) + (1 - y_val) * np.log(1 - mu + 1e-15))
                avg_losses[i] += log_loss / n_folds
            elif family == "poisson":
                # Poisson loss
                mu = np.exp(np.clip(preds, -50, 50))
                poisson_loss = np.mean(mu - y_val * np.log(mu + 1e-15))
                avg_losses[i] += poisson_loss / n_folds
            elif family == "multinomial":
                # Treat as binomial for now
                mu = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
                log_loss = -np.mean(y_val * np.log(mu + 1e-15) + (1 - y_val) * np.log(1 - mu + 1e-15))
                avg_losses[i] += log_loss / n_folds
            else:
                # Default to MSE
                avg_losses[i] += np.mean((preds - y_val) ** 2) / n_folds

    best_idx = int(np.argmin(avg_losses))
    best_lmda = lambda_path[best_idx]

    # 6. 在全量 LOO 特征上进行最终拟合
    final_betas, final_intercepts = _fit_lasso_path(
        loo_fits, y, lambda_path, negative_penalty, fit_intercept,
        lr=lr,
        feature_weights=feature_weights,
        group_signs=group_signs,
        group_penalty=group_penalty,
        group_weights=group_weights_arr,
        family=family
    )

    # 将最终结果装入我们的适配器
    adapter_model = PyTorchGrpnetAdapter(final_betas, final_intercepts, lambda_path)

    # 7. 调用原生格式化函数
    gamma_hat, gamma_0, beta_coefs = _format_output(
        lasso_model=adapter_model,
        beta_coefs_fit=beta_coefs_fit,
        beta_intercepts=beta_intercepts,
        zero_var_idx=zero_var_idx,
        X=X,
        fit_intercept=fit_intercept
    )

    # 定义一个伪装的可视化函数
    def mock_cv_plot():
        plt.plot(np.log(lambda_path), avg_losses)
        plt.xlabel("Log(Lambda)")
        plt.ylabel("CV MSE Loss")
        plt.title("Custom UniLasso CV Plot")
        plt.show()

    if verbose:
        _print_unilasso_results(gamma_hat, lambda_path, best_idx)
        mock_cv_plot()

    # 8. 返回结果
    result = UniLassoCVResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=adapter_model,
        lmdas=lambda_path,
        avg_losses=avg_losses,
        cv_plot=mock_cv_plot,
        best_idx=best_idx,
        best_lmda=best_lmda
    )

    # 附加分组信息 (可选)
    if enable_group_constraint and groups is not None:
        result.groups = groups
        result.group_signs = group_signs

    return result



from typing import List, Optional, Union
import numpy as np

# 假设已经导入了 _prepare_unilasso_input, _format_output, _print_unilasso_results, UniLassoResult
# 以及我们之前编写的 _fit_pytorch_lasso_path 和 PyTorchGrpnetAdapter

def fit_uni(
    X: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    lmdas: Optional[Union[float, List[float], np.ndarray]] = None,
    n_lmdas: Optional[int] = 100,
    lmda_min_ratio: Optional[float] = 1e-2,
    negative_penalty: float = 10.0,  # 创新点：暴露软惩罚参数
    verbose: bool = False,
    backend: str = "numba",
    lr: Optional[float] = None,
    # 新增：自适应惩罚参数
    adaptive_weighting: bool = False,
    weight_method: str = "t_statistic",
    weight_max_scale: float = 10.0,
    # 新增：组约束参数
    enable_group_constraint: bool = False,
    corr_threshold: float = 0.7,
    group_penalty: float = 5.0,
    max_group_size: int = 20,
    # 新增：非线性单变量模型参数
    univariate_model: str = "linear",
    spline_df: int = 5,
    spline_degree: int = 3,
    tree_max_depth: int = 2
) -> UniLassoResult:
    """
    GroupAdaUniLasso: 全GLM家族升级的单变量引导 Lasso 回归。

    在指定的正则化路径上拟合模型，支持对负系数的自定义软惩罚、
    特征级自适应惩罚、组级符号一致性约束以及非线性单变量模型。

    Parameters
    ----------
    family : str, optional
        GLM家族: 'gaussian' (线性回归), 'binomial' (二分类),
        'multinomial' (多分类), 'poisson' (泊松回归), 'cox' (生存分析)
    backend : str, optional
        Solver backend to use: 'numba' (default, fast) or 'pytorch' (original).
    adaptive_weighting : bool, optional
        开启特征级自适应惩罚 (默认关闭，与原版行为一致)
    weight_method : str, optional
        显著性权重计算方法: 't_statistic', 'p_value', 'correlation'
    weight_max_scale : float, optional
        权重最大上限 (默认10.0)
    enable_group_constraint : bool, optional
        开启组级一致性约束 (默认关闭，与原版行为一致)
    corr_threshold : float, optional
        共线性分组阈值 (默认0.7)
    group_penalty : float, optional
        全局组惩罚强度 (默认5.0)
    max_group_size : int, optional
        最大组大小 (默认20)
    univariate_model : str, optional
        单变量模型类型: 'linear' (线性，默认), 'spline' (样条), 'tree' (决策树)
    spline_df : int, optional
        样条回归自由度 (univariate_model='spline' 时使用，默认5)
    spline_degree : int, optional
        样条多项式次数 (默认3为三次样条)
    tree_max_depth : int, optional
        决策树最大深度 (univariate_model='tree' 时使用，默认2)
    """
    # 校验 univariate_model 参数
    if univariate_model not in VALID_UNIVARIATE_MODELS:
        raise ValueError(f"univariate_model must be one of {VALID_UNIVARIATE_MODELS}")

    # 向后兼容：当新功能都关闭时，回退到原有行为
    use_original = (not adaptive_weighting) and (not enable_group_constraint) and (univariate_model == "linear")

    # 1. 前置校验
    if backend not in ["numba", "pytorch"]:
        raise ValueError("backend must be either 'numba' or 'pytorch'")

    # 如果新功能关闭且需要用原始adelie（保持向后兼容）
    # 注意：cox模型仍然使用原始adelie，因为需要特殊处理
    if use_original and family == "cox":
        # 回退到原始adelie实现（cox在原始实现中有完整支持）
        return fit_unilasso(X, y, family, lmdas, n_lmdas, lmda_min_ratio, verbose)

    # Select solver based on backend
    if backend == "numba":
        _fit_lasso_path = _fit_numba_lasso_path
    else:
        _fit_lasso_path = _fit_pytorch_lasso_path

    # 2. 数据准备与预处理
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(
        X, y, family, lmdas,
        univariate_model=univariate_model,
        spline_df=spline_df,
        spline_degree=spline_degree,
        tree_max_depth=tree_max_depth
    )

    fit_intercept = False if family == "cox" else True

    # 设置默认学习率（根据GLM家族）
    if lr is None:
        if family == "gaussian":
            lr = 0.01
        elif family == "binomial":
            lr = 0.1
        elif family == "poisson":
            lr = 0.05
        elif family == "multinomial":
            lr = 0.1
        else:
            lr = 0.01

    # 3. 确定正则化路径 (Lambda Path)
    if original_lmdas is not None:
        lambda_path = np.sort(np.array(original_lmdas))[::-1]
    else:
        lambda_path = _configure_lmda_path(
                    X=loo_fits,
                    y=y,
                    family=family,
                    n_lmdas=n_lmdas,
                    lmda_min_ratio=lmda_min_ratio
                )

    # 4. 计算自适应权重和分组约束 (新功能)
    feature_weights = None
    group_signs = None
    group_weights_arr = None
    groups = None

    if adaptive_weighting or enable_group_constraint:
        # 构建单变量结果字典
        univariate_results = {
            'beta': beta_coefs_fit,
            't_stats': np.ones_like(beta_coefs_fit),  # 占位
            'p_values': np.ones_like(beta_coefs_fit),  # 占位
            'correlations': np.zeros_like(beta_coefs_fit)  # 占位
        }

        # 计算特征相关性
        if X.shape[1] > 0:
            univariate_results['correlations'] = np.corrcoef(X.T, y)[0, 1:] if X.shape[1] > 1 else np.array([np.corrcoef(X[:, 0], y)[0, 1]])

    if adaptive_weighting:
        feature_weights = _compute_feature_significance_weights(
            univariate_results, weight_method, weight_max_scale
        )
    else:
        feature_weights = np.ones(len(beta_coefs_fit))

    if enable_group_constraint:
        # 计算特征相关矩阵
        if len(beta_coefs_fit) > 1:
            corr_matrix = _parallel_corr_matrix(X)
        else:
            corr_matrix = np.array([[1.0]])

        # 贪心分组
        groups = _greedy_correlation_grouping(
            corr_matrix, corr_threshold, max_group_size
        )

        # 计算组惩罚权重和符号
        group_signs, group_weights_arr = _compute_group_penalty_weights(
            groups, beta_coefs_fit, feature_weights
        )
    else:
        group_signs = np.ones(len(beta_coefs_fit))
        group_weights_arr = np.ones(len(beta_coefs_fit))

    # 5. 调用核心求解器
    betas_matrix, intercepts_array = _fit_lasso_path(
        X_train=loo_fits,
        y_train=y,
        lmdas=lambda_path,
        negative_penalty=negative_penalty,
        fit_intercept=fit_intercept,
        lr=lr,
        feature_weights=feature_weights,
        group_signs=group_signs,
        group_penalty=group_penalty,
        group_weights=group_weights_arr,
        family=family
    )

    # 6. 适配器模式启动
    adapter_model = PyTorchGrpnetAdapter(betas_matrix, intercepts_array, lambda_path)

    # 7. 参数还原与格式化
    gamma_hat, gamma_0, beta_coefs = _format_output(
        lasso_model=adapter_model,
        beta_coefs_fit=beta_coefs_fit,
        beta_intercepts=beta_intercepts,
        zero_var_idx=zero_var_idx,
        X=X,
        fit_intercept=fit_intercept,
        reverse_indices=None
    )

    if verbose:
        _print_unilasso_results(gamma_hat, lambda_path)

    # 8. 返回标准结果对象 (附加分组信息)
    unilasso_result = UniLassoResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=adapter_model,
        lmdas=lambda_path
    )

    # 附加分组信息 (可选)
    if enable_group_constraint and groups is not None:
        unilasso_result.groups = groups
        unilasso_result.group_signs = group_signs

    return unilasso_result

def plot_uni(unilasso_fit) -> None:
    """
    创新版 Lasso 路径可视化。
    使用行业标准 -log(lambda) 作为横轴，使得从左到右代表模型复杂度增加（正则化减弱）。
    """
    assert hasattr(unilasso_fit, "coefs") and hasattr(unilasso_fit, "lmdas"), \
        "Input must have 'coefs' and 'lmdas' attributes."

    coefs, lambdas = unilasso_fit.coefs, unilasso_fit.lmdas
    
    if coefs.ndim == 1 or len(lambdas) == 1:
        print("Only one regularization parameter was used. No path to plot.")
        return

    plt.figure(figsize=(8, 6))
    
    # --- 核心改动点 ---
    # 改为使用 -log(lambda)，这是业界更通用的做法
    neg_log_lambdas = -np.log(lambdas) 

    # Compute the number of nonzero coefficients at each lambda
    n_nonzero = np.sum(coefs != 0, axis=1)

    # Plot coefficient paths
    for i in range(coefs.shape[1]):  
        plt.plot(neg_log_lambdas, coefs[:, i], lw=2)

    # --- 标签更新 ---
    plt.xlabel(r"$-\log(\lambda)$", fontsize=12)  # 更新 LaTeX 标签
    plt.ylabel("Coefficients", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # 添加顶部的第二 X 轴（用于显示非零特征数量）
    ax1 = plt.gca()  
    ax2 = ax1.twiny()  
    ax2.set_xlim(ax1.get_xlim())  
    
    # 动态计算刻度，保持美观
    tick_indices = np.linspace(0, len(neg_log_lambdas) - 1, min(6, len(neg_log_lambdas)), dtype=int)
    ax2.set_xticks(neg_log_lambdas[tick_indices])  
    ax2.set_xticklabels(n_nonzero[tick_indices]) 
    
    ax2.set_xlabel("Number of Active Coefficients", fontsize=12)

    plt.show()
    
    
def plot_cv_uni(cv_result) -> None:
    """
    创新版交叉验证损失曲线可视化。
    彻底接管绘图逻辑，使用 -log(lambda) 作为横轴，并高亮最佳参数点。
    """
    # 1. 前置契约校验 (Contract Check)
    assert hasattr(cv_result, "lmdas") and hasattr(cv_result, "avg_losses"), \
        "输入对象必须包含 'lmdas' 和 'avg_losses' 属性"
    
    # 2. 从对象中提取原始数据 (打破原先的黑盒调用机制)
    lambdas = cv_result.lmdas
    avg_losses = cv_result.avg_losses
    best_lmda = cv_result.best_lmda
    
    # --- 核心改动：转换为 -log ---
    neg_log_lambdas = -np.log(lambdas)
    
    plt.figure(figsize=(8, 6))
    
    # 绘制交叉验证平均损失曲线
    plt.plot(
        neg_log_lambdas, 
        avg_losses, 
        marker='o', 
        linestyle='-', 
        color='royalblue', 
        markersize=5, 
        linewidth=2,
        label='CV MSE Loss'
    )
    
    # --- 工业级体验增强：标出最佳点 ---
    if best_lmda is not None:
        best_neg_log = -np.log(best_lmda)
        # 获取最佳 lambda 对应的 loss 值用于画点
        best_idx = cv_result.best_idx
        best_loss = avg_losses[best_idx]
        
        # 画一条垂直虚线辅助对齐
        plt.axvline(x=best_neg_log, color='crimson', linestyle='--', alpha=0.7, label=f'Best $-\\log(\\lambda)$ = {best_neg_log:.2f}')
        # 在最低点画一个红星
        plt.plot(best_neg_log, best_loss, marker='*', color='crimson', markersize=12)

    # 设置符合行业标准的标签
    plt.xlabel(r"$-\log(\lambda)$", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title("Cross-Validation Error Curve", fontsize=14, pad=15)
    
    # 优化网格和图例，提升图表可读性
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True)
    
    plt.show()