"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""



import numpy as np
import pandas as pd
from numba import jit
import adelie as ad


from typing import List, Optional, Tuple, Union, Callable
import logging

from .univariate_regression import fit_loo_univariate_models
from .config import VALID_FAMILIES
from .utils import warn_zero_variance, warn_removed_lmdas


# Configure logger
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


class UniLassoResult:
    """
    Class to store results of UniLasso, encapsulating model outputs with hidden attributes.
    """

    def __init__(self, 
                 coef: np.ndarray, 
                 intercept: np.ndarray, 
                 family: str,
                 gamma: np.ndarray, 
                 gamma_intercept: np.ndarray, 
                 beta: np.ndarray, 
                 beta_intercepts: np.ndarray, 
                 lasso_model: ad.grpnet, 
                 lmdas: np.ndarray, 
                 avg_losses: Optional[np.ndarray] = None,
                 cv_plot: Optional[Callable] = None, 
                 best_idx: Optional[int] = None,
                 best_lmda: Optional[float] = None):
        """
        Initializes the UniLasso result object.

        Parameters:
        - coef (np.ndarray): Coefficients of the univariate-guided lasso.
        - intercept (np.ndarray): Intercept of the univariate-guided lasso.
        - family (str): Family of the response variable ('gaussian', 'binomial', 'cox').
        - gamma (np.ndarray): Coefficients of the univariate-guided lasso; hidden attribute to avoid confusion.
        - gamma_intercept (np.ndarray): Intercept of the univariate-guided lasso, hidden attribute to avoid confusion.
        - beta (np.ndarray): Coefficients of the univariate regression, hidden attribute to avoid confusion.
        - beta_intercepts (np.ndarray): Intercept of the univariate regression, hidden attribute to avoid confusion.
        - lasso_model (ad.grpnet): The fitted Lasso model.
        - lmdas (np.ndarray): Regularization path.
        - cv_plot (Optional[Callable]): Function to generate cross-validation plot, if available.
        - best_idx (Optional[int]): Index of the best-performing regularization parameter.
        - best_lmda (Optional[float]): Best regularization parameter.
        """
        self.coef = coef
        self.intercept = intercept
        self.family = family
        self._gamma = gamma  
        self._gamma_intercept = gamma_intercept  
        self._beta = beta  
        self._beta_intercepts = beta_intercepts  
        self.lasso_model = lasso_model
        self.lmdas = lmdas
        self.avg_losses = avg_losses
        self.cv_plot = cv_plot
        self.best_idx = best_idx
        self.best_lmda = best_lmda

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
        return (f"UniLassoResult(coef={self.coef.shape}, intercept={self.intercept.shape}, "
                f"lasso_model={type(self.lasso_model).__name__}, lmdas={self.lmdas})")
    



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
                lmdas: Optional[Union[float, List[float], np.ndarray]]
) -> Tuple[np.ndarray, 
           np.ndarray, 
           np.ndarray, 
           ad.glm.GlmBase64, 
           List[ad.constraint.ConstraintBase64], 
           Optional[np.ndarray]]:
    """Prepare input for UniLasso."""
    X, y, family, lmdas, zero_var_idx = _format_unilasso_input(X, y, family, lmdas)

    loo_fits, beta_intercepts, beta_coefs_fit = fit_univariate_models(X, y, family=family)
    loo_fits = np.asfortranarray(loo_fits)

    glm_family = _get_glm_family(family, y)
    constraints = [ad.constraint.lower(b=np.zeros(1)) for _ in range(X.shape[1])]

    return loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx




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
            lmdas: np.ndarray, 
            best_idx: Optional[int] = None
) -> None:
    """Print UniLasso results."""
    num_selected = np.sum(theta_hat != 0, axis=1)

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



# ------------------------------------------------------------------------------
# Perform cross-validation UniLasso
# ------------------------------------------------------------------------------


def cv_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            n_folds: int = 5,
            lmda_min_ratio: float = 1e-5,
            verbose: bool = False,
            seed: Optional[int] = None
) ->  UniLassoResult:
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
    loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X, y, family, None)
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

    


    if verbose:
        _print_unilasso_results(gamma_hat, cv_lasso.lmdas, int(cv_lasso.best_idx))

    unilasso_result = UniLassoResult(
        coef=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=cv_lasso.lmdas,
        avg_losses=cv_lasso.avg_losses,
        cv_plot=cv_lasso.plot_loss,
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
            verbose: bool = False
) -> UniLassoResult:
    """
    Perform UniLasso with specified regularization parameters.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        lmdas: Lasso regularization parameter(s).
        verbose: Whether to print results.

    Returns:
        Dictionary containing UniLasso results.
    """
    loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx = _prepare_unilasso_input(X, y, family, lmdas)
    assert lmdas is not None, "Regularizers must be specified for UniLasso."

    fit_intercept = False if family == "cox" else True

    lasso_model = ad.grpnet(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=fit_intercept,
        lmda_path=lmdas,
        constraints=constraints,
    )

    glm_lmdas = np.array(lasso_model.lmdas)

    if not np.all(np.isin(lmdas, glm_lmdas)):
        warn_removed_lmdas(np.setdiff1d(lmdas, glm_lmdas))

    matching_idx = np.where(np.isin(glm_lmdas, lmdas))[0]
    lmdas = lmdas[matching_idx]

    if len(lmdas) == 0:
        raise ValueError("No regularization strengths remain after removing invalid values")

    reverse_indices = np.arange(len(glm_lmdas))
    reverse_indices = reverse_indices[::-1]


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
        coef=gamma_hat,
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


def predict(X: np.ndarray, 
            result: UniLassoResult,
            lmda_idx: Optional[int] = None) -> np.ndarray:
    """
    Predict response variable using UniLasso model.

    Args:
        X: Feature matrix of shape (n, p).
        result: UniLasso result object.
        lmda_idx: Index of the regularization parameter to use for prediction.

    Returns:
        Predicted response variable.
    """

    if not type(result) == UniLassoResult:
        raise ValueError("`result` must be a UniLassoResult object.")
    
    X, _ = _format_unilasso_feature_matrix(X, remove_zero_var=False)

    if lmda_idx is not None:
        assert lmda_idx >= 0 and lmda_idx < len(result.lmdas), "Invalid regularization parameter index."
        y_hat = X @ result.coef[lmda_idx] + result.intercept[lmda_idx]
    else:
        y_hat = X @ result.coef.T + result.intercept 
          
    return y_hat
