"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""



import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
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



def plot_coef_path(unilasso_fit) -> None:
    """
    Plots the Lasso coefficient paths as a function of the regularization parameter (lambda).

    Parameters:
    - unilasso_fit: UniLassoResult object containing fitted coefficients and lambda values.
    """
    
    assert hasattr(unilasso_fit, "coefs") and hasattr(unilasso_fit, "lmdas"), \
        "Input must have 'coefs' and 'lmdas' attributes."

    coefs, lambdas = unilasso_fit.coefs, unilasso_fit.lmdas
    if coefs.ndim == 1 or len(lambdas) == 1:
        print("Only one regularization parameter was used. No path to plot.")

    plt.figure(figsize=(8, 6))
    log_lambdas = np.log(lambdas)  

    for i in range(coefs.shape[1]):  
        plt.plot(log_lambdas, coefs[:, i], label=f"Feature {i}", lw=2)

    plt.xlabel(r"$\log(\lambda)$", fontsize=12)
    plt.ylabel("Coefficients", fontsize=12)
    plt.title("Lasso Coefficient Paths", fontsize=14)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
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
          
    return y_hat
