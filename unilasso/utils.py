import warnings
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# Warning Utilities
# ------------------------------------------------------------------------------

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    colorama_available = True
except ImportError:
    colorama_available = False


def in_interactive_mode():
    """
    Check if the code is running in an interactive environment (e.g. Jupyter Notebook, IPython, Python REPL).
    """
    try:
        # Jupyter Notebook / IPython
        get_ipython()
        return True
    except NameError:
        # Standard Python REPL or script
        return hasattr(sys, 'ps1')


@contextmanager
def colored_output():
    if colorama_available:
        yield Fore, Style
    else:
        class DummyColor:
            def __getattr__(self, name):
                return ''
        yield DummyColor(), DummyColor()


def custom_warning_formatter(message, category, filename, lineno, line=None):
    if in_interactive_mode():
        from IPython.display import HTML, display
        display(HTML(f'<p style="color:red; font-weight:bold;"> {message}</p>'))
        return ""  
    else:
        with colored_output() as (Fore, Style):
            return f"{Fore.RED}Warning: {message}{Style.RESET_ALL}\n"

def setup_warnings():
    warnings.formatwarning = custom_warning_formatter

def warn_zero_variance(num_removed, total_features):
    warning_message = f"{num_removed} out of {total_features} features have zero variance and will be removed."
    if in_interactive_mode():
        from IPython.display import HTML, display
        display(HTML(f'<p style="color:red; font-weight:bold;"> {warning_message}</p>'))
    else:
        warnings.warn(warning_message)


def warn_removed_lmdas(removed_lmdas):
    warning_message = f"Warning: The following regularization strengths were removed: {removed_lmdas}"
    if in_interactive_mode():
        from IPython.display import HTML, display
        display(HTML(f'<p style="color:red; font-weight:bold;"> {warning_message}</p>'))
    else:
        warnings.warn(warning_message)




# ------------------------------------------------------------------------------
# Simulation Utilities
# ------------------------------------------------------------------------------

def simulate_gaussian_data(n=1000,
                           p=5,
                           x_mean=0,
                           x_sd=1,
                           ysd=1,
                           beta=None,
                           seed=None):
    """
    Simulates Gaussian data from a linear regression model.

    Parameters:
    - n (int): Number of observations.
    - p (int): Number of covariates.
    - x_mean (float): Mean of the covariates.
    - x_sd (float): Standard deviation of the covariates.
    - ysd (float): Standard deviation of the noise.
    - beta (ndarray): True regression coefficients (if None, generated randomly).
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - X : np.ndarray, y : np.ndarray
    """
    if seed:
        np.random.seed(seed)

    if beta is None:
        beta = np.random.uniform(-1, 1, size=p)
    
    X = np.random.normal(size=(n, p), loc=x_mean, scale=x_sd)
    y = X @ beta + np.random.normal(size=n)

    return X, y


def simulate_binomial_data(n=1000,
                           p=5,
                           x_mean=0,
                           x_sd=1,
                           beta=None,
                           seed=None):
    """
    Simulates binomial data from a logistic regression model.
    
    Parameters:
    - n (int): Number of observations.
    - p (int): Number of covariates.
    - x_mean (float): Mean of the covariates.
    - x_sd (float): Standard deviation of the covariates.
    - beta (ndarray): True regression coefficients (if None, generated randomly).
    - seed (int): Random seed for reproducibility.

    Returns:
    - X : np.ndarray, y : np.ndarray
    """
    if seed:
        np.random.seed(seed)

    if beta is None:
        beta = np.random.uniform(-1, 1, size=p)
    
    X = np.random.normal(size=(n, p), loc=x_mean, scale=x_sd)
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p, size=n)

    return X, y



def simulate_cox_data(n=1000, 
                      p=5, 
                      x_mean=0,
                      x_sd=1,
                      beta=None, 
                      censoring_rate=0.3,
                      seed=None):
    """
    Simulates survival data from a Cox proportional hazards model.

    Parameters:
    - n (int): Number of observations.
    - p (int): Number of covariates.
    - beta (ndarray): True regression coefficients (if None, generated randomly).
    - censoring_rate (float): Approximate proportion of censored data.
    - seed (int): Random seed for reproducibility.

    Returns:
    - X : np.ndarray, y : pd.DataFrame with columns ["time", "status"]
    """
    if seed:
        np.random.seed(seed)

    X = np.random.normal(size=(n, p), loc=x_mean, scale=x_sd)

    if beta is None:
        beta = np.random.uniform(-1, 1, size=p)

    # Compute hazard rate exp(X beta)
    lambda_ = np.exp(X @ beta)

    # Generate status times using an exponential distribution
    U = np.random.uniform(size=n)
    T = -np.log(U) / lambda_  # Inverse transform sampling

    # Generate censoring times from an independent distribution
    C = np.random.exponential(scale=np.quantile(T, 1 - censoring_rate), size=n)

    #  Observed time = min(T, C), status indicator = 1 if status observed
    time = np.minimum(T, C)
    status = (T <= C).astype(int)

    y = pd.DataFrame({'time': time, 'status': status})

    return X, y



