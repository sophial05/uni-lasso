__all__ = ["fit_unilasso", "fit_loo_univariate_models"]


from .uni_lasso import fit_unilasso, cv_unilasso, predict, extract_cv_unilasso, plot_coef_path
from .univariate_regression import fit_loo_univariate_models
from .utils import setup_warnings, simulate_gaussian_data, simulate_binomial_data, simulate_cox_data


setup_warnings()

# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Define exported functions
__all__ = ['fit_unilasso', 'cv_unilasso', 'extract_cv_unilasso', 'plot_coef_path',
           'predict',
           'fit_loo_univariate_models',
           'simulate_gaussian_data', 'simulate_binomial_data', 'simulate_cox_data']


# Package metadata
__version__ = "1.0.0"
__author__ = "Sophia Lu"
__email__ = "sophialu@stanford.edu"


__doc__ = """
UniLasso: A package for performing Univariate-Guided Sparse Regression.
See arxiv paper: https://arxiv.org/abs/2501.18360

This package provides implementations of UniLasso and cross-validated UniLasso,
along with utility functions for handling different types of generalized linear 
models (Gaussian, Binomial, Cox).
"""