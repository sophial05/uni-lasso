"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core functions for Univariate-Guided Lasso regression.
Reference: https://arxiv.org/abs/2501.18360
"""


import numpy as np
from numba import jit, prange
from typing import Dict, Tuple, Optional



# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

@jit(nopython=True, parallel=True, cache=True)
def std_axis_0(a):
    res = np.empty(a.shape[0])
    for i in prange(a.shape[0]):
        res[i] = np.std(a[i])
    return res

@jit(nopython=True, parallel=True, cache=True)
def mean_axis_0(a):
    res = np.empty(a.shape[0])
    for i in prange(a.shape[0]):
        res[i] = np.mean(a[i])
    return res


# ------------------------------------------------------------------------------
# Leave-One-Out (LOO) Computation for Univariate Regression
# ------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def compute_loo_coef_numba(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    n, p = X.shape
    y = y.flatten()

    xbar = mean_axis_0(X.T)
    s = std_axis_0(X.T)
    ybar = np.mean(y)

    Xs = (X - xbar) / s

    beta = Xs.T @ y / n
    centered_y = y - ybar
    
    Ri = np.empty((n, p))
    for i in prange(p):
        Ri[:, i] = n * (centered_y - Xs[:, i] * beta[i]) / (n - 1 - Xs[:, i]**2)
    
    F = y[:, np.newaxis] - Ri

    beta = beta / s
    beta0 = ybar - xbar * beta
    
    return F, beta, beta0



def leave_one_out(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate regression models.
    
    Args:
        X: n x p model matrix
        y: n-vector response    
        
    Returns:
    A dictionary containing:
    - "F": Prevalidated fit matrix (leave-one-out predictions)
    - "beta": Univariate regression coefficients for each column of X
    - "beta0": Intercepts for each regression model
    """
    F, beta, beta0 = compute_loo_coef_numba(X, y)
    return {"F": F, "beta": beta, "beta0": beta0}


# ------------------------------------------------------------------------------
# Leave-One-Out for Logistic Regression
# ------------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def wlsu_numba(X: np.ndarray, W: np.ndarray, Z: np.ndarray):
    totW = np.sum(W, axis=0)
    xbar = np.sum(W * X, axis=0) / totW
    Xm = X - xbar
    s = np.sqrt(np.sum(W * Xm**2, axis=0) / totW)
    Xs = Xm / s
    beta = np.sum(Xs * W * Z, axis=0) / totW
    beta0 = np.sum(W * Z, axis=0) / totW
    Eta = beta0 + Xs * beta
    return beta, beta0, Eta, xbar, s

@jit(nopython=True, cache=True)
def compute_loo_coef_binary_numba(X: np.ndarray, y: np.ndarray, nit: int = 2):
    n, p = X.shape
    y = y.flatten()
    
    mus = (y + 0.5) / 2
    w = mus * (1 - mus)
    etas = np.log(mus / (1 - mus))
    z = etas + (y - mus) / w
    W = w[:, None]
    Z = z[:, None]
    
    beta, beta0, Eta, xbar, s = wlsu_numba(X, W, Z)
    
    for _ in range(1, nit):
        Mus = 1 / (1 + np.exp(-Eta))
        W = Mus * (1 - Mus)
        Z = Eta + (y[:, None] - Mus) / W
        beta, beta0, Eta, xbar, s = wlsu_numba(X, W, Z)
    
    Ws = np.sqrt(W / (np.sum(W, axis=0)/n))
    Xs = Ws * (X - xbar) / s
    Ri = (n * (Ws * (Z - beta0) - Xs * beta)) / (n - Ws**2 - Xs**2)
    F = Z - Ri / Ws
    
    return F, beta / s, beta0 - xbar * beta / s

def leave_one_out_logistic(X: np.ndarray, y: np.ndarray, nit: int = 2) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate logistic regression models.
    
    Args:
        X: n x p model matrix
        y: n-vector binary response
        nit: Number of iterations for the optimization
    
    Returns:
        A dictionary containing:
        - "F": Prevalidated fit matrix (leave-one-out predictions)
        - "beta": Univariate regression coefficients for each column of X
        - "beta0": Intercepts for each regression model
    """
    F, beta, beta0 = compute_loo_coef_binary_numba(X, y, nit)
    return {"F": F, "beta": beta, "beta0": beta0}



# ------------------------------------------------------------------------------
# Leave-One-Out for Cox Model
# ------------------------------------------------------------------------------

def identify_unique_and_tied_groups(values, indices):
    """
    Function to identify the first occurrences of unique values 
    and groups of tied indices in a NumPy array.

    Args:
    values: NumPy array of values to analyze for uniqueness and ties.
    indices: NumPy array of indices corresponding to values.

    Returns:
    A dictionary containing:
    - "index_first": Indices of the first occurrence of unique values in `values`.
    - "index_ties": Dictionary mapping tied values to a list of their corresponding indices.
    """
    if not isinstance(values, np.ndarray):
        values = np.asarray(values)
    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices)

    # Identify unique values, their first occurrences, and counts
    unique_values, first_occurrences, counts = np.unique(values, return_index=True, return_counts=True)

    # Get the first occurrence indices
    index_first = indices[first_occurrences]

    # Identify tied values (those with counts > 1)
    tied_mask = counts > 1
    tied_values = unique_values[tied_mask]

    # Map tied values to their indices
    index_ties = {
        val: indices[values == val].tolist()
        for val in tied_values
    }

    return {"index_first": index_first.tolist(), "index_ties": index_ties}


def coxgradu(eta: np.ndarray, 
             time: np.ndarray, 
             d: np.ndarray, 
             w: Optional[np.ndarray] = None, 
             o: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Args:
    eta: np.array, shape (n, p), matrix of univariate Cox fits.
    time: np.array, shape (n,), survival times.
    d: np.array, shape (n,), status indicators (1 if event occurred, 0 if censored).
    w: np.array, shape (n,), optional weights. Default is equal weighting.
    o: np.array, shape (n,), optional order vector. Default is ordering by time and status.

    Returns:
    A dictionary containing:
    - grad: Gradient matrix, shape (n, p).
    - diag_hessian: Diagonal Hessian matrix, shape (n, p).
    - o: Order vector used in the computation.
    """
    if not isinstance(eta, np.ndarray):
        eta = np.asarray(eta)
    if not isinstance(time, np.ndarray):
        time = np.asarray(time)
    if not isinstance(d, np.ndarray):
        d = np.asarray(d)

    p = eta.shape[1]
    nobs = len(time)
    if w is None:
        w = np.ones(nobs)
    w = w / np.sum(w)  # Normalize weights
    eta = eta - np.mean(eta, axis=0)  # Center eta to prevent large exponents

    # Order time, d, and w; for ties, prioritize events over censored
    if o is None:
        o = np.lexsort((-d, time))
    exp_eta = np.exp(eta[o, :])
    time = time[o]
    d = d[o]
    w = w[o]

    # Compute cumulative risk denominator
    rskden = np.flip(np.cumsum(np.flip(exp_eta * w[:, None], axis=0), axis=0), axis=0)

    # Identify tied death times
    tied_deaths = identify_unique_and_tied_groups(time[d == 1], np.arange(len(d))[d == 1])

    # Adjust weights and indicators for tied deaths
    dd = d.copy()
    ww = w.copy()
    tied_groups = tied_deaths['index_ties']
    if len(tied_groups) > 0:
        for group in tied_groups.values():
            if len(group) > 1:
                dd[group] = 0
                ww[group[0]] = np.sum(w[group])
    unique_idx = tied_deaths['index_first']
    if len(unique_idx) > 0:
        dd[unique_idx] = 1

    # Cumulative counts for risk sets
    rskcount = np.cumsum(dd)

    # Compute partial sums of 1 / rskden at risk sets
    valid_idx = np.where(dd == 1)[0]
    rskdeninv = ww[valid_idx, None] / rskden[valid_idx, :]
    rskdeninv = np.cumsum(rskdeninv, axis=0)
    rskdeninv = np.concatenate([np.zeros((1, p)), rskdeninv], axis=0)

    # Compute gradient
    grad = w[:, None] * (d[:, None] - exp_eta * rskdeninv[rskcount, :])
    grad = grad[np.argsort(o), :]

    # Compute diagonal of Hessian
    rskdeninv2 = ww[valid_idx, None] / (rskden[valid_idx, :] ** 2)
    rskdeninv2 = np.cumsum(rskdeninv2, axis=0)
    rskdeninv2 = np.concatenate([np.zeros((1, p)), rskdeninv2], axis=0)
    w_exp_eta = w[:, None] * exp_eta
    diag_hessian = (
        w_exp_eta**2 * rskdeninv2[rskcount, :]
        - w_exp_eta * rskdeninv[rskcount, :]
    )
    diag_hessian = diag_hessian[np.argsort(o), :]

    return {"grad": grad, "diag_hessian": diag_hessian, "o": o}


def leave_one_out_cox(X : np.ndarray,
                         y : np.ndarray,
                         nit : int = 2) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate Cox regression models.
    
    Args:
    X: NumPy array, n x p model matrix.
    y: NumPy structured array with fields 'time' and 'status'.
    nit: Number of iterations for the optimization.

    Returns:
    A dictionary containing:
    - "F": Prevalidated fit matrix (leave-one-out predictions).
    - "beta": Univariate regression coefficients for each column of X.
    - "beta0": Intercept term (initialized to zeros).
    """
    _, p = X.shape
    time = y["time"]
    d = y["status"]
    n = len(time)

    time = np.asarray(time)
    d = np.asarray(d)

    # Initialization
    eta = np.zeros((n, p))
    gradob = coxgradu(eta, time, d)
    o = gradob["o"]
    W = -gradob["diag_hessian"]
    Z = eta + gradob["grad"] / W

    # Weighted least squares update
    def wlsu_ni(X, W, Z):
        beta =  np.sum(X * W * Z, axis=0) / np.sum(X**2 * W, axis=0)
        eta = X * beta
        return {"beta": beta, "eta": eta}

    wob = wlsu_ni(X, W, Z)
    for _ in range(nit):
        gradob = coxgradu(wob["eta"], time, d, o=o)
        W = -gradob["diag_hessian"]
        Z = wob["eta"] + gradob["grad"] / W
        wob = wlsu_ni(X, W, Z)

    X2w = X**2 * W
    X2w = X2w / X2w.sum(axis=0) 
    Ri = (Z - X * wob["beta"]) / (1 - X2w)

    return {
        "F": Z - Ri,
        "beta": wob["beta"],
        "beta0": np.zeros(p)  # Intercept terms set to zeros
    }