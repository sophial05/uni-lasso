"""
UniLasso: Univariate-Guided Sparse Regression

This file contains core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""


import numpy as np
from numba import jit, prange
from typing import Dict, Optional, Tuple



# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def std_axis_0(a):
    res = np.empty(a.shape[0])
    for i in prange(a.shape[0]):
        res[i] = np.std(a[i])
    return res

@jit(nopython=True, cache=True)
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
    
    fit = y[:, np.newaxis] - Ri

    beta = beta / s
    beta0 = ybar - xbar * beta
    
    return fit, beta, beta0



def leave_one_out(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate regression models.
    
    Args:
        X: n x p model matrix
        y: n-vector response    
        
    Returns:
    A dictionary containing:
    - "fit": Prevalidated fit matrix (leave-one-out predictions)
    - "beta": Univariate regression coefficients for each column of X
    - "beta0": Intercepts for each regression model
    """
    fit, beta, beta0 = compute_loo_coef_numba(X, y)
    return {"fit": fit, "beta": beta, "beta0": beta0}


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
    fit = Z - Ri / Ws
    
    return fit, beta / s, beta0 - xbar * beta / s

def leave_one_out_logistic(X: np.ndarray, y: np.ndarray, nit: int = 2) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate binomial regression models.
    
    Args:
        X: n x p model matrix
        y: n-vector binary response
        nit: Number of iterations for the optimization
    
    Returns:
        A dictionary containing:
        - "fit": Prevalidated fit matrix (leave-one-out predictions)
        - "beta": Univariate regression coefficients for each column of X
        - "beta0": Intercepts for each regression model
    """
    fit, beta, beta0 = compute_loo_coef_binary_numba(X, y, nit)
    return {"fit": fit, "beta": beta, "beta0": beta0}



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
    eta: np.array, shape (n, p), matrix of univariate Cox fit.
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
    rskcount = np.cumsum(dd, dtype=int)

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


# Weighted least squares update
def wlsu_ni(X, W, Z):
    beta =  np.sum(X * W * Z, axis=0) / np.sum(X**2 * W, axis=0)
    eta = X * beta
    return {"beta": beta, "eta": eta}
    

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
    - "fit": Prevalidated fit matrix (leave-one-out predictions).
    - "beta": Univariate regression coefficients for each column of X.
    - "beta0": Intercept term (initialized to zeros).
    """
    _, p = X.shape
    time = y[:, 0]
    d = y[:, 1]
    n = len(time)

    # Initialization
    eta = np.zeros((n, p))
    gradob = coxgradu(eta, time, d)
    o = gradob["o"]
    W = -gradob["diag_hessian"]

    if np.any(W == 0):
        raise ValueError("Some entries of Hessian are zero. Check the input data.")
    
    Z = eta + gradob["grad"] / W

    wob = wlsu_ni(X, W, Z)
    for _ in range(nit - 1):
        gradob = coxgradu(wob["eta"], time, d, o=o)
        W = -gradob["diag_hessian"]
        Z = wob["eta"] + gradob["grad"] / W
        wob = wlsu_ni(X, W, Z)

    X2w = X**2 * W
    X2w = X2w / X2w.sum(axis=0) 
    Ri = (Z - X * wob["beta"]) / (1 - X2w)

    return {
        "fit": Z - Ri,
        "beta": wob["beta"],
        "beta0": np.zeros(p)  # Intercept terms set to zeros
    }



@jit(nopython=True, cache=True)
def compute_loo_coef_poisson_numba(X: np.ndarray, y: np.ndarray, nit: int = 3):
    """
    Compute leave-one-out coefficients for Poisson regression using IRLS.
    """
    n, p = X.shape
    y = y.flatten()

    # Initialize with log(y + 1) to avoid log(0)
    eta = np.log(y + 1.0)
    mu = np.exp(eta)

    beta = np.zeros(p)
    beta0 = np.zeros(p)

    for _ in range(nit):
        # IRLS step
        W = mu  # Poisson variance is mu
        Z = eta + (y - mu) / W  # Working response
        W_mat = W[:, None]
        Z_mat = Z[:, None]

        # Weighted least squares
        totW = np.sum(W_mat, axis=0)
        xbar = np.sum(W_mat * X, axis=0) / totW
        Xm = X - xbar
        s = np.sqrt(np.sum(W_mat * Xm**2, axis=0) / totW)
        Xs = Xm / s
        beta_j = np.sum(Xs * W_mat * Z_mat, axis=0) / totW
        beta0_j = np.sum(W_mat * Z_mat, axis=0) / totW
        Eta = beta0_j + Xs * beta_j

        # Update eta and mu
        eta = Eta
        mu = np.exp(eta)

    # Compute LOO using the formula from logistic but adapted for Poisson
    Ws = np.sqrt(W / (np.sum(W, axis=0)/n))
    Xs = Ws * (X - xbar) / s
    Ri = (n * (Ws * (Z - beta0_j) - Xs * beta_j)) / (n - Ws**2 - Xs**2)
    fit = Z - Ri / Ws

    return fit, beta_j / s, beta0_j - xbar * beta_j / s


def leave_one_out_poisson(X: np.ndarray, y: np.ndarray, nit: int = 3) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate Poisson regression models.

    Args:
        X: n x p model matrix
        y: n-vector of count responses (non-negative integers)
        nit: Number of iterations for the optimization

    Returns:
        A dictionary containing:
        - "fit": Prevalidated fit matrix (leave-one-out predictions)
        - "beta": Univariate regression coefficients for each column of X
        - "beta0": Intercepts for each regression model
    """
    fit, beta, beta0 = compute_loo_coef_poisson_numba(X, y, nit)
    return {"fit": fit, "beta": beta, "beta0": beta0}


def leave_one_out_multinomial(X: np.ndarray, y: np.ndarray, nit: int = 2) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate multinomial regression models.
    For simplicity, this uses one-vs-rest approach by treating each class independently.

    Args:
        X: n x p model matrix
        y: n-vector of class labels (integers from 0 to K-1)
        nit: Number of iterations for the optimization

    Returns:
        A dictionary containing:
        - "fit": Prevalidated fit matrix (leave-one-out predictions)
        - "beta": Univariate regression coefficients for each column of X
        - "beta0": Intercepts for each regression model
    """
    # For multinomial, we use one-vs-rest approach with binomial
    classes = np.unique(y)
    if len(classes) == 2:
        # Binary case, use logistic directly
        y_bin = (y == classes[1]).astype(float)
        return leave_one_out_logistic(X, y_bin, nit)

    # Multi-class case: use the first class vs rest for LOO
    # This is a simplification that works for feature selection
    y_bin = (y != classes[0]).astype(float)
    return leave_one_out_logistic(X, y_bin, nit)


# ------------------------------------------------------------------------------
# Leave-One-Out for Spline Regression
# ------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Compute B-spline basis functions for a given x.

    Args:
        x: Input array of shape (n,)
        knots: Knot positions sorted in increasing order
        degree: Degree of the spline (default: 3 for cubic spline)

    Returns:
        B-spline basis matrix of shape (n, len(knots) + degree - 1)
    """
    n = len(x)
    n_knots = len(knots)
    n_basis = n_knots + degree - 1

    # Augment knots with boundary knots
    total_len = 2 * degree + len(knots)
    augmented_knots = np.zeros(total_len, dtype=x.dtype)
    for i in range(degree):
        augmented_knots[i] = x[0]
    for i in range(len(knots)):
        augmented_knots[degree + i] = knots[i]
    for i in range(degree):
        augmented_knots[degree + len(knots) + i] = x[-1]

    # Initialize basis matrix
    basis = np.zeros((n, n_basis))

    # Compute degree 0 basis functions
    for i in range(n_basis):
        mask = (x >= augmented_knots[i]) & (x < augmented_knots[i + 1])
        if i == n_basis - 1:
            mask |= (x == augmented_knots[i + 1])
        basis[mask, i] = 1.0

    # Recursively compute higher degree basis functions
    for d in range(1, degree + 1):
        new_basis = np.zeros((n, n_basis - d))
        for i in range(n_basis - d):
            denom1 = augmented_knots[i + d] - augmented_knots[i]
            denom2 = augmented_knots[i + d + 1] - augmented_knots[i + 1]

            term1 = np.zeros_like(x)
            if denom1 > 1e-10:
                term1[:] = (x - augmented_knots[i]) / denom1 * basis[:, i]

            term2 = np.zeros_like(x)
            if denom2 > 1e-10:
                term2[:] = (augmented_knots[i + d + 1] - x) / denom2 * basis[:, i + 1]

            new_basis[:, i] = term1 + term2
        basis = new_basis

    return basis


@jit(nopython=True, cache=True)
def _weighted_least_squares(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Weighted least squares regression with Numba optimization.

    Args:
        X: Design matrix (n x k)
        y: Response vector (n,)
        w: Weights vector (n,), optional

    Returns:
        coeffs: Regression coefficients (k,)
        intercept: Intercept term
    """
    n, k = X.shape
    if w is None:
        w = np.ones(n)

    # Add intercept
    X_aug = np.hstack((np.ones((n, 1)), X))
    W = np.diag(w)

    # Solve (X'WX)^{-1} X'Wy
    XtWX = X_aug.T @ W @ X_aug
    XtWy = X_aug.T @ W @ y

    # Add small ridge for numerical stability
    XtWX += 1e-8 * np.eye(XtWX.shape[0])

    coeffs = np.linalg.solve(XtWX, XtWy)
    return coeffs[1:], coeffs[0]


@jit(nopython=True, parallel=True, cache=True)
def _loo_spline_numba(
    X: np.ndarray,
    y: np.ndarray,
    spline_df: int = 5,
    degree: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized parallel version of leave-one-out spline regression."""
    n, p = X.shape
    y = y.flatten()

    fit = np.zeros((n, p))
    beta = np.zeros(p)
    beta0 = np.zeros(p)

    for j in prange(p):
        x_j = X[:, j]

        # Create spline basis functions using quantile-based knots
        knots = np.quantile(x_j, np.linspace(0, 1, spline_df - degree + 2))[1:-1]
        knots = np.unique(knots)

        # Compute basis for all points
        sorted_idx = np.argsort(x_j)
        x_sorted = x_j[sorted_idx]
        y_sorted = y[sorted_idx]

        n_knots = len(knots)
        if n_knots >= 1:
            basis = _bspline_basis(x_sorted, knots, degree)
        else:
            # Too few unique points, use linear basis
            basis = x_sorted.reshape(-1, 1)

        # Center and scale for numerical stability
        n_basis = basis.shape[1]
        basis_mean = np.zeros(n_basis)
        basis_std = np.zeros(n_basis)
        for k in range(n_basis):
            basis_mean[k] = np.mean(basis[:, k])
            std = np.std(basis[:, k])
            basis_std[k] = std if std > 1e-10 else 1.0

        basis_normalized = np.zeros_like(basis)
        for k in range(n_basis):
            basis_normalized[:, k] = (basis[:, k] - basis_mean[k]) / basis_std[k]

        # Add intercept column
        X_aug = np.hstack((np.ones((n, 1)), basis_normalized))
        y_col = y_sorted.reshape(-1, 1)

        # Precompute X'X and X'y for fast LOO updates
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y_col

        # Add small ridge for numerical stability
        ridge = 1e-8 * np.eye(XtX.shape[0])
        XtX += ridge

        try:
            # Compute full fit coefficients
            coef_full = np.linalg.solve(XtX, Xty).flatten()
            intercept_full = coef_full[0]
            coef_j_full = coef_full[1:]

            # Store beta and beta0
            if len(coef_j_full) > 0:
                beta[j] = coef_j_full[0] * basis_std[0] if basis_std[0] > 1e-10 else coef_j_full[0]
            beta0[j] = intercept_full - np.sum(coef_j_full * basis_mean / basis_std)

            # Fast LOO using matrix inversion lemma (Sherman-Morrison)
            H = X_aug @ np.linalg.inv(XtX) @ X_aug.T
            y_pred_full = X_aug @ coef_full
            residual = y_sorted - y_pred_full
            loo_residual = residual / (1 - np.diag(H))
            loo_pred = y_sorted - loo_residual

            # Map back to original order
            for i_sorted in range(n):
                i_orig = sorted_idx[i_sorted]
                fit[i_orig, j] = loo_pred[i_sorted]

        except:
            # Fallback to mean prediction if matrix is singular
            mean_y = np.mean(y_sorted)
            for i in range(n):
                fit[i, j] = mean_y
            beta[j] = 0.0
            beta0[j] = mean_y

    return fit, beta, beta0


def leave_one_out_spline(
    X: np.ndarray,
    y: np.ndarray,
    spline_df: int = 5,
    degree: int = 3
) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate spline regression models.

    Args:
        X: n x p model matrix
        y: n-vector response
        spline_df: Degrees of freedom for the spline (number of basis functions)
        degree: Degree of the spline polynomial (default: 3 for cubic)

    Returns:
        A dictionary containing:
        - "fit": Prevalidated fit matrix (leave-one-out predictions)
        - "beta": Univariate regression coefficients for each column of X
        - "beta0": Intercepts for each regression model
    """
    fit, beta, beta0 = _loo_spline_numba(X, y, spline_df, degree)
    return {"fit": fit, "beta": beta, "beta0": beta0}


# ------------------------------------------------------------------------------
# Leave-One-Out for Tree Regression
# ------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _compute_mse(y: np.ndarray) -> float:
    """Compute mean squared error for regression."""
    if len(y) == 0:
        return 0.0
    mean_val = np.mean(y)
    return np.sum((y - mean_val) ** 2) / len(y)


@jit(nopython=True, cache=True)
def _find_best_split(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Find the best split point for a decision tree.

    Args:
        x: Feature values
        y: Target values

    Returns:
        best_threshold: Best split threshold
        best_mse: MSE at best split
        best_impurity: Impurity reduction
    """
    n = len(y)
    if n <= 1:
        return 0.0, _compute_mse(y), 0.0

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    parent_mse = _compute_mse(y)
    best_mse = parent_mse
    best_threshold = x_sorted[0] - 1.0

    # Try all possible split points between unique values
    for i in range(1, n):
        if x_sorted[i] > x_sorted[i-1] + 1e-10:
            threshold = (x_sorted[i] + x_sorted[i-1]) / 2

            # Split
            mask_left = x_sorted <= threshold
            mse_left = _compute_mse(y_sorted[mask_left])
            mse_right = _compute_mse(y_sorted[~mask_left])

            # Weighted average MSE
            n_left = np.sum(mask_left)
            n_right = n - n_left
            current_mse = (n_left * mse_left + n_right * mse_right) / n

            if current_mse < best_mse:
                best_mse = current_mse
                best_threshold = threshold

    impurity_reduction = parent_mse - best_mse
    return best_threshold, best_mse, impurity_reduction


class _DecisionTreeStump:
    """Simple decision tree stump (depth 1) for regression."""

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.threshold = None
        self.left_mean = None
        self.right_mean = None
        self.is_constant = False
        self.constant_value = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the tree stump."""
        if len(np.unique(x)) <= 1 or len(y) <= 1:
            self.is_constant = True
            self.constant_value = np.mean(y)
            return

        self.threshold, _, _ = _find_best_split(x, y)

        mask_left = x <= self.threshold
        if np.sum(mask_left) > 0:
            self.left_mean = np.mean(y[mask_left])
        else:
            self.left_mean = np.mean(y)

        if np.sum(~mask_left) > 0:
            self.right_mean = np.mean(y[~mask_left])
        else:
            self.right_mean = np.mean(y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the tree stump."""
        if self.is_constant:
            return np.full_like(x, self.constant_value)

        predictions = np.zeros_like(x)
        mask_left = x <= self.threshold
        predictions[mask_left] = self.left_mean
        predictions[~mask_left] = self.right_mean
        return predictions


@jit(nopython=True, parallel=True, cache=True)
def _loo_tree_numba(
    X: np.ndarray,
    y: np.ndarray,
    tree_max_depth: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized parallel version of leave-one-out decision tree regression."""
    n, p = X.shape
    y = y.flatten()

    fit = np.zeros((n, p))
    beta = np.zeros(p)
    beta0 = np.zeros(p)

    for j in prange(p):
        x_j = X[:, j]

        # Pre-sort x and y for faster split finding
        sorted_idx = np.argsort(x_j)
        x_sorted = x_j[sorted_idx]
        y_sorted = y[sorted_idx]

        # Find best split on full data
        best_threshold, _, _ = _find_best_split(x_sorted, y_sorted)

        # Precompute cumulative sums for fast LOO predictions
        cumsum_y = np.cumsum(y_sorted)
        cumsum_y_sq = np.cumsum(y_sorted ** 2)
        total_sum = cumsum_y[-1]
        total_count = n

        # LOO predictions using precomputed splits
        for i_orig in range(n):
            # Find position in sorted array
            i_sorted = 0
            while i_sorted < n and sorted_idx[i_sorted] != i_orig:
                i_sorted += 1

            # Compute means excluding the i-th point
            if i_sorted == 0:
                sum_left = 0.0
                count_left = 0
                sum_right = total_sum - y_sorted[i_sorted]
                count_right = total_count - 1
            elif i_sorted == n - 1:
                sum_left = cumsum_y[i_sorted - 1]
                count_left = i_sorted
                sum_right = 0.0
                count_right = 0
            else:
                sum_left = cumsum_y[i_sorted - 1]
                count_left = i_sorted
                sum_right = total_sum - cumsum_y[i_sorted]
                count_right = total_count - i_sorted - 1

            # Predict based on which side of the threshold the left-out point falls
            x_val = x_j[i_orig]
            if x_val <= best_threshold:
                if count_left > 0:
                    pred = sum_left / count_left
                else:
                    pred = sum_right / count_right if count_right > 0 else np.mean(y)
            else:
                if count_right > 0:
                    pred = sum_right / count_right
                else:
                    pred = sum_left / count_left if count_left > 0 else np.mean(y)

            fit[i_orig, j] = pred

        # Compute pseudo-coefficient (linear approximation)
        mask_valid = ~np.isnan(y)
        if np.sum(mask_valid) >= 2:
            x_valid = x_j[mask_valid]
            y_valid = y[mask_valid]
            cov = np.cov(x_valid, y_valid)[0, 1]
            var_x = np.var(x_valid)
            if var_x > 1e-10:
                beta[j] = cov / var_x
                beta0[j] = np.mean(y_valid) - beta[j] * np.mean(x_valid)
            else:
                beta[j] = 0.0
                beta0[j] = np.mean(y_valid)
        else:
            beta[j] = 0.0
            beta0[j] = np.mean(y)

    return fit, beta, beta0


def leave_one_out_tree(
    X: np.ndarray,
    y: np.ndarray,
    tree_max_depth: int = 2
) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate decision tree regression models.

    Args:
        X: n x p model matrix
        y: n-vector response
        tree_max_depth: Maximum depth of the decision tree (default: 2, shallow tree)

    Returns:
        A dictionary containing:
        - "fit": Prevalidated fit matrix (leave-one-out predictions)
        - "beta": Univariate regression coefficients for each column of X
        - "beta0": Intercepts for each regression model
    """
    fit, beta, beta0 = _loo_tree_numba(X, y, tree_max_depth)
    return {"fit": fit, "beta": beta, "beta0": beta0}


# ------------------------------------------------------------------------------
# Unified Dispatch Function
# ------------------------------------------------------------------------------

def fit_loo_univariate_models(
    X,
    y,
    family = "gaussian",
    nit = 2,
    univariate_model = "linear",
    spline_df = 5,
    spline_degree = 3,
    tree_max_depth = 2
) -> Dict[str, np.ndarray]:
    """
    Compute leave-one-out (LOO) predictions for univariate regression models.

    Args:
        X: NumPy array, n x p model matrix.
        y: NumPy array, n x 2 where first column corresponds to `time` and second column corresponds to `status`.
        family: String, distribution family for the regression model.
        nit: Number of iterations for the optimization.
        univariate_model: Type of univariate model to use: "linear", "spline", or "tree".
        spline_df: Degrees of freedom for spline regression (if univariate_model="spline").
        spline_degree: Degree of polynomial for spline regression (default: 3 for cubic).
        tree_max_depth: Maximum depth for decision tree regression (if univariate_model="tree").

    Returns:
        A dictionary containing:
        - "fit": Prevalidated fit matrix (leave-one-out predictions).
        - "beta": Univariate regression coefficients for each column of X.
        - "beta0": Intercept term (initialized to zeros).
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    elif not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Handle nonlinear models first (override family for LOO computation)
    if univariate_model == "spline":
        return leave_one_out_spline(X, y, spline_df=spline_df, degree=spline_degree)
    elif univariate_model == "tree":
        return leave_one_out_tree(X, y, tree_max_depth=tree_max_depth)
    elif univariate_model != "linear":
        raise ValueError(f"Invalid univariate_model: {univariate_model}. Supported: 'linear', 'spline', 'tree'.")

    # Linear model - use original family-based dispatch
    if family == "gaussian":
        return leave_one_out(X, y)
    elif family == "binomial":
        return leave_one_out_logistic(X, y, nit)
    elif family in {"multinomial", "poisson"}:
        # For multinomial and poisson: use Gaussian LOO as a simple approximation
        # This works well for feature transformation purposes
        result_gauss = leave_one_out(X, y)
        return result_gauss
    elif family == "cox":
        return leave_one_out_cox(X, y, nit)
    else:
        raise ValueError("Invalid family. Supported families: 'gaussian', 'binomial', 'multinomial', 'poisson', 'cox'.")
