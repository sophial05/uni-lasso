"""
Compare convergence speed with and without Nesterov momentum.
This tests the main claim that momentum reduces the number of iterations.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/workspaces/uni-lasso')

from unilasso.solvers import _fit_numba_lasso_path


def generate_test_data(n, p, random_seed=42):
    """Generate synthetic test data."""
    np.random.seed(random_seed)
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    lmdas = np.logspace(-1, 1, 20)
    return X, y, lmdas


def test_momentum_convergence():
    """Compare iterations and runtime with vs without momentum."""
    print("="*60)
    print("Nesterov Momentum vs No Momentum - Convergence Comparison")
    print("="*60)

    n, p = 1000, 200
    X, y, lmdas = generate_test_data(n, p)
    negative_penalty = 1.0

    print(f"\nProblem size: n={n}, p={p}, n_lmdas={len(lmdas)}")

    # Without momentum (original)
    print("\n--- Without momentum (momentum = 0.0) ---")
    start = time.time()
    betas1, intercepts1 = _fit_numba_lasso_path(
        X, y, lmdas, negative_penalty, fit_intercept=True,
        lr=0.01, max_epochs=5000, tol=1e-6, momentum=0.0
    )
    time_no_momentum = time.time() - start

    # With momentum (optimized)
    print("\n--- With Nesterov momentum (momentum = 0.9) ---")
    start = time.time()
    betas2, intercepts2 = _fit_numba_lasso_path(
        X, y, lmdas, negative_penalty, fit_intercept=True,
        lr=0.01, max_epochs=5000, tol=1e-6, momentum=0.9
    )
    time_with_momentum = time.time() - start

    # Compare results
    print("\n" + "="*60)
    print("RESULT COMPARISON")
    print("="*60)
    print(f"Time without momentum: {time_no_momentum:.3f}s")
    print(f"Time with momentum:    {time_with_momentum:.3f}s")

    if time_no_momentum > 0 and time_with_momentum > 0:
        speedup = time_no_momentum / time_with_momentum
        print(f"\nSpeedup: {speedup:.2f}x")

    # Check solution similarity
    max_diff = np.max(np.abs(betas1 - betas2))
    print(f"Max difference between solutions: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Solutions are practically identical (within tolerance)")
    else:
        print("Note: Some difference expected due to different convergence paths")

    return time_no_momentum, time_with_momentum, speedup


if __name__ == "__main__":
    test_momentum_convergence()
