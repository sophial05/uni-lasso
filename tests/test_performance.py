"""
Performance Benchmark Tests for GroupAdaUniLasso

This script measures actual runtime performance before and after optimizations
to verify the claimed speedups.
"""

import sys
import time
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, '/workspaces/uni-lasso')

from unilasso.uni_lasso import fit_uni


def generate_data(n, p, random_seed=42):
    """Generate synthetic regression data with sparse true coefficients."""
    np.random.seed(random_seed)
    X = np.random.randn(n, p)

    # Sparse true coefficients: only 5% of features are relevant
    true_beta = np.zeros(p)
    n_nonzero = int(0.05 * p)
    nonzero_indices = np.random.choice(p, n_nonzero, replace=False)
    true_beta[nonzero_indices] = np.random.randn(n_nonzero) * 3

    # Generate response
    y = X @ true_beta + np.random.randn(n) * 0.5

    return X, y, true_beta


def generate_correlated_data(n, p, correlation=0.8, random_seed=42):
    """Generate synthetic data with highly correlated features (for group constraint testing)."""
    np.random.seed(random_seed)

    # Create covariance matrix with block correlation structure
    cov = np.eye(p)
    block_size = 10
    for i in range(0, p, block_size):
        end = min(i + block_size, p)
        for j in range(i, end):
            for k in range(i, end):
                if j != k:
                    cov[j, k] = correlation

    # Generate multivariate normal data
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, cov, size=n)

    # Sparse true coefficients - one per group
    true_beta = np.zeros(p)
    for i in range(0, p, block_size):
        true_beta[i] = np.random.randn() * 2

    y = X @ true_beta + np.random.randn(n) * 0.3

    return X, y, true_beta


def benchmark_one_case(name, X, y, **kwargs):
    """Run benchmark for one case and return elapsed time."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Dimensions: n={X.shape[0]}, p={X.shape[1]}")
    print(f"Parameters: {kwargs}")
    print(f"{'-'*60}")

    # Warm-up run (compiles Numba functions)
    print("Warm-up run (compiling Numba)...")
    warmup_start = time.time()
    _ = fit_uni(X, y, **kwargs)
    warmup_time = time.time() - warmup_start
    print(f"Warm-up done in {warmup_time:.3f}s")

    # Actual timed run
    print("Running timed benchmark...")
    start_time = time.time()
    result = fit_uni(X, y, **kwargs)
    elapsed_time = time.time() - start_time

    print(f"Total time: {elapsed_time:.3f}s")
    print(f"Number of non-zero coefficients (lambda_max): {np.sum(result.coefs[..., 0] != 0)}")
    n_nonzero_total = np.sum(np.any(result.coefs != 0, axis=0))
    print(f"Total features selected across path: {n_nonzero_total}")

    return {
        'name': name,
        'n': X.shape[0],
        'p': X.shape[1],
        'params': str(kwargs),
        'time_seconds': elapsed_time,
        'warmup_seconds': warmup_time,
        'n_nonzero': np.sum(result.coefs[..., 0] != 0),
        'n_nonzero_total': n_nonzero_total,
        'result': result
    }


def run_all_benchmarks():
    """Run all benchmark cases."""
    results = []

    # Case 1: Small dataset, linear, Gaussian
    print("\n" + "="*60)
    print("CASE 1: Small dataset (n=100, p=20) - Linear Gaussian")
    print("="*60)
    n, p = 100, 20
    X, y, _ = generate_data(n, p)
    res = benchmark_one_case(
        "Small-Linear-Gaussian", X, y,
        family="gaussian",
        univariate_model="linear",
        enable_group_constraint=False,
        momentum=0.9,  # With momentum (optimized)
        verbose=False
    )
    results.append(res)

    # Case 2: Small dataset with momentum off (for comparison)
    print("\n" + "="*60)
    print("CASE 1b: Small dataset (n=100, p=20) - Linear Gaussian (no momentum)")
    print("="*60)
    res_no_momentum = benchmark_one_case(
        "Small-Linear-Gaussian-no-momentum", X, y,
        family="gaussian",
        univariate_model="linear",
        enable_group_constraint=False,
        momentum=0.0,  # Without momentum (original)
        verbose=False
    )
    results.append(res_no_momentum)

    # Case 3: Medium dataset, linear Gaussian
    print("\n" + "="*60)
    print("CASE 2: Medium dataset (n=1000, p=200) - Linear Gaussian")
    print("="*60)
    n, p = 1000, 200
    X, y, _ = generate_data(n, p)
    res = benchmark_one_case(
        "Medium-Linear-Gaussian", X, y,
        family="gaussian",
        univariate_model="linear",
        enable_group_constraint=False,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Case 4: Medium dataset with group constraint (correlated features)
    print("\n" + "="*60)
    print("CASE 3: Medium dataset (n=1000, p=200) - With Group Constraint (correlated)")
    print("="*60)
    n, p = 1000, 200
    X, y, _ = generate_correlated_data(n, p)
    res = benchmark_one_case(
        "Medium-GroupConstraint", X, y,
        family="gaussian",
        univariate_model="linear",
        enable_group_constraint=True,
        corr_threshold=0.7,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Case 5: Medium dataset, Poisson GLM
    print("\n" + "="*60)
    print("CASE 4: Medium dataset (n=1000, p=200) - Poisson GLM")
    print("="*60)
    n, p = 1000, 200
    X, y, _ = generate_data(n, p)
    # Make y non-negative for Poisson
    y = np.exp(y / 5)  # Scale to reasonable counts
    y = np.ceil(y).astype(int)
    res = benchmark_one_case(
        "Medium-Poisson", X, y,
        family="poisson",
        univariate_model="linear",
        enable_group_constraint=False,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Case 6: Medium dataset, Spline nonlinear
    print("\n" + "="*60)
    print("CASE 5: Medium dataset (n=1000, p=200) - Spline Nonlinear")
    print("="*60)
    n, p = 1000, 200
    X, y, _ = generate_data(n, p)
    res = benchmark_one_case(
        "Medium-Spline-Nonlinear", X, y,
        family="gaussian",
        univariate_model="spline",
        enable_group_constraint=False,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Case 7: Medium dataset, Tree nonlinear
    print("\n" + "="*60)
    print("CASE 6: Medium dataset (n=1000, p=200) - Tree Nonlinear")
    print("="*60)
    n, p = 1000, 200
    X, y, _ = generate_data(n, p)
    res = benchmark_one_case(
        "Medium-Tree-Nonlinear", X, y,
        family="gaussian",
        univariate_model="tree",
        enable_group_constraint=False,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Case 8: Large dataset (n=5000, p=1000)
    print("\n" + "="*60)
    print("CASE 7: Large dataset (n=5000, p=1000) - Linear Gaussian")
    print("="*60)
    n, p = 5000, 1000
    X, y, _ = generate_data(n, p)
    res = benchmark_one_case(
        "Large-Linear-Gaussian", X, y,
        family="gaussian",
        univariate_model="linear",
        enable_group_constraint=False,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Case 9: High-dimensional dataset (n=1000, p=5000) - smaller than 10k for testing
    print("\n" + "="*60)
    print("CASE 8: High-dimensional (n=1000, p=5000) - With Group Constraint")
    print("="*60)
    n, p = 1000, 5000
    X, y, _ = generate_correlated_data(n, p)
    res = benchmark_one_case(
        "HighDim-GroupConstraint", X, y,
        family="gaussian",
        univariate_model="linear",
        enable_group_constraint=True,
        corr_threshold=0.7,
        momentum=0.9,
        verbose=False
    )
    results.append(res)

    # Summary table
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    summary = pd.DataFrame([
        {
            'Name': r['name'],
            'n': r['n'],
            'p': r['p'],
            'Time(s)': f"{r['time_seconds']:.3f}",
            'Non-zero': r['n_nonzero'],
            'Total selected': r.get('n_nonzero_total', r['n_nonzero'])
        }
        for r in results
    ])
    print(summary.to_string(index=False))

    # Compare momentum vs no momentum
    if len(results) >= 2:
        time_with_momentum = results[0]['time_seconds']
        time_no_momentum = results[1]['time_seconds']
        if time_with_momentum > 0:
            speedup = time_no_momentum / time_with_momentum
            print(f"\nSpeedup from Nesterov momentum: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    print("Starting GroupAdaUniLasso Performance Benchmark...")
    print(f"Date: {time.ctime()}")
    print(f"NumPy version: {np.__version__}")

    start_total = time.time()
    results = run_all_benchmarks()
    total_elapsed = time.time() - start_total

    print(f"\nTotal benchmark runtime: {total_elapsed:.3f}s")
