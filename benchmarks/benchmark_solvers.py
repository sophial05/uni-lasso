"""
Benchmark script to compare PyTorch and Numba solver performance.

This script measures the speedup of the Numba backend compared to the
original PyTorch backend across different dataset sizes.
"""

import sys
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, '/workspaces/uni-lasso')


def generate_dataset(n: int, p: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset."""
    np.random.seed(seed)
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    # Make first 10 features non-zero
    beta_true[:10] = np.random.randn(10) * 2
    y = X @ beta_true + np.random.randn(n) * 0.5
    return X, y


def benchmark_solver(
    X: np.ndarray,
    y: np.ndarray,
    backend: str,
    n_lmdas: int = 50,
    n_runs: int = 3
) -> Dict:
    """Benchmark a solver backend."""
    from unilasso.uni_lasso import fit_uni

    times = []
    results = []

    # Warm-up run (especially important for Numba JIT)
    if backend == "numba":
        print(f"  Warm-up run for {backend}...")
        _ = fit_uni(X, y, n_lmdas=10, backend=backend, verbose=False)

    for i in range(n_runs):
        start = time.time()
        result = fit_uni(X, y, n_lmdas=n_lmdas, backend=backend, verbose=False)
        end = time.time()
        times.append(end - start)
        results.append(result)

    return {
        "backend": backend,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "times": times,
        "result": results[-1]  # Keep last result for validation
    }


def run_benchmarks(dataset_sizes: List[Tuple[int, int]]) -> Dict:
    """Run benchmarks across multiple dataset sizes."""
    results = {}

    for n, p in dataset_sizes:
        print(f"\n{'='*60}")
        print(f"Dataset size: n={n}, p={p}")
        print(f"{'='*60}")

        X, y = generate_dataset(n, p)

        # Benchmark PyTorch
        print("\nBenchmarking PyTorch backend...")
        pytorch_result = benchmark_solver(X, y, backend="pytorch")
        print(f"  Mean time: {pytorch_result['mean_time']:.4f}s "
              f"(±{pytorch_result['std_time']:.4f}s)")

        # Benchmark Numba
        print("\nBenchmarking Numba backend...")
        numba_result = benchmark_solver(X, y, backend="numba")
        print(f"  Mean time: {numba_result['mean_time']:.4f}s "
              f"(±{numba_result['std_time']:.4f}s)")

        # Calculate speedup
        speedup = pytorch_result['mean_time'] / numba_result['mean_time']
        print(f"\n  Numba speedup: {speedup:.2f}x")

        # Validate results are the same
        pytorch_coefs = pytorch_result['result'].coefs
        numba_coefs = numba_result['result'].coefs
        max_diff = np.max(np.abs(pytorch_coefs - numba_coefs))
        print(f"  Max coefficient difference: {max_diff:.6f}")

        results[(n, p)] = {
            "pytorch": pytorch_result,
            "numba": numba_result,
            "speedup": speedup,
            "max_diff": max_diff
        }

    return results


def print_summary(results: Dict):
    """Print a summary of benchmark results."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Dataset':<20} {'PyTorch (s)':<12} {'Numba (s)':<12} {'Speedup':<10} {'Max Diff':<10}")
    print("-"*80)

    for (n, p), data in results.items():
        pt_time = data['pytorch']['mean_time']
        nb_time = data['numba']['mean_time']
        speedup = data['speedup']
        max_diff = data['max_diff']
        print(f"n={n:<5}, p={p:<5} {pt_time:<12.4f} {nb_time:<12.4f} {speedup:<10.2f} {max_diff:<10.6f}")

    print("="*80)


def plot_results(results: Dict, save_path: str = None):
    """Plot benchmark results."""
    sizes = list(results.keys())
    n_sizes = len(sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot times
    size_labels = [f"n={n}\np={p}" for n, p in sizes]
    x = np.arange(n_sizes)
    width = 0.35

    pt_times = [results[s]['pytorch']['mean_time'] for s in sizes]
    nb_times = [results[s]['numba']['mean_time'] for s in sizes]

    ax1.bar(x - width/2, pt_times, width, label='PyTorch', color='royalblue')
    ax1.bar(x + width/2, nb_times, width, label='Numba', color='orange')
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Solver Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot speedup
    speedups = [results[s]['speedup'] for s in sizes]
    ax2.bar(x, speedups, color='forestgreen', alpha=0.7)
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Speedup (PyTorch / Numba)')
    ax2.set_title('Numba Speedup Over PyTorch')
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()


def main():
    """Run the full benchmark suite."""
    print("="*80)
    print("UniLasso Solver Benchmark")
    print("="*80)

    # Define dataset sizes to test
    dataset_sizes = [
        (100, 10),
        (100, 50),
        (500, 50),
        (1000, 100),
    ]

    # Run benchmarks
    results = run_benchmarks(dataset_sizes)

    # Print summary
    print_summary(results)

    # Plot results
    plot_results(results, save_path='/workspaces/uni-lasso/benchmarks/benchmark_plot.png')

    return results


if __name__ == "__main__":
    results = main()
