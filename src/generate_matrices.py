"""Matrix data generator for Assignment 1 benchmarks.

Generates pairs of random n x n matrices as shared CSV files for use across
Python, Java, and C implementations. Also generates small validation matrices
(identity, zero, known values) and reference products for correctness checking.

Design:
  Uses Python's standard library (random, csv) exclusively. This guarantees
  100% reproducibility across any machine and operating system without
  relying on external C-extension binaries or DLL dependencies.

Seed:        42 (random.Random(42))
Value range: [0.0, 1.0) — 64-bit float
CSV format:  comma-separated, 6 decimal places
"""

import csv
import os
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [100, 250, 500, 750, 1000]
VALIDATION_SIZE = 4
SEED = 42


# ---------------------------------------------------------------------------
# Matrix helpers (standard library)
# ---------------------------------------------------------------------------
def write_csv_matrix(filepath: str, matrix: list[list[float]]) -> None:
    """Save a 2D float matrix to a CSV file formatted to 6 decimal places."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow([f"{val:.6f}" for val in row])


def read_csv_matrix(filepath: str) -> list[list[float]]:
    """Read a 2D float matrix from a CSV file."""
    matrix = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix.append([float(val) for val in row])
    return matrix


def reference_multiply(matrix_a: list[list[float]],
                       matrix_b: list[list[float]],
                       n: int) -> list[list[float]]:
    """Compute C = A x B in double precision for reference validation."""
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += matrix_a[i][k] * matrix_b[k][j]
            c[i][j] = s
    return c


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate_random_matrices(data_dir: str, rng: random.Random) -> None:
    """Generate random matrix pairs A and B for each benchmark size."""
    for n in SIZES:
        matrix_a = [[rng.random() for _ in range(n)] for _ in range(n)]
        matrix_b = [[rng.random() for _ in range(n)] for _ in range(n)]

        file_a = os.path.join(data_dir, f"A_{n}.csv")
        file_b = os.path.join(data_dir, f"B_{n}.csv")

        write_csv_matrix(file_a, matrix_a)
        write_csv_matrix(file_b, matrix_b)

        print(f"  Generated {n}x{n} matrices A and B.")


def generate_validation_matrices(data_dir: str) -> None:
    """Generate small matrices with known results for correctness testing.

    Creates:
      - identity_4.csv  : 4x4 identity matrix (A x I = A)
      - zero_4.csv      : 4x4 zero matrix (A x 0 = 0)
      - known_A_4.csv   : 4x4 matrix with fixed known values
      - known_B_4.csv   : 4x4 matrix with fixed known values
      - expected_C_4.csv: known_A x known_B calculated analytically
    """
    n = VALIDATION_SIZE

    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    zero = [[0.0] * n for _ in range(n)]

    known_a = [
        [1.0,  2.0,  3.0,  4.0],
        [5.0,  6.0,  7.0,  8.0],
        [9.0,  10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    known_b = [
        [2.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 3.0],
    ]
    expected_c = reference_multiply(known_a, known_b, n)

    write_csv_matrix(os.path.join(data_dir, "identity_4.csv"), identity)
    write_csv_matrix(os.path.join(data_dir, "zero_4.csv"), zero)
    write_csv_matrix(os.path.join(data_dir, "known_A_4.csv"), known_a)
    write_csv_matrix(os.path.join(data_dir, "known_B_4.csv"), known_b)
    write_csv_matrix(os.path.join(data_dir, "expected_C_4.csv"), expected_c)

    print(f"  Generated {n}x{n} validation matrices (identity, zero, known).")


def generate_reference_results(data_dir: str) -> None:
    """Generate reference multiplication results for the smallest benchmark sizes.

    Computes reference product from the exact rounded values in the CSV files.
    """
    ref_dir = os.path.join(data_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    # Generate reference for the first benchmark size (100) used in validation
    for n in [SIZES[0]]:
        file_a = os.path.join(data_dir, f"A_{n}.csv")
        file_b = os.path.join(data_dir, f"B_{n}.csv")
        matrix_a = read_csv_matrix(file_a)
        matrix_b = read_csv_matrix(file_b)
        product = reference_multiply(matrix_a, matrix_b, n)
        write_csv_matrix(os.path.join(ref_dir, f"C_{n}.csv"), product)
        print(f"  Generated {n}x{n} reference result C = A x B.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = random.Random(SEED)

    print("=== Generating Random Benchmark Matrices ===")
    generate_random_matrices(data_dir, rng)

    print("\n=== Generating Validation Matrices ===")
    generate_validation_matrices(data_dir)

    print("\n=== Generating Reference Results ===")
    generate_reference_results(data_dir)

    print(f"\nAll datasets generated successfully in: {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    main()