import csv
import os
import sys
import time
import tracemalloc


SIZES = [100, 250, 500, 750, 1000]
NUM_REPETITIONS = 5
NUM_WARMUP_RUNS = 1
VALIDATION_SIZE = 4
ABSOLUTE_TOLERANCE = 1e-4   


def load_matrix(filepath: str) -> list[list[float]]:
    matrix = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix.append([float(val) for val in row])
    return matrix


def multiply(matrix_a: list[list[float]],
             matrix_b: list[list[float]],
             n: int) -> list[list[float]]:
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = s
    return result


def matrices_are_close(matrix_a: list[list[float]],
                       matrix_b: list[list[float]],
                       n: int) -> bool:
    for i in range(n):
        for j in range(n):
            if abs(matrix_a[i][j] - matrix_b[i][j]) > ABSOLUTE_TOLERANCE:
                return False
    return True


def validate_correctness(data_dir: str) -> bool:
    print("=== Correctness Validation ===")
    n = VALIDATION_SIZE
    all_passed = True

    # Test 1 — identity property
    matrix_a = load_matrix(os.path.join(data_dir, "known_A_4.csv"))
    identity = load_matrix(os.path.join(data_dir, "identity_4.csv"))
    result = multiply(matrix_a, identity, n)
    if matrices_are_close(result, matrix_a, n):
        print("  [PASS] A x Identity = A")
    else:
        print("  [FAIL] A x Identity != A")
        all_passed = False

    # Test 2 — zero property
    zero = load_matrix(os.path.join(data_dir, "zero_4.csv"))
    result = multiply(matrix_a, zero, n)
    expected_zero = [[0.0] * n for _ in range(n)]
    if matrices_are_close(result, expected_zero, n):
        print("  [PASS] A x Zero = Zero")
    else:
        print("  [FAIL] A x Zero != Zero")
        all_passed = False

    # Test 3 — known result
    matrix_b = load_matrix(os.path.join(data_dir, "known_B_4.csv"))
    expected = load_matrix(os.path.join(data_dir, "expected_C_4.csv"))
    result = multiply(matrix_a, matrix_b, n)
    if matrices_are_close(result, expected, n):
        print("  [PASS] Known A x B matches expected result")
    else:
        print("  [FAIL] Known A x B does not match expected result")
        all_passed = False

    # Test 4 — reference check for smallest benchmark size
    smallest = SIZES[0]
    ref_path = os.path.join(data_dir, "reference", f"C_{smallest}.csv")
    if os.path.exists(ref_path):
        bench_a = load_matrix(os.path.join(data_dir, f"A_{smallest}.csv"))
        bench_b = load_matrix(os.path.join(data_dir, f"B_{smallest}.csv"))
        ref = load_matrix(ref_path)
        computed = multiply(bench_a, bench_b, smallest)
        if matrices_are_close(computed, ref, smallest):
            print(f"  [PASS] {smallest}x{smallest} matches numpy reference")
        else:
            print(f"  [FAIL] {smallest}x{smallest} differs from numpy reference")
            all_passed = False
    else:
        print(f"  [SKIP] Reference file not found: {ref_path}")

    return all_passed


def run_benchmark(data_dir: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "python_results.csv")

    with open(results_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["language", "n", "repetition", "time_s", "memory_mb"])

        for n in SIZES:
            file_a = os.path.join(data_dir, f"A_{n}.csv")
            file_b = os.path.join(data_dir, f"B_{n}.csv")

            if not os.path.exists(file_a) or not os.path.exists(file_b):
                print(f"\n  [SKIP] Input files not found for n={n}")
                continue

            print(f"\nBenchmarking n={n}  "
                  f"(pure Python — may be slow for large sizes)...")

            # Load data OUTSIDE the timed region
            matrix_a = load_matrix(file_a)
            matrix_b = load_matrix(file_b)

            # Warm-up runs (excluded from measurements)
            for _ in range(NUM_WARMUP_RUNS):
                multiply(matrix_a, matrix_b, n)

            # Measured repetitions
            for rep in range(1, NUM_REPETITIONS + 1):
                tracemalloc.start()
                start = time.perf_counter()

                result = multiply(matrix_a, matrix_b, n)

                elapsed_s = time.perf_counter() - start
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Consume the result to prevent any optimisation
                _ = result[0][0]

                peak_mb = peak_bytes / (1024 * 1024)

                writer.writerow(["python", n, rep,
                                 f"{elapsed_s:.6f}", f"{peak_mb:.6f}"])
                csvfile.flush()

                print(f"  Rep {rep}/{NUM_REPETITIONS}: "
                      f"{elapsed_s:.4f} s | {peak_mb:.2f} MB (tracemalloc)")

    print(f"\nResults saved to: {os.path.abspath(results_file)}")


def main() -> None:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..")
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")

    if not validate_correctness(data_dir):
        print("\n[ERROR] Correctness validation failed. "
              "Fix issues before benchmarking.")
        sys.exit(1)

    print("\nAll correctness checks passed.\n")
    run_benchmark(data_dir, results_dir)


if __name__ == "__main__":
    main()