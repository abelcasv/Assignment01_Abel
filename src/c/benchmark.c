#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <windows.h>
#include <psapi.h>


static const int SIZES[] = {100, 250, 500, 750, 1000};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
static const int VALIDATION_SIZE = 4;
static const int NUM_REPETITIONS = 5;
static const int NUM_WARMUP_RUNS = 1;
static const double ABSOLUTE_TOLERANCE = 1e-4;

static volatile double result_sink;


static double* load_matrix(const char* filepath, int n) {
    double* matrix = (double*)malloc((size_t)n * n * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "Error: allocation failed for %dx%d matrix\n", n, n);
        exit(EXIT_FAILURE);
    }

    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error: cannot open file: %s\n", filepath);
        free(matrix);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == n - 1) {
                fscanf(file, "%lf", &matrix[i * n + j]);
            } else {
                fscanf(file, "%lf,", &matrix[i * n + j]);
            }
        }
    }
    fclose(file);
    return matrix;
}


static void multiply(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}


static int matrices_are_close(const double* a, const double* b, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(a[i] - b[i]) > ABSOLUTE_TOLERANCE) {
            return 0;
        }
    }
    return 1;
}


static int validate_correctness(const char* data_dir) {
    printf("=== Correctness Validation ===\n");
    int all_passed = 1;
    int n = VALIDATION_SIZE;
    char path[512];

    /* Load validation matrices */
    snprintf(path, sizeof(path), "%s/known_A_4.csv", data_dir);
    double* known_a = load_matrix(path, n);

    snprintf(path, sizeof(path), "%s/identity_4.csv", data_dir);
    double* identity = load_matrix(path, n);

    snprintf(path, sizeof(path), "%s/zero_4.csv", data_dir);
    double* zero_mat = load_matrix(path, n);

    snprintf(path, sizeof(path), "%s/known_B_4.csv", data_dir);
    double* known_b = load_matrix(path, n);

    snprintf(path, sizeof(path), "%s/expected_C_4.csv", data_dir);
    double* expected_c = load_matrix(path, n);

    double* result = (double*)malloc((size_t)n * n * sizeof(double));
    double* zero_expected = (double*)calloc((size_t)n * n, sizeof(double));

    /* Test 1 — identity property */
    multiply(known_a, identity, result, n);
    if (matrices_are_close(result, known_a, n)) {
        printf("  [PASS] A x Identity = A\n");
    } else {
        printf("  [FAIL] A x Identity != A\n");
        all_passed = 0;
    }

    /* Test 2 — zero property */
    multiply(known_a, zero_mat, result, n);
    if (matrices_are_close(result, zero_expected, n)) {
        printf("  [PASS] A x Zero = Zero\n");
    } else {
        printf("  [FAIL] A x Zero != Zero\n");
        all_passed = 0;
    }

    /* Test 3 — known result */
    multiply(known_a, known_b, result, n);
    if (matrices_are_close(result, expected_c, n)) {
        printf("  [PASS] Known A x B matches expected result\n");
    } else {
        printf("  [FAIL] Known A x B does not match expected result\n");
        all_passed = 0;
    }

    free(known_a);
    free(identity);
    free(zero_mat);
    free(known_b);
    free(expected_c);
    free(result);
    free(zero_expected);

    /* Test 4 — reference check for smallest benchmark size */
    int smallest = SIZES[0];
    snprintf(path, sizeof(path), "%s/reference/C_%d.csv", data_dir, smallest);
    FILE* ref_file = fopen(path, "r");
    if (ref_file) {
        fclose(ref_file);

        char path_a[512], path_b[512];
        snprintf(path_a, sizeof(path_a), "%s/A_%d.csv", data_dir, smallest);
        snprintf(path_b, sizeof(path_b), "%s/B_%d.csv", data_dir, smallest);

        double* bench_a = load_matrix(path_a, smallest);
        double* bench_b = load_matrix(path_b, smallest);
        double* ref = load_matrix(path, smallest);
        double* computed = (double*)malloc(
                (size_t)smallest * smallest * sizeof(double));

        multiply(bench_a, bench_b, computed, smallest);
        if (matrices_are_close(computed, ref, smallest)) {
            printf("  [PASS] %dx%d matches numpy reference\n",
                   smallest, smallest);
        } else {
            printf("  [FAIL] %dx%d differs from numpy reference\n",
                   smallest, smallest);
            all_passed = 0;
        }

        free(bench_a);
        free(bench_b);
        free(ref);
        free(computed);
    } else {
        printf("  [SKIP] Reference file not found: %s\n", path);
    }

    return all_passed;
}


static double get_working_set_mb(void) {
    PROCESS_MEMORY_COUNTERS mem_info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &mem_info,
                             sizeof(mem_info))) {
        return mem_info.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}


int main(int argc, char* argv[]) {
    const char* base_dir = (argc > 1) ? argv[1] : "../..";

    char data_dir[512], results_dir[512], results_path[512];
    snprintf(data_dir, sizeof(data_dir), "%s/data", base_dir);
    snprintf(results_dir, sizeof(results_dir), "%s/results", base_dir);
    snprintf(results_path, sizeof(results_path),
             "%s/c_results.csv", results_dir);

    /* Ensure results directory exists */
    _mkdir(results_dir);

    /* Step 1: Validate correctness */
    if (!validate_correctness(data_dir)) {
        fprintf(stderr,
                "\n[ERROR] Correctness validation failed.\n");
        return EXIT_FAILURE;
    }
    printf("\nAll correctness checks passed.\n\n");

    /* Step 2: Run benchmarks */
    FILE* writer = fopen(results_path, "w");
    if (!writer) {
        fprintf(stderr, "Error: cannot create results file: %s\n",
                results_path);
        return EXIT_FAILURE;
    }
    fprintf(writer, "language,n,repetition,time_s,memory_mb\n");

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    for (int s = 0; s < NUM_SIZES; s++) {
        int n = SIZES[s];
        printf("Benchmarking n=%d...\n", n);

        char file_a[512], file_b[512];
        snprintf(file_a, sizeof(file_a), "%s/A_%d.csv", data_dir, n);
        snprintf(file_b, sizeof(file_b), "%s/B_%d.csv", data_dir, n);

        /* Load data OUTSIDE the timed region */
        double* a = load_matrix(file_a, n);
        double* b = load_matrix(file_b, n);
        double* c = (double*)malloc((size_t)n * n * sizeof(double));

        if (!c) {
            fprintf(stderr, "Error: allocation failed for n=%d\n", n);
            free(a);
            free(b);
            continue;
        }

        /* Warm-up run (excluded from measurements) */
        for (int w = 0; w < NUM_WARMUP_RUNS; w++) {
            multiply(a, b, c, n);
            result_sink = c[0];
        }

        /* Measured repetitions */
        for (int rep = 1; rep <= NUM_REPETITIONS; rep++) {
            LARGE_INTEGER start_time, end_time;

            QueryPerformanceCounter(&start_time);
            multiply(a, b, c, n);
            QueryPerformanceCounter(&end_time);

            /* Consume result to prevent dead-code elimination */
            result_sink = c[0];

            double duration_s = (double)(end_time.QuadPart
                    - start_time.QuadPart) / frequency.QuadPart;
            double rss_mb = get_working_set_mb();

            fprintf(writer, "c,%d,%d,%.6f,%.6f\n",
                    n, rep, duration_s, rss_mb);
            fflush(writer);

            printf("  Rep %d/%d: %.4f s | %.2f MB (RSS)\n",
                   rep, NUM_REPETITIONS, duration_s, rss_mb);
        }

        free(a);
        free(b);
        free(c);
    }

    fclose(writer);
    printf("\nResults saved to: %s\n", results_path);

    return EXIT_SUCCESS;
}