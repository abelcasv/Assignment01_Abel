import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;

public class MatrixBenchmark {

    private static final int[] SIZES = {100, 250, 500, 750, 1000};
    private static final int VALIDATION_SIZE = 4;
    private static final int NUM_REPETITIONS = 5;
    private static final int NUM_WARMUP_RUNS = 2;
    private static final double ABSOLUTE_TOLERANCE = 1e-4;

    private static volatile double resultSink;


    public static double[][] loadMatrix(String filepath, int n)
            throws IOException {
        double[][] matrix = new double[n][n];
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            for (int row = 0; row < n; row++) {
                String line = br.readLine();
                if (line == null) {
                    throw new IOException(
                            "Unexpected end of file at row " + row);
                }
                String[] values = line.split(",");
                for (int col = 0; col < n; col++) {
                    matrix[row][col] = Double.parseDouble(values[col]);
                }
            }
        }
        return matrix;
    }


    public static double[][] multiply(double[][] a, double[][] b, int n) {
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }
        return c;
    }


    private static boolean matricesAreClose(double[][] a, double[][] b,
                                            int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(a[i][j] - b[i][j]) > ABSOLUTE_TOLERANCE) {
                    return false;
                }
            }
        }
        return true;
    }


    private static boolean validateCorrectness(String dataDir)
            throws IOException {
        System.out.println("=== Correctness Validation ===");
        int n = VALIDATION_SIZE;
        boolean allPassed = true;

        double[][] a = loadMatrix(dataDir + "/known_A_4.csv", n);
        double[][] identity = loadMatrix(dataDir + "/identity_4.csv", n);
        double[][] result;

        // Test 1 — identity property
        result = multiply(a, identity, n);
        if (matricesAreClose(result, a, n)) {
            System.out.println("  [PASS] A x Identity = A");
        } else {
            System.out.println("  [FAIL] A x Identity != A");
            allPassed = false;
        }

        // Test 2 — zero property
        double[][] zero = loadMatrix(dataDir + "/zero_4.csv", n);
        result = multiply(a, zero, n);
        double[][] expectedZero = new double[n][n];
        if (matricesAreClose(result, expectedZero, n)) {
            System.out.println("  [PASS] A x Zero = Zero");
        } else {
            System.out.println("  [FAIL] A x Zero != Zero");
            allPassed = false;
        }

        // Test 3 — known result
        double[][] b = loadMatrix(dataDir + "/known_B_4.csv", n);
        double[][] expected = loadMatrix(dataDir + "/expected_C_4.csv", n);
        result = multiply(a, b, n);
        if (matricesAreClose(result, expected, n)) {
            System.out.println("  [PASS] Known A x B matches expected result");
        } else {
            System.out.println("  [FAIL] Known A x B does not match expected");
            allPassed = false;
        }

        // Test 4 — reference check for smallest benchmark size
        int smallest = SIZES[0];
        String refPath = dataDir + "/reference/C_" + smallest + ".csv";
        if (new File(refPath).exists()) {
            double[][] benchA = loadMatrix(
                    dataDir + "/A_" + smallest + ".csv", smallest);
            double[][] benchB = loadMatrix(
                    dataDir + "/B_" + smallest + ".csv", smallest);
            double[][] ref = loadMatrix(refPath, smallest);
            double[][] computed = multiply(benchA, benchB, smallest);
            if (matricesAreClose(computed, ref, smallest)) {
                System.out.printf("  [PASS] %dx%d matches numpy reference%n",
                        smallest, smallest);
            } else {
                System.out.printf("  [FAIL] %dx%d differs from numpy ref%n",
                        smallest, smallest);
                allPassed = false;
            }
        } else {
            System.out.println("  [SKIP] Reference file not found: " + refPath);
        }

        return allPassed;
    }


    public static void main(String[] args) {
        String baseDir = args.length > 0 ? args[0] : "../..";
        String dataDir = baseDir + "/data";
        String resultsDir = baseDir + "/results";

        new File(resultsDir).mkdirs();
        String resultsFile = resultsDir + "/java_results.csv";

        try {
            // Step 1: Validate correctness
            if (!validateCorrectness(dataDir)) {
                System.err.println(
                        "\n[ERROR] Correctness validation failed.");
                System.exit(1);
            }
            System.out.println("\nAll correctness checks passed.\n");

            // Step 2: Run benchmarks
            Runtime runtime = Runtime.getRuntime();

            try (PrintWriter writer =
                         new PrintWriter(new FileWriter(resultsFile))) {
                writer.println("language,n,repetition,time_s,memory_mb");

                for (int n : SIZES) {
                    System.out.println("Benchmarking n=" + n + "...");

                    String fileA = dataDir + "/A_" + n + ".csv";
                    String fileB = dataDir + "/B_" + n + ".csv";

                    if (!new File(fileA).exists()
                            || !new File(fileB).exists()) {
                        System.out.println("  [SKIP] Input files not found");
                        continue;
                    }

                    // Load data OUTSIDE the timed region
                    double[][] a = loadMatrix(fileA, n);
                    double[][] b = loadMatrix(fileB, n);

                    // Warm-up runs — trigger JIT (excluded from measurements)
                    for (int w = 0; w < NUM_WARMUP_RUNS; w++) {
                        double[][] warmup = multiply(a, b, n);
                        resultSink = warmup[0][0];
                    }

                    // Measured repetitions
                    for (int rep = 1; rep <= NUM_REPETITIONS; rep++) {
                        System.gc();
                        long heapBefore = runtime.totalMemory()
                                - runtime.freeMemory();

                        long startTime = System.nanoTime();
                        double[][] c = multiply(a, b, n);
                        long endTime = System.nanoTime();

                        // Keep c alive for heap measurement
                        resultSink = c[n - 1][n - 1];
                        long heapAfter = runtime.totalMemory()
                                - runtime.freeMemory();

                        double durationS =
                                (endTime - startTime) / 1_000_000_000.0;
                        double heapDeltaMB =
                                (heapAfter - heapBefore) / (1024.0 * 1024.0);

                        writer.printf(Locale.US, "java,%d,%d,%.6f,%.6f%n",
                                n, rep, durationS, heapDeltaMB);
                        writer.flush();

                        System.out.printf(Locale.US,
                                "  Rep %d/%d: %.4f s | %.2f MB (heap delta)%n",
                                rep, NUM_REPETITIONS, durationS, heapDeltaMB);
                    }
                }
            }

            System.out.println("\nResults saved to: "
                    + new File(resultsFile).getAbsolutePath());

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }
}