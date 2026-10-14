# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

**Course:** Big Data · Academic Year 2026-2027  
**Degree:** Grado en Ciencia e Ingeniería de Datos  
**Institution:** Universidad de Las Palmas de Gran Canaria  
**Student:** Abel Castro Villar  

---

## 1. Research Objective & Hypothesis

### Research Question
> How does matrix size ($n$) affect runtime and memory use for the straightforward $O(n^3)$ triple-loop matrix multiplication algorithm implemented in Python, Java, and C?

### Hypothesis
All three implementations execute the same arithmetic operations ($2n^3$ FLOPs) and exhibit cubic execution time growth ($O(n^3)$). However, the absolute constant factors differ substantially:
- **Python (CPython 3.14):** Slowest due to bytecode interpretation, dynamic type checking, and boxed objects.
- **Java (OpenJDK 21 HotSpot):** Intermediate-to-fast due to just-in-time (JIT) compilation and aggressive runtime loop optimisations.
- **C (GCC 16.1 -O3):** Highly optimized compiled machine code with direct contiguous row-major memory access and no managed runtime overhead.

---

## 2. Experimental Environment

All experiments were conducted on the same physical host under controlled conditions:

| Parameter | Specification |
|---|---|
| **Operating System** | Windows 11 (64-bit) |
| **CPU Model** | 13th Gen Intel(R) Core(TM) i5-1335U |
| **Physical Cores / Threads** | 10 physical cores / 12 logical processors |
| **RAM** | 16.0 GB |
| **Python Version** | CPython 3.14.0 |
| **Java Version** | Oracle OpenJDK 21.0.12 (64-Bit Server VM, build 21.0.12+7-LTS-205) |
| **C Compiler & Flags** | GCC 16.1.0 (MSYS2 UCRT64), flags: `-Wall -Wextra -O3 -std=c11 -lpsapi` |
| **Random Seed** | 42 (Mersenne Twister standard generator) |
| **Numeric Type** | 64-bit floating point (`double` / IEEE 754 float64) |
| **Input Value Range** | $[0.0, 1.0)$ formatted to 6 decimal places (`%.6f`) |

---

## 3. Experimental Protocol

1. **Controlled Shared Inputs:** Identical matrices $A$ and $B$ were generated using a shared seed (42) and stored in CSV format in `data/`. Every language reads these exact files.
2. **Correctness Validation Before Benchmarking:**
   - $A \times I_4 = A$ (Identity property)
   - $A \times 0_4 = 0_4$ (Zero matrix property)
   - Known analytical test case ($4 \times 4$ fixed integer matrices)
   - Cross-language reference verification against reference products for $n=100$.
   - Numerical tolerance: absolute tolerance $\epsilon = 10^{-4}$ (accounting for CSV text roundtrip and floating point evaluation order).
3. **Timing Boundary:** **Kernel only.** Matrix reading, allocation, and output writing are performed outside the timed region.
   - Monotonic high-resolution timers: `time.perf_counter()` (Python), `System.nanoTime()` (Java), `QueryPerformanceCounter()` (C).
4. **Dead-Code Elimination Prevention:** Computation results are consumed via volatile sinks (`volatile` in C and Java, cell extraction in Python) to prevent compiler/JIT removal.
5. **Warm-up and Repetitions:**
   - Warm-up: 1 run (C, Python), 2 runs (Java, to trigger JIT compilation). Warm-ups are strictly excluded from reported statistics.
   - Repetitions: 5 recorded runs per configuration. Central metric: **Median**. Variability: **Interquartile Range (IQR)**.
6. **Time Budget & Limits:** Declared runtime budget of 120 seconds per run. Python exceeded the budget at $n=750$ (>300 s/rep) and was safely truncated at its largest feasible size ($n=500$). C and Java completed all configurations up to $n=1000$.

---

## 4. Empirical Results

### Summary Table — Median Runtime and Memory (IQR) across 5 Repetitions

| Language | $n$ | Median Time (s) | Time IQR (s) | Median Memory (MB) | Memory Metric | Reps | Status |
|---|---|---|---|---|---|---|---|
| **C** | 100 | **0.005871** | 0.000018 | 4.51 | Process RSS | 5 | Completed |
| **C** | 250 | **0.091229** | 0.005744 | 5.83 | Process RSS | 5 | Completed |
| **C** | 500 | **0.772579** | 0.021614 | 10.14 | Process RSS | 5 | Completed |
| **C** | 750 | **3.224082** | 1.213902 | 17.29 | Process RSS | 5 | Completed |
| **C** | 1000 | **7.126446** | 0.028500 | 27.29 | Process RSS | 5 | Completed |
| **Java** | 100 | **0.001974** | 0.000081 | 0.09 | JVM Heap Delta | 5 | Completed |
| **Java** | 250 | **0.027626** | 0.001548 | 0.51 | JVM Heap Delta | 5 | Completed |
| **Java** | 500 | **0.343741** | 0.002203 | 1.97 | JVM Heap Delta | 5 | Completed |
| **Java** | 750 | **1.466174** | 0.009132 | 4.36 | JVM Heap Delta | 5 | Completed |
| **Java** | 1000 | **3.329269** | 0.061847 | 8.00 | JVM Heap Delta | 5 | Completed |
| **Python** | 100 | **0.532843** | 0.075128 | 0.30 | tracemalloc Peak | 5 | Completed |
| **Python** | 250 | **11.084877** | 3.625842 | 1.92 | tracemalloc Peak | 5 | Completed |
| **Python** | 500 | **99.940090** | 0.864411 | 7.65 | tracemalloc Peak | 5 | Largest Feasible |
| **Python** | 750 | — | — | — | — | 0 | *Timeout (>300 s budget)* |
| **Python** | 1000 | — | — | — | — | 0 | *Budget Exceeded* |

*Note on Memory Metrics: The memory quantities measure different runtime dimensions and must not be compared directly as identical. C measures whole-process Working Set (RSS); Java measures JVM heap difference before/after multiplication; Python measures Python-managed heap allocations via tracemalloc.*

> **For the full scientific analysis, scalability plots, and theoretical discussion, see the research report:**  
> 📄 [Assignment01_Report.md](Assignment01_Report.md)

---

## 5. How to Reproduce

### Step 1: Generate Datasets
```powershell
py src/generate_matrices.py
```

### Step 2: Run Python Benchmark
```powershell
py src/python/benchmark.py
```

### Step 3: Compile and Run Java Benchmark
```powershell
javac src/java/MatrixBenchmark.java
java -cp src/java MatrixBenchmark .
```

### Step 4: Compile and Run C Benchmark
```powershell
gcc -Wall -Wextra -O3 -std=c11 src/c/benchmark.c -o src/c/benchmark.exe -lpsapi
./src/c/benchmark.exe .
```

### Step 5: Generate Summary Statistics and Plots
```powershell
py src/analysis/analyze_results.py
```

Outputs will be saved to `results/summary.csv` and `results/plots/`.