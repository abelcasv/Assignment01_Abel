import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

LANGUAGE_FILES = {
    "python": "python_results.csv",
    "java":   "java_results.csv",
    "c":      "c_results.csv",
}

MEMORY_METRICS = {
    "python": "tracemalloc peak",
    "java":   "JVM heap delta",
    "c":      "process RSS",
}

LANGUAGE_COLORS = {
    "python": "#3776AB",
    "java":   "#F89820",
    "c":      "#A8B9CC",
}


def load_all_results() -> pd.DataFrame:
    """Load and concatenate all per-language CSV files."""
    frames = []
    for lang, filename in LANGUAGE_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            frames.append(df)
            print(f"  Loaded {len(df)} rows from {filename}")
        else:
            print(f"  [WARNING] {filename} not found — skipping {lang}")

    if not frames:
        print("[ERROR] No result files found in:", RESULTS_DIR)
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median, IQR for time and memory grouped by (language, n)."""
    summary = df.groupby(["language", "n"]).agg(
        median_time=("time_s", "median"),
        q1_time=("time_s", lambda x: x.quantile(0.25)),
        q3_time=("time_s", lambda x: x.quantile(0.75)),
        median_memory=("memory_mb", "median"),
        q1_memory=("memory_mb", lambda x: x.quantile(0.25)),
        q3_memory=("memory_mb", lambda x: x.quantile(0.75)),
        num_reps=("repetition", "count"),
    ).reset_index()

    summary["iqr_time"] = summary["q3_time"] - summary["q1_time"]
    summary["iqr_memory"] = summary["q3_memory"] - summary["q1_memory"]

    return summary


def plot_time_comparison(summary: pd.DataFrame) -> None:
    """Median execution time vs matrix size for all languages (log-y)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for lang in sorted(summary["language"].unique()):
        data = summary[summary["language"] == lang].sort_values("n")
        color = LANGUAGE_COLORS.get(lang, None)
        ax.errorbar(
            data["n"], data["median_time"],
            yerr=[data["median_time"] - data["q1_time"],
                  data["q3_time"] - data["median_time"]],
            marker="o", capsize=4, label=lang.capitalize(), color=color,
        )

    ax.set_xlabel("Matrix Size (n)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Matrix Multiplication — Execution Time vs Matrix Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "time_comparison.png"), dpi=150)
    plt.close()
    print("  Saved time_comparison.png")


def plot_memory_comparison(summary: pd.DataFrame) -> None:
    """Median memory vs matrix size — noting different metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for lang in sorted(summary["language"].unique()):
        data = summary[summary["language"] == lang].sort_values("n")
        metric = MEMORY_METRICS.get(lang, "unknown")
        color = LANGUAGE_COLORS.get(lang, None)
        ax.plot(data["n"], data["median_memory"],
                marker="s", label=f"{lang.capitalize()} ({metric})",
                color=color)

    ax.set_xlabel("Matrix Size (n)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Matrix Multiplication — Memory Usage vs Matrix Size\n"
                 "(Note: metrics differ across languages — see README)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "memory_comparison.png"), dpi=150)
    plt.close()
    print("  Saved memory_comparison.png")


def plot_cubic_growth(summary: pd.DataFrame) -> None:
    """Log-log plot with O(n^3) reference line for growth analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for lang in sorted(summary["language"].unique()):
        data = summary[summary["language"] == lang].sort_values("n")
        color = LANGUAGE_COLORS.get(lang, None)
        ax.plot(data["n"], data["median_time"],
                marker="o", label=lang.capitalize(), color=color)

    # O(n^3) reference — scaled to pass through the C data at smallest size
    c_data = summary[summary["language"] == "c"].sort_values("n")
    if not c_data.empty:
        n_ref = c_data["n"].iloc[0]
        t_ref = c_data["median_time"].iloc[0]
        ns = np.array(sorted(summary["n"].unique()))
        cubic_ref = t_ref * (ns / n_ref) ** 3
        ax.plot(ns, cubic_ref, "--", color="gray", alpha=0.6,
                linewidth=2, label="O(n³) reference")

    ax.set_xlabel("Matrix Size (n)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Time Growth Analysis (log-log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cubic_analysis.png"), dpi=150)
    plt.close()
    print("  Saved cubic_analysis.png")


def print_summary_table(summary: pd.DataFrame) -> None:
    """Pretty-print the summary statistics to the console."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE — Median (IQR)")
    print("=" * 80)

    for lang in sorted(summary["language"].unique()):
        metric = MEMORY_METRICS.get(lang, "unknown")
        print(f"\n--- {lang.upper()} (memory: {metric}) ---")
        data = summary[summary["language"] == lang].sort_values("n")
        for _, row in data.iterrows():
            print(f"  n={int(row['n']):>5d}:  time = {row['median_time']:.6f} s "
                  f"(IQR {row['iqr_time']:.6f} s)  |  "
                  f"memory = {row['median_memory']:.2f} MB "
                  f"(IQR {row['iqr_memory']:.2f} MB)  "
                  f"[{int(row['num_reps'])} reps]")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=== Loading Results ===")
    df = load_all_results()

    print("\n=== Computing Summary Statistics ===")
    summary = compute_summary(df)

    # Save summary CSV
    summary_file = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_file, index=False, float_format="%.6f")
    print(f"  Summary saved to: {summary_file}")

    print_summary_table(summary)

    print("\n=== Generating Plots ===")
    plot_time_comparison(summary)
    plot_memory_comparison(summary)
    plot_cubic_growth(summary)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
