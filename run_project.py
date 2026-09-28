"""
run_project.py
===============
Master orchestration script for the Smart Attendance Anomaly Detector.

Running this script executes the complete pipeline end-to-end:

    python run_project.py

Steps
-----
1. Generate dataset (if data/attendance_dataset.csv does not exist).
2. Train all models and run hyperparameter experiments.
3. Report final results summary.

After running this script, start the web app:

    streamlit run app.py
"""

import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║     SMART ATTENDANCE ANOMALY DETECTOR                        ║
║     Complete Academic AI Project Pipeline                    ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_step(description: str, command: list, cwd: str = ROOT) -> bool:
    """Run a subprocess step and report success/failure."""
    print(f"\n{'─'*60}")
    print(f"  ▶  {description}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(command, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓  Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗  Failed (exit code {result.returncode})")
        return False


def main():
    print(BANNER)

    # ── Step 1: Generate dataset ────────────────────────────────────────
    data_csv = os.path.join(ROOT, "data", "attendance_dataset.csv")
    if not os.path.exists(data_csv):
        ok = run_step(
            "Generating synthetic attendance dataset …",
            [sys.executable, os.path.join(ROOT, "data", "generate_dataset.py")],
        )
        if not ok:
            print("\nDataset generation failed. Aborting.")
            sys.exit(1)
    else:
        print(f"\n  Dataset already exists: {data_csv}")
        print("  Skipping generation. Delete the file to regenerate.\n")

    # ── Step 2: Train models & run experiments ──────────────────────────
    ok = run_step(
        "Training models, hyperparameter search, evaluation, plots …",
        [sys.executable, "-m", "src.train"],
    )
    if not ok:
        print("\nTraining pipeline failed.")
        sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    results_csv = os.path.join(ROOT, "results", "model_comparison.csv")
    if os.path.exists(results_csv):
        import pandas as pd
        df = pd.read_csv(results_csv, index_col=0)
        print("\n  Final Model Comparison (Test Set):\n")
        print(df.to_string())

    report_path = os.path.join(ROOT, "results", "final_report.md")
    plots_dir   = os.path.join(ROOT, "results", "plots")
    n_plots     = len([f for f in os.listdir(plots_dir) if f.endswith(".png")]) \
                  if os.path.isdir(plots_dir) else 0

    print(f"\n  Results saved in  : results/")
    print(f"  Report saved in   : {report_path}")
    print(f"  Plots generated   : {n_plots} PNG files in results/plots/")
    print(f"\n  To launch the web app:")
    print(f"    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
