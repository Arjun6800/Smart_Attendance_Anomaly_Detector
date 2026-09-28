"""
src/visualization.py
=====================
All plotting functions for the Smart Attendance Anomaly Detector.

Each function saves its plot to ``results/plots/`` and optionally
returns the matplotlib Figure object (for embedding in Streamlit).

Design Principles
-----------------
- Every plot has a title, axis labels, and a legend where applicable.
- Seaborn is used for statistical plots.
- Matplotlib is used for custom layouts.
- The style is consistent across all plots (dark background, readable fonts).
- No decorative / meaningless plots are included.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

# ─────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PLOT_DPI    = 120
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "normal":  "#4CAF50",
    "anomaly": "#F44336",
    "if":      "#2196F3",
    "lof":     "#FF9800",
    "hybrid":  "#9C27B0",
}


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 1. Attendance Hour Distribution
# ─────────────────────────────────────────────

def plot_attendance_distribution(df: pd.DataFrame) -> str:
    """Plot histogram of check-in hours split by normal/anomaly."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hour distribution
    ax = axes[0]
    for label, color, name in [(0, COLORS["normal"], "Normal"),
                                (1, COLORS["anomaly"], "Anomaly")]:
        subset = df[df["is_anomaly"] == label]["hour"]
        ax.hist(subset, bins=24, alpha=0.65, color=color, label=name, edgecolor="white")
    ax.set_title("Attendance Check-in Hour Distribution")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax.legend()

    # Weekly attendance rate distribution
    ax2 = axes[1]
    for label, color, name in [(0, COLORS["normal"], "Normal"),
                                (1, COLORS["anomaly"], "Anomaly")]:
        subset = df[df["is_anomaly"] == label]["weekly_attendance_rate"]
        ax2.hist(subset, bins=30, alpha=0.65, color=color, label=name, edgecolor="white")
    ax2.set_title("Weekly Attendance Rate Distribution")
    ax2.set_xlabel("Weekly Attendance Rate")
    ax2.set_ylabel("Count")
    ax2.legend()

    fig.suptitle("Dataset Feature Distributions: Normal vs Anomaly", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "attendance_distribution.png")


# ─────────────────────────────────────────────
# 2. Anomaly Type Breakdown
# ─────────────────────────────────────────────

def plot_anomaly_type_breakdown(df: pd.DataFrame) -> str:
    """Bar chart showing counts of each injected anomaly type."""
    if "anomaly_type" not in df.columns:
        return ""
    counts = df["anomaly_type"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("Set2", len(counts)), edgecolor="white")
    ax.set_title("Anomaly Type Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.set_xlabel("Anomaly Type")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return _save(fig, "anomaly_type_breakdown.png")


# ─────────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str,
) -> str:
    """Plot and save a confusion matrix for a given model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Normal", "Anomaly"],
        colorbar=False,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────
# 4. ROC Curves (all models on one plot)
# ─────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    scores_dict: dict,   # {"Model Name": score_array, ...}
) -> str:
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = [COLORS["if"], COLORS["lof"], COLORS["hybrid"]]

    for (name, scores), color in zip(scores_dict.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Algorithm Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "roc_curves.png")


# ─────────────────────────────────────────────
# 5. Precision-Recall Curves
# ─────────────────────────────────────────────

def plot_pr_curves(
    y_true: np.ndarray,
    scores_dict: dict,
) -> str:
    """Overlay Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = [COLORS["if"], COLORS["lof"], COLORS["hybrid"]]

    for (name, scores), color in zip(scores_dict.items(), palette):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc_val = average_precision_score(y_true, scores)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name}  (AP = {pr_auc_val:.4f})")

    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – Algorithm Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "precision_recall_curves.png")


# ─────────────────────────────────────────────
# 6. Algorithm Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_algorithm_comparison(comparison_df: pd.DataFrame) -> str:
    """
    Grouped bar chart comparing Precision, Recall, F1, ROC-AUC, PR-AUC
    across all models.
    """
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    available = [m for m in metrics if m in comparison_df.columns]

    models = comparison_df.index.tolist()
    x      = np.arange(len(available))
    width  = 0.25 if len(models) == 3 else 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = [COLORS["if"], COLORS["lof"], COLORS["hybrid"]]

    for i, (model, color) in enumerate(zip(models, palette)):
        values = [float(comparison_df.loc[model, m]) for m in available]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=color, alpha=0.85,
                      edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in available])
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score")
    ax.set_title("Algorithm Performance Comparison (Test Set)", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "algorithm_comparison.png")


# ─────────────────────────────────────────────
# 7. Anomaly Score Distribution
# ─────────────────────────────────────────────

def plot_anomaly_score_distribution(
    y_true:     np.ndarray,
    scores_dict: dict,
) -> str:
    """KDE plot of anomaly scores split by true label for each model."""
    n_models = len(scores_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    palette = [COLORS["if"], COLORS["lof"], COLORS["hybrid"]]
    for ax, (name, scores), color in zip(axes, scores_dict.items(), palette):
        scores = np.asarray(scores)
        normal_scores  = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        ax.hist(normal_scores,  bins=40, alpha=0.6, color=COLORS["normal"],
                label="Normal",  density=True, edgecolor="white")
        ax.hist(anomaly_scores, bins=40, alpha=0.6, color=COLORS["anomaly"],
                label="Anomaly", density=True, edgecolor="white")
        ax.set_title(f"{name}\nAnomaly Score Distribution", fontweight="bold")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Anomaly Score Distributions by True Label", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "anomaly_scores.png")


# ─────────────────────────────────────────────
# 8. Top Suspicious Students
# ─────────────────────────────────────────────

def plot_suspicious_students(
    df_results: pd.DataFrame,
    score_col:  str = "hybrid_score",
    top_n:      int = 15,
) -> str:
    """Horizontal bar chart of the most suspicious students by average anomaly score."""
    if "student_id" not in df_results.columns or score_col not in df_results.columns:
        return ""

    top = (
        df_results.groupby("student_id")[score_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["student_id"][::-1], top[score_col][::-1],
                   color=COLORS["anomaly"], alpha=0.8, edgecolor="white")
    ax.set_xlabel("Average Hybrid Anomaly Score")
    ax.set_title(f"Top {top_n} Most Suspicious Students (Average Hybrid Score)",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, top[score_col][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    return _save(fig, "suspicious_students.png")


# ─────────────────────────────────────────────
# 9. Alpha Experiment (Hybrid Model)
# ─────────────────────────────────────────────

def plot_alpha_experiment(alpha_results: list) -> str:
    """Line plot of F1-score vs alpha for the hybrid model tuning."""
    df = pd.DataFrame(alpha_results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["alpha"], df["f1"], marker="o", color=COLORS["hybrid"],
            linewidth=2, markersize=8, label="Validation F1")
    best_row = df.loc[df["f1"].idxmax()]
    ax.axvline(x=best_row["alpha"], color="red", linestyle="--", lw=1.5,
               label=f"Best α = {best_row['alpha']:.2f}  (F1 = {best_row['f1']:.4f})")
    ax.set_xlabel("Alpha (IF weight)")
    ax.set_ylabel("F1-Score (Validation Set)")
    ax.set_title("Hybrid Model – Alpha Tuning Experiment\n[Student-Designed Improvement]",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(df["alpha"])
    fig.tight_layout()
    return _save(fig, "alpha_experiment.png")


# ─────────────────────────────────────────────
# 10. Hyperparameter Experiment Results
# ─────────────────────────────────────────────

def plot_hyperparameter_results(hp_df: pd.DataFrame) -> str:
    """Faceted line plots for IF and LOF hyperparameter sweeps."""
    if hp_df.empty:
        return ""

    models = hp_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        subset = hp_df[hp_df["model"] == model_name]
        if "n_estimators" in subset.columns:
            for cont in subset["contamination"].unique():
                grp = subset[subset["contamination"] == cont]
                ax.plot(grp["n_estimators"], grp["f1"], marker="o",
                        label=f"cont={cont}")
            ax.set_xlabel("n_estimators")
        elif "n_neighbors" in subset.columns:
            for cont in subset["contamination"].unique():
                grp = subset[subset["contamination"] == cont]
                ax.plot(grp["n_neighbors"], grp["f1"], marker="s",
                        label=f"cont={cont}")
            ax.set_xlabel("n_neighbors")
        ax.set_ylabel("F1-Score (Validation)")
        ax.set_title(f"{model_name} – Hyperparameter Search", fontweight="bold")
        ax.legend()

    fig.suptitle("Hyperparameter Experiment Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "hyperparameter_results.png")


# ─────────────────────────────────────────────
# 11. Attendance Over Time (per student sample)
# ─────────────────────────────────────────────

def plot_attendance_over_time(df: pd.DataFrame, student_id: str | None = None) -> str:
    """
    Plot a student's weekly_attendance_rate over time.
    If student_id is None, picks the student with the most anomalies.
    """
    if "is_anomaly" not in df.columns or "date" not in df.columns:
        return ""

    # Select student
    if student_id is None:
        student_id = (
            df[df["is_anomaly"] == 1]
            .groupby("student_id")["is_anomaly"]
            .count()
            .idxmax()
        )

    sub = df[df["student_id"] == student_id].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 5))
    normal  = sub[sub["is_anomaly"] == 0]
    anomaly = sub[sub["is_anomaly"] == 1]

    ax.plot(sub["date"], sub["weekly_attendance_rate"],
            color="steelblue", lw=1.5, alpha=0.7, label="Weekly Rate")
    ax.scatter(normal["date"],  normal["weekly_attendance_rate"],
               color=COLORS["normal"],  s=50, zorder=5, label="Normal")
    ax.scatter(anomaly["date"], anomaly["weekly_attendance_rate"],
               color=COLORS["anomaly"], s=100, marker="X", zorder=6, label="Anomaly")

    ax.set_title(f"Attendance Behaviour Over Time – {student_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Attendance Rate")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "attendance_over_time.png")
