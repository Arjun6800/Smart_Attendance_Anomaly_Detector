"""
src/evaluate.py
================
Quantitative evaluation module for the Smart Attendance Anomaly Detector.

Computes standard classification metrics adapted for the anomaly detection context.

Why Standard Accuracy is Insufficient
--------------------------------------
With 8% anomaly rate, a naive model that labels EVERYTHING as "normal" achieves
92% accuracy while having 0% recall for anomalies.  This is useless in practice.

Therefore we prioritise:
- Precision : Of all records flagged as anomalies, what fraction are real anomalies?
- Recall    : Of all real anomalies, what fraction did we detect?
- F1-score  : Harmonic mean of precision and recall (balanced metric).
- ROC-AUC   : Area under the ROC curve (threshold-independent).
- PR-AUC    : Area under the Precision-Recall curve (better for imbalanced data).

Metrics Computed
----------------
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix (TP, TN, FP, FN)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,   # PR-AUC
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_score: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true     : Ground-truth labels (0=normal, 1=anomaly).
    y_pred     : Predicted labels (0=normal, 1=anomaly).
    y_score    : Continuous anomaly scores (higher = more anomalous).
    model_name : Descriptive name for logging purposes.

    Returns
    -------
    dict
        Dictionary of metric name → float value.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    # ── Confusion matrix components ──────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm[actual][predicted]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC is undefined if only one class is present
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC (average precision) is the area under the precision-recall curve.
    # More informative than ROC-AUC for highly imbalanced datasets.
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = float("nan")

    metrics = {
        "model":     model_name,
        "accuracy":  round(acc,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "roc_auc":   round(roc_auc,   4),
        "pr_auc":    round(pr_auc,    4),
        "tp":        int(tp),
        "fp":        int(fp),
        "tn":        int(tn),
        "fn":        int(fn),
    }

    logger.info(
        "%s → Precision=%.4f  Recall=%.4f  F1=%.4f  ROC-AUC=%.4f  PR-AUC=%.4f",
        model_name, precision, recall, f1, roc_auc, pr_auc
    )
    return metrics


def compare_models(results: list) -> pd.DataFrame:
    """
    Build a comparison table from a list of metric dictionaries.

    Parameters
    ----------
    results : list of dict
        Each dict is the output of ``compute_metrics()``.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    df = pd.DataFrame(results)
    display_cols = ["model", "precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"]
    df = df[[c for c in display_cols if c in df.columns]]
    df = df.set_index("model")
    return df


def threshold_sweep(
    y_true:    np.ndarray,
    y_score:   np.ndarray,
    thresholds: list,
) -> pd.DataFrame:
    """
    Evaluate different anomaly thresholds on a given dataset split.

    Used on the VALIDATION SET to select the best operating threshold.

    Parameters
    ----------
    y_true      : Ground-truth labels.
    y_score     : Continuous anomaly scores.
    thresholds  : List of threshold values to evaluate.

    Returns
    -------
    pd.DataFrame
        One row per threshold with precision / recall / F1.
    """
    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


def explain_anomaly(row: pd.Series) -> str:
    """
    Rule-based explanation layer.

    Generates a human-readable reason WHY a record was flagged.

    IMPORTANT: These rules are heuristics applied AFTER the AI model.
    They do NOT represent the mathematical reasoning of Isolation Forest or LOF.
    The AI score is computed independently; these rules help humans understand
    the most prominent signal in the flagged record.

    Parameters
    ----------
    row : pd.Series
        A single attendance record with feature columns.

    Returns
    -------
    str
        Human-readable explanation string.
    """
    reasons = []

    # ── Time-based signals ────────────────────────────────────────────────
    hour = row.get("hour", -1)
    if hour != -1 and (hour < 6 or hour > 21):
        reasons.append(f"Unusual check-in time ({int(hour):02d}:00 – outside normal hours)")

    # ── Proxy signals ─────────────────────────────────────────────────────
    dev_count = row.get("device_student_count", 1)
    if dev_count > 5:
        reasons.append(f"Device shared by {int(dev_count)} students (proxy risk)")

    ip_count = row.get("ip_student_count", 1)
    if ip_count > 8:
        reasons.append(f"IP address shared by {int(ip_count)} students")

    # ── Gap-based signals ─────────────────────────────────────────────────
    gap = row.get("previous_attendance_gap", 1.0)
    if gap < 0.05:
        reasons.append(f"Extremely short gap ({gap:.4f} hrs) since last check-in (burst risk)")

    # ── Rate-based signals ────────────────────────────────────────────────
    weekly_rate = row.get("weekly_attendance_rate", 0.5)
    if weekly_rate > 0.98:
        reasons.append("Suspiciously perfect attendance rate (> 98%)")
    elif weekly_rate < 0.2:
        reasons.append(f"Very low attendance rate ({weekly_rate:.0%})")

    # ── Historical deviation signals ──────────────────────────────────────
    hist_dev = row.get("historical_deviation", 0.0)
    if hist_dev > 0.4:
        reasons.append(f"Large deviation from historical pattern ({hist_dev:.2f})")

    # ── Proxy risk indicator ──────────────────────────────────────────────
    if row.get("proxy_risk_indicator", 0) == 1:
        reasons.append("Proxy risk flag triggered (device/IP pattern)")

    if not reasons:
        reasons.append("Anomalous pattern detected by AI model (no single dominant signal)")

    return " | ".join(reasons)
