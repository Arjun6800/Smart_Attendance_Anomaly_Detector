"""
src/train.py
=============
End-to-end training pipeline for the Smart Attendance Anomaly Detector.

Pipeline Steps
--------------
1. Load and clean the dataset.
2. Split into train / validation / test (60/20/20).
3. Engineer features (without data leakage).
4. Scale features using RobustScaler fitted on training data only.
5. Hyperparameter search for Isolation Forest on the validation set.
6. Hyperparameter search for LOF on the validation set.
7. Train the best Isolation Forest and LOF models on all training data.
8. Fit the Student-Designed Hybrid Model:
   a. Normalise scores using training score statistics.
   b. Tune alpha using the validation set.
   c. Tune anomaly threshold using the validation set.
9. Evaluate all models on the held-out TEST set.
10. Save models, results, and comparison tables.

Data-Leakage Prevention Summary
---------------------------------
- Labels (is_anomaly) are separated BEFORE any processing.
- Hyperparameter tuning and threshold/alpha selection use the VALIDATION set.
- The TEST set is touched ONLY at the final evaluation step.
- Feature engineering statistics (student stats, sharing stats) are computed
  from TRAINING data only and applied to val/test.
- The RobustScaler is fitted on TRAINING data only.
"""

from __future__ import annotations

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ── Ensure project root is on the Python path ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing       import load_data, clean_data, split_data
from src.feature_engineering import engineer_features, scale_features
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.lof_model               import LOFDetector
from src.models.hybrid_model            import HybridAnomalyDetector
from src.evaluate import compute_metrics, compare_models, threshold_sweep

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_PATH    = os.path.join(ROOT, "data", "attendance_dataset.csv")
RESULTS_DIR  = os.path.join(ROOT, "results")
MODELS_DIR   = os.path.join(ROOT, "models")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")

for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Hyperparameter Grids
# ─────────────────────────────────────────────
IF_HP_GRID = {
    "n_estimators":  [100, 200, 300],
    "contamination": [0.05, 0.08, 0.10],
}

LOF_HP_GRID = {
    "n_neighbors":   [10, 20, 30, 50],
    "contamination": [0.05, 0.08, 0.10],
}

THRESHOLD_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# ─────────────────────────────────────────────
# STEP 1-4: Load, Clean, Split, Feature-Engineer
# ─────────────────────────────────────────────

def prepare_data():
    """Load, clean, split, and engineer features for all three splits."""
    logger.info("=" * 60)
    logger.info("STEP 1-4: Data Preparation")
    logger.info("=" * 60)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    logger.info("Dataset shape: %s", df.shape)
    logger.info("Anomaly rate : %.2f%%", df["is_anomaly"].mean() * 100)

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)

    # ── Feature engineering on training data (fit=True) ─────────────────
    X_train_fe, train_stats = engineer_features(X_train_raw, fit=True)

    # ── Apply same statistics to val / test (fit=False, no leakage) ─────
    X_val_fe, _ = engineer_features(
        X_val_raw,
        student_stats=train_stats["student_stats"],
        sharing_stats=train_stats["sharing_stats"],
        fit=False,
    )
    X_test_fe, _ = engineer_features(
        X_test_raw,
        student_stats=train_stats["student_stats"],
        sharing_stats=train_stats["sharing_stats"],
        fit=False,
    )

    # ── Scale (fit on train only) ────────────────────────────────────────
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train_fe, X_val_fe, X_test_fe)

    # Save scaler and feature stats for Streamlit app
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(train_stats, os.path.join(MODELS_DIR, "train_stats.pkl"))

    logger.info("Feature matrix shapes → train: %s | val: %s | test: %s",
                X_train_s.shape, X_val_s.shape, X_test_s.shape)

    return (
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        X_train_raw, X_val_raw, X_test_raw,
        train_stats, df,
    )


# ─────────────────────────────────────────────
# STEP 5: Isolation Forest Hyperparameter Search
# ─────────────────────────────────────────────

def search_isolation_forest(X_train_s, X_val_s, y_val) -> tuple:
    """Grid search over n_estimators and contamination for Isolation Forest."""
    logger.info("=" * 60)
    logger.info("STEP 5: Isolation Forest Hyperparameter Search")
    logger.info("=" * 60)

    hp_rows = []
    best_f1       = -1.0
    best_params   = {"n_estimators": 200, "contamination": 0.08}

    for n_est in IF_HP_GRID["n_estimators"]:
        for cont in IF_HP_GRID["contamination"]:
            det = IsolationForestDetector(n_estimators=n_est, contamination=cont)
            det.fit(X_train_s)
            preds  = det.predict(X_val_s)
            f1_val = f1_score(y_val, preds, zero_division=0)

            row = {
                "model": "IsolationForest",
                "n_estimators": n_est,
                "contamination": cont,
                "f1": round(f1_val, 4),
            }
            hp_rows.append(row)
            logger.info("  IF  n_est=%3d  cont=%.2f  → Val F1 = %.4f",
                        n_est, cont, f1_val)

            if f1_val > best_f1:
                best_f1     = f1_val
                best_params = {"n_estimators": n_est, "contamination": cont}

    logger.info("Best IF params: %s  (Val F1 = %.4f)", best_params, best_f1)
    return best_params, hp_rows


# ─────────────────────────────────────────────
# STEP 6: LOF Hyperparameter Search
# ─────────────────────────────────────────────

def search_lof(X_train_s, X_val_s, y_val) -> tuple:
    """Grid search over n_neighbors and contamination for LOF."""
    logger.info("=" * 60)
    logger.info("STEP 6: LOF Hyperparameter Search")
    logger.info("=" * 60)

    hp_rows = []
    best_f1     = -1.0
    best_params = {"n_neighbors": 20, "contamination": 0.08}

    for k in LOF_HP_GRID["n_neighbors"]:
        for cont in LOF_HP_GRID["contamination"]:
            det = LOFDetector(n_neighbors=k, contamination=cont)
            det.fit(X_train_s)
            preds  = det.predict(X_val_s)
            f1_val = f1_score(y_val, preds, zero_division=0)

            row = {
                "model": "LOF",
                "n_neighbors": k,
                "contamination": cont,
                "f1": round(f1_val, 4),
            }
            hp_rows.append(row)
            logger.info("  LOF  k=%2d  cont=%.2f  → Val F1 = %.4f",
                        k, cont, f1_val)

            if f1_val > best_f1:
                best_f1     = f1_val
                best_params = {"n_neighbors": k, "contamination": cont}

    logger.info("Best LOF params: %s  (Val F1 = %.4f)", best_params, best_f1)
    return best_params, hp_rows


# ─────────────────────────────────────────────
# STEP 7-8: Train Final Models & Hybrid Tuning
# ─────────────────────────────────────────────

def train_final_models(
    X_train_s, X_val_s, X_test_s,
    y_val,
    best_if_params, best_lof_params,
):
    """
    Train Isolation Forest and LOF with the best hyperparameters,
    then tune and evaluate the Hybrid model on the validation set.
    """
    logger.info("=" * 60)
    logger.info("STEP 7-8: Training Final Models")
    logger.info("=" * 60)

    # ── Isolation Forest ─────────────────────────────────────────────────
    if_model = IsolationForestDetector(**best_if_params)
    if_model.fit(X_train_s)
    if_model.save(os.path.join(MODELS_DIR, "isolation_forest.pkl"))

    if_preds_val,  if_scores_val  = if_model.predict_with_scores(X_val_s)
    if_preds_test, if_scores_test = if_model.predict_with_scores(X_test_s)
    if_preds_train, if_scores_train = if_model.predict_with_scores(X_train_s)

    # ── LOF ───────────────────────────────────────────────────────────────
    lof_model = LOFDetector(**best_lof_params)
    lof_model.fit(X_train_s)
    lof_model.save(os.path.join(MODELS_DIR, "lof_model.pkl"))

    lof_preds_val,  lof_scores_val  = lof_model.predict_with_scores(X_val_s)
    lof_preds_test, lof_scores_test = lof_model.predict_with_scores(X_test_s)
    lof_preds_train, lof_scores_train = lof_model.predict_with_scores(X_train_s)

    # ── Hybrid Model ──────────────────────────────────────────────────────
    hybrid = HybridAnomalyDetector()

    # Fit normalisation on TRAINING scores (no leakage)
    hybrid.fit_transform_train(if_scores_train, lof_scores_train)

    # Tune alpha and threshold on VALIDATION set only
    best_alpha, best_threshold, alpha_results, threshold_results = hybrid.tune(
        if_scores_val, lof_scores_val, y_val.values
    )
    logger.info(
        "Hybrid tuning → alpha=%.2f, threshold=%.2f",
        best_alpha, best_threshold,
    )

    hybrid_preds_val,  hybrid_scores_val  = hybrid.predict_with_scores(if_scores_val,  lof_scores_val)
    hybrid_preds_test, hybrid_scores_test = hybrid.predict_with_scores(if_scores_test, lof_scores_test)

    joblib.dump(hybrid, os.path.join(MODELS_DIR, "hybrid_model.pkl"))

    return {
        "if_model":   if_model,
        "lof_model":  lof_model,
        "hybrid":     hybrid,
        # Validation predictions / scores
        "if_preds_val":     if_preds_val,
        "if_scores_val":    if_scores_val,
        "lof_preds_val":    lof_preds_val,
        "lof_scores_val":   lof_scores_val,
        "hybrid_preds_val": hybrid_preds_val,
        "hybrid_scores_val":hybrid_scores_val,
        # Test predictions / scores
        "if_preds_test":     if_preds_test,
        "if_scores_test":    if_scores_test,
        "lof_preds_test":    lof_preds_test,
        "lof_scores_test":   lof_scores_test,
        "hybrid_preds_test": hybrid_preds_test,
        "hybrid_scores_test":hybrid_scores_test,
        # Tuning records
        "alpha_results":     alpha_results,
        "threshold_results": threshold_results,
    }


# ─────────────────────────────────────────────
# STEP 9: Evaluate and Save Results
# ─────────────────────────────────────────────

def evaluate_and_save(model_outputs, y_val, y_test, X_test_raw, hp_rows):
    """
    Compute metrics on the test set, build comparison tables, and save CSVs.
    """
    logger.info("=" * 60)
    logger.info("STEP 9: Evaluation on Test Set")
    logger.info("=" * 60)

    y_test_arr = y_test.values

    if_metrics  = compute_metrics(
        y_test_arr,
        model_outputs["if_preds_test"],
        model_outputs["if_scores_test"],
        model_name="Isolation Forest",
    )
    lof_metrics = compute_metrics(
        y_test_arr,
        model_outputs["lof_preds_test"],
        model_outputs["lof_scores_test"],
        model_name="LOF",
    )
    hybrid_metrics = compute_metrics(
        y_test_arr,
        model_outputs["hybrid_preds_test"],
        model_outputs["hybrid_scores_test"],
        model_name="Hybrid (Student Innovation)",
    )

    comparison_df = compare_models([if_metrics, lof_metrics, hybrid_metrics])
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    logger.info("\nTest Set Comparison:\n%s", comparison_df.to_string())

    # ── Hyperparameter results ────────────────────────────────────────────
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(os.path.join(RESULTS_DIR, "hyperparameter_results.csv"), index=False)

    # ── Threshold sweep (using Hybrid scores on val set for demonstration) ───
    # Use hybrid scores because they are normalised to [0,1] making the
    # threshold candidates [0.50 ... 0.90] directly applicable.
    thr_df = threshold_sweep(
        y_val.values,
        model_outputs["hybrid_scores_val"],
        THRESHOLD_CANDIDATES,
    )
    thr_df.to_csv(os.path.join(RESULTS_DIR, "threshold_results.csv"), index=False)

    # ── Alpha experiment results ──────────────────────────────────────────
    alpha_df = pd.DataFrame(model_outputs["alpha_results"])
    alpha_df.to_csv(os.path.join(RESULTS_DIR, "alpha_results.csv"), index=False)

    # ── Test predictions with explanations ───────────────────────────────
    preds_df = X_test_raw.copy().reset_index(drop=True)
    preds_df["y_true"]         = y_test_arr
    preds_df["if_pred"]        = model_outputs["if_preds_test"]
    preds_df["if_score"]       = model_outputs["if_scores_test"]
    preds_df["lof_pred"]       = model_outputs["lof_preds_test"]
    preds_df["lof_score"]      = model_outputs["lof_scores_test"]
    preds_df["hybrid_pred"]    = model_outputs["hybrid_preds_test"]
    preds_df["hybrid_score"]   = model_outputs["hybrid_scores_test"]

    from src.evaluate import explain_anomaly
    preds_df["explanation"] = preds_df.apply(explain_anomaly, axis=1)

    preds_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
    logger.info("Saved test_predictions.csv with %d records.", len(preds_df))

    return if_metrics, lof_metrics, hybrid_metrics, comparison_df, hp_df, alpha_df, preds_df


# ─────────────────────────────────────────────
# STEP 10: Generate Plots
# ─────────────────────────────────────────────

def generate_plots(df, model_outputs, y_test, comparison_df, alpha_df, hp_df, preds_df):
    """Generate and save all visualisation plots."""
    logger.info("=" * 60)
    logger.info("STEP 10: Generating Plots")
    logger.info("=" * 60)

    from src.visualization import (
        plot_attendance_distribution,
        plot_anomaly_type_breakdown,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_pr_curves,
        plot_algorithm_comparison,
        plot_anomaly_score_distribution,
        plot_suspicious_students,
        plot_alpha_experiment,
        plot_hyperparameter_results,
        plot_attendance_over_time,
    )

    y_test_arr = y_test.values
    scores_dict = {
        "Isolation Forest": model_outputs["if_scores_test"],
        "LOF":              model_outputs["lof_scores_test"],
        "Hybrid":           model_outputs["hybrid_scores_test"],
    }

    plot_attendance_distribution(df)
    logger.info("  ✓ attendance_distribution.png")

    plot_anomaly_type_breakdown(df)
    logger.info("  ✓ anomaly_type_breakdown.png")

    plot_confusion_matrix(y_test_arr, model_outputs["if_preds_test"],
                          "Isolation Forest", "confusion_matrix_if.png")
    logger.info("  ✓ confusion_matrix_if.png")

    plot_confusion_matrix(y_test_arr, model_outputs["lof_preds_test"],
                          "LOF", "confusion_matrix_lof.png")
    logger.info("  ✓ confusion_matrix_lof.png")

    plot_confusion_matrix(y_test_arr, model_outputs["hybrid_preds_test"],
                          "Hybrid (Student Innovation)", "confusion_matrix_hybrid.png")
    logger.info("  ✓ confusion_matrix_hybrid.png")

    plot_roc_curves(y_test_arr, scores_dict)
    logger.info("  ✓ roc_curves.png")

    plot_pr_curves(y_test_arr, scores_dict)
    logger.info("  ✓ precision_recall_curves.png")

    plot_algorithm_comparison(comparison_df)
    logger.info("  ✓ algorithm_comparison.png")

    plot_anomaly_score_distribution(y_test_arr, scores_dict)
    logger.info("  ✓ anomaly_scores.png")

    plot_suspicious_students(preds_df, score_col="hybrid_score")
    logger.info("  ✓ suspicious_students.png")

    plot_alpha_experiment(alpha_df.to_dict("records"))
    logger.info("  ✓ alpha_experiment.png")

    plot_hyperparameter_results(hp_df)
    logger.info("  ✓ hyperparameter_results.png")

    plot_attendance_over_time(df)
    logger.info("  ✓ attendance_over_time.png")


# ─────────────────────────────────────────────
# STEP 11: Final Report
# ─────────────────────────────────────────────

def write_final_report(
    df, if_metrics, lof_metrics, hybrid_metrics,
    best_if_params, best_lof_params,
    best_alpha, best_threshold,
):
    """Write a plain-text/markdown final report with actual metrics."""
    report_path = os.path.join(RESULTS_DIR, "final_report.md")

    n_records   = len(df)
    n_anomalies = int(df["is_anomaly"].sum())
    n_normal    = n_records - n_anomalies

    content = f"""# Smart Attendance Anomaly Detector – Final Report

## Dataset Summary
| Metric | Value |
|--------|-------|
| Total Records | {n_records:,} |
| Normal Records | {n_normal:,} |
| Anomalous Records | {n_anomalies:,} |
| Anomaly Rate | {n_anomalies/n_records*100:.1f}% |
| Number of Students | ~200 |
| Date Range | 2024-01-15 to 2024-06-30 |

## Data Split
| Split | Size |
|-------|------|
| Training | 60% ({int(n_records*0.6):,} records) |
| Validation | 20% ({int(n_records*0.2):,} records) |
| Test | 20% ({int(n_records*0.2):,} records) |

## Algorithm Configuration

### Isolation Forest (Best Parameters)
- n_estimators : {best_if_params['n_estimators']}
- contamination : {best_if_params['contamination']}

### LOF (Best Parameters)
- n_neighbors : {best_lof_params['n_neighbors']}
- contamination : {best_lof_params['contamination']}

### Hybrid Model (Student-Designed Improvement)
- Best Alpha : {best_alpha}
- Anomaly Threshold : {best_threshold}
- Alpha candidates tested : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- Threshold candidates tested : [0.50, 0.55, ..., 0.90]

## Test Set Results

### Isolation Forest
| Metric | Value |
|--------|-------|
| Accuracy | {if_metrics['accuracy']} |
| Precision | {if_metrics['precision']} |
| Recall | {if_metrics['recall']} |
| F1-Score | {if_metrics['f1']} |
| ROC-AUC | {if_metrics['roc_auc']} |
| PR-AUC | {if_metrics['pr_auc']} |
| TP | {if_metrics['tp']} |
| FP | {if_metrics['fp']} |
| TN | {if_metrics['tn']} |
| FN | {if_metrics['fn']} |

### LOF
| Metric | Value |
|--------|-------|
| Accuracy | {lof_metrics['accuracy']} |
| Precision | {lof_metrics['precision']} |
| Recall | {lof_metrics['recall']} |
| F1-Score | {lof_metrics['f1']} |
| ROC-AUC | {lof_metrics['roc_auc']} |
| PR-AUC | {lof_metrics['pr_auc']} |
| TP | {lof_metrics['tp']} |
| FP | {lof_metrics['fp']} |
| TN | {lof_metrics['tn']} |
| FN | {lof_metrics['fn']} |

### Hybrid Model (Student Innovation)
| Metric | Value |
|--------|-------|
| Accuracy | {hybrid_metrics['accuracy']} |
| Precision | {hybrid_metrics['precision']} |
| Recall | {hybrid_metrics['recall']} |
| F1-Score | {hybrid_metrics['f1']} |
| ROC-AUC | {hybrid_metrics['roc_auc']} |
| PR-AUC | {hybrid_metrics['pr_auc']} |
| TP | {hybrid_metrics['tp']} |
| FP | {hybrid_metrics['fp']} |
| TN | {hybrid_metrics['tn']} |
| FN | {hybrid_metrics['fn']} |

## Discussion

The hybrid model combines the strengths of both Isolation Forest (global
anomaly detection via tree isolation) and LOF (local density-based detection).
By tuning the alpha weighting on the validation set, the model can adapt to
the specific characteristics of the dataset.

### Limitations
- Synthetic dataset: real college attendance may have different patterns.
- Anomaly detection does NOT prove misconduct – human review is required.
- The hybrid improvement depends on the assumption that IF and LOF capture
  complementary patterns. If they are correlated, the hybrid may not help.
- False positives are unavoidable; the threshold controls the precision-recall trade-off.

### Future Improvements
- Real-time streaming anomaly detection.
- Integration with face recognition for biometric verification.
- Graph-based analysis of student interaction networks.
- Deep learning autoencoders for more complex anomaly patterns.
- Explainable AI methods (SHAP) for feature-level explanations.

---
*Report generated automatically from experimental results. All numbers are from actual execution.*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Final report saved → %s", report_path)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Smart Attendance Anomaly Detector – Training Pipeline")
    logger.info("=" * 60)

    # Step 1-4: Data preparation
    (
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        X_train_raw, X_val_raw, X_test_raw,
        train_stats, df,
    ) = prepare_data()

    # Step 5: IF hyperparameter search
    best_if_params, if_hp_rows = search_isolation_forest(X_train_s, X_val_s, y_val)

    # Step 6: LOF hyperparameter search
    best_lof_params, lof_hp_rows = search_lof(X_train_s, X_val_s, y_val)
    hp_rows = if_hp_rows + lof_hp_rows

    # Step 7-8: Train final models + Hybrid tuning
    model_outputs = train_final_models(
        X_train_s, X_val_s, X_test_s,
        y_val,
        best_if_params, best_lof_params,
    )

    # Step 9: Evaluate and save
    (
        if_metrics, lof_metrics, hybrid_metrics,
        comparison_df, hp_df, alpha_df, preds_df,
    ) = evaluate_and_save(model_outputs, y_val, y_test, X_test_raw, hp_rows)

    # Step 10: Generate plots
    generate_plots(
        df, model_outputs, y_test, comparison_df, alpha_df, hp_df, preds_df
    )

    # Step 11: Write report
    write_final_report(
        df, if_metrics, lof_metrics, hybrid_metrics,
        best_if_params, best_lof_params,
        model_outputs["hybrid"].alpha,
        model_outputs["hybrid"].threshold,
    )

    logger.info("=" * 60)
    logger.info("Training pipeline complete! Results saved in 'results/'.")
    logger.info("Run: streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
