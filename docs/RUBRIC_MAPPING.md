# Project Evaluation Mapping – Smart Attendance Anomaly Detector
## Architectural and Feature Implementation Mapping

---

## 1. Problem Formulation and Originality

**What We Did:**
- Formulated a real-world problem: detecting suspicious attendance patterns in student records that conventional systems miss.
- Identified 6 distinct anomaly types (proxy, time, location, behavioural, burst, historical deviation).
- Proposed an original hybrid anomaly scoring approach combining two established algorithms.
- Designed a synthetic but realistic dataset with ground-truth labels for reproducible evaluation.

**Files:**
- `README.md` – Problem Statement and Motivation sections
- `data/generate_dataset.py` – 6 anomaly types with realistic injection logic
- `src/models/hybrid_model.py` – Original student-designed hybrid approach

---

## 2. AI Concepts / Algorithm Implementation

**What We Did:**
- Fully implemented **Isolation Forest** using sklearn, with complete documentation of algorithm mechanics (path length, score direction, contamination).
- Fully implemented **Local Outlier Factor** with `novelty=True` for transductive-to-inductive use, with complete explanation of LRD, LOF score, k-NN.
- Implemented a **Student-Designed Hybrid** model combining both algorithms with normalised scoring.
- All code has thorough docstrings, type hints, and algorithmic explanations in comments.

**Files:**
- `src/models/isolation_forest_model.py`
- `src/models/lof_model.py`
- `src/models/hybrid_model.py`
- `src/feature_engineering.py`
- `src/preprocessing.py`

---

## 3. Comparison of Algorithms

**What We Did:**
- Evaluated all three models on the same held-out test set using identical features.
- Compared on 6 metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- Generated confusion matrices for all three models.
- Produced ROC curves and Precision-Recall curves on a single plot for visual comparison.
- Built an algorithm comparison bar chart.
- Discussed where each algorithm performs better/worse based on experimental results.

**Files:**
- `results/model_comparison.csv` – Quantitative comparison table
- `results/plots/algorithm_comparison.png`
- `results/plots/roc_curves.png`
- `results/plots/precision_recall_curves.png`
- `results/plots/confusion_matrix_if.png`, `_lof.png`, `_hybrid.png`
- `results/final_report.md` – Discussion section

---

## 4. Dataset / Environment and Experimentation

**What We Did:**
- Created a synthetic dataset with 10,000 records, ~200 students, 8% anomaly rate.
- Dataset has 6 distinct anomaly types with a ground-truth label (`is_anomaly`).
- Fixed random seed (42) ensures full reproducibility.
- Proper 60/20/20 train/validation/test split with stratification.
- Hyperparameter experiments: 3×3 grid for IF, 4×3 grid for LOF.
- Alpha experiment: 7 candidate values.
- Threshold experiment: 9 candidate values.

**Files:**
- `data/generate_dataset.py`
- `data/attendance_dataset.csv`
- `data/README.md`
- `results/hyperparameter_results.csv`
- `results/threshold_results.csv`
- `results/alpha_results.csv`

---

## 5. Evaluation and Interpretation

**What We Did:**
- Implemented 6 evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- Explicitly justified why accuracy alone is insufficient (class imbalance, 8% anomaly rate).
- Generated confusion matrices with TP/TN/FP/FN counts.
- Implemented rule-based explanation layer to explain WHY each record was flagged.
- Generated ROC and PR curves with AUC values.
- Wrote a `final_report.md` with full metric discussion, limitations, and future work.

**Files:**
- `src/evaluate.py` – `compute_metrics()`, `threshold_sweep()`, `explain_anomaly()`
- `results/model_comparison.csv`
- `results/test_predictions.csv` (includes explanation column)
- `results/final_report.md`
- `results/plots/` (multiple evaluation plots)

---

## 6. Student's Own Improvement / Innovation

**What We Did (Student-Designed Improvement):**
- Designed a **Hybrid Attendance Anomaly Score** that combines IF and LOF scores.
- Score normalisation prevents one algorithm from dominating due to scale differences.
- Alpha parameter (IF weight) is tuned experimentally on the validation set.
- Anomaly threshold is also tuned on the validation set.
- The improvement is **evaluated honestly** – we report whether it actually outperforms individual models.
- The report explicitly discusses when the hybrid may fail.

**Formula:**
```
HybridScore = α × IF_norm + (1 - α) × LOF_norm
```

**Clearly labelled as "Student-Designed Improvement" throughout:**
- Code comments in `hybrid_model.py`
- README.md Student Innovation section
- Streamlit app Algorithm Comparison page
- `results/final_report.md`

**Files:**
- `src/models/hybrid_model.py`
- `results/alpha_results.csv`
- `results/plots/alpha_experiment.png`

---

## 7. Working Application / Interactive Demo

**What We Did:**
- Built a **multi-page Streamlit web application** (`app.py`) with 6 pages:
  1. Dashboard (summary, top suspicious students)
  2. Upload & Detect (upload CSV or use demo data, download predictions)
  3. Student Analysis (per-student timeline and anomaly events)
  4. Algorithm Comparison (metrics, charts, alpha tuning)
  5. Visualisations (browse all generated plots)
  6. Model Information (algorithm explanations, ethics)
- Demo data (`data/demo_attendance.csv`) works immediately without internet.
- The app can run detection on uploaded CSVs in real time.
- Predictions are downloadable as CSV.

**Files:**
- `app.py`
- `data/demo_attendance.csv`
- `run_project.py` (one-command pipeline)

**To demonstrate:**
```bash
streamlit run app.py
```

---

## 8. Viva and Technical Understanding

**What We Did:**
- Created `docs/VIVA.md` with 30 questions and concise answers.
- Questions cover: algorithm mechanics, feature engineering, evaluation metrics,
  data leakage, hybrid innovation, limitations, ethical considerations.
- All answers are at undergraduate level – clear and explainable without memorisation.

**Files:**
- `docs/VIVA.md`
- `docs/RUBRIC_MAPPING.md` (this file)

---

## Summary Table

| Evaluation Dimension | Key Deliverables & Implementation |
|----------------------|-----------------------------------|
| Problem Formulation & Originality | Hybrid approach + 6 anomaly types |
| AI Concepts & Algorithm Implementation | IF + LOF + Hybrid fully implemented |
| Comparison of Algorithms | 6 metrics + confusion matrices + ROC/PR curves |
| Dataset, Environment & Experimentation | 10K synthetic records + HP grid search |
| Evaluation and Interpretation | Full metrics suite + rule-based explanations |
| Student-Designed Innovation | Hybrid model with alpha tuning |
| Working Application & Demo | Streamlit app with 6 pages |
| Viva & Technical Understanding | docs/VIVA.md with 30 comprehensive Q&A |
