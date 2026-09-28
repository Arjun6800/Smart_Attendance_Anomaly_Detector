# Smart Attendance Anomaly Detector

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)

An AI-powered system that analyses student attendance records and identifies suspicious or unusual patterns using **Isolation Forest**, **Local Outlier Factor (LOF)**, and a **Student-Designed Hybrid Anomaly Scoring** method.

---

## 1. Problem Statement

Conventional attendance systems record only **present** or **absent**. They cannot detect:

- **Proxy attendance** – A friend marks attendance on behalf of another student
- **Attendance bursts** – Multiple check-ins within an impossibly short time
- **Unusual device/IP patterns** – Same device used by many different students simultaneously
- **Time anomalies** – Attendance recorded at midnight or 3 AM
- **Behavioural anomalies** – A usually regular student suddenly shows erratic attendance
- **Historical deviations** – A student's attendance pattern suddenly diverges from their own baseline

---

## 2. Motivation

As educational institutions move towards digital attendance systems, fraudulent patterns become increasingly sophisticated. Rule-based filters (e.g., "flag if attendance < 75%") are too rigid and miss complex multi-feature anomalies. AI-based anomaly detection can analyse combinations of features — time, location, device, IP, historical behaviour — and surface suspicious records that no single rule would catch.

---

## 3. Objectives

1. Build a robust anomaly detection system on student attendance data.
2. Implement and compare Isolation Forest and LOF.
3. Design and evaluate a student-originated Hybrid Anomaly Score.
4. Produce quantitative experimental results with proper train/val/test splits.
5. Provide human-readable explanations for every flagged record.
6. Deliver an interactive web application for demonstration.

---

## 4. AI Approaches

### Isolation Forest

Tree-based ensemble anomaly detection (Liu et al., 2008). Builds random Isolation Trees and measures path length to isolate each data point. Anomalies have shorter path lengths because they are rare and different.

**Best for:** Global outliers, high-dimensional data, fast training.

### Local Outlier Factor (LOF)

Density-based anomaly detection (Breunig et al., 2000). Computes the ratio of local reachability density of each point compared to its k-nearest neighbours. Points in sparse regions relative to their neighbours score as anomalies.

**Best for:** Local anomalies, dense cluster structures, irregular shapes.

### Hybrid Model (Student Innovation)

```
HybridScore = α × IF_score_norm + (1 - α) × LOF_score_norm
```

Both scores normalised to [0, 1] using training statistics. Alpha and threshold tuned on validation set.

---

## 5. Student-Designed Improvement

**Motivation:** IF and LOF have complementary strengths. IF misses local anomalies; LOF misses some global ones. Combining their normalised scores with data-driven weighting aims to leverage both.

**How scores are combined:**
1. Obtain raw IF score (negated decision_function)
2. Obtain raw LOF score (negated decision_function)  
3. Normalise both using training min/max (no leakage)
4. Combine: `HybridScore = α × IF_norm + (1-α) × LOF_norm`
5. Apply tuned threshold for binary predictions

**Alpha selection:** F1-score maximisation on validation set across candidates [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].

**Honest note:** The hybrid improvement is validated experimentally. If results show it does not outperform individual models, this is reported honestly with discussion of reasons.

**When hybrid may fail:** If IF and LOF errors are correlated (they both miss or flag the same records), combining them adds no benefit and may degrade a better individual model.

---

## 6. Dataset

- **Type:** Synthetic (no real student data)
- **Records:** 10,000
- **Students:** ~200
- **Anomaly rate:** ~8% (800 records)
- **Date range:** 2024-01-15 to 2024-06-30
- **Random seed:** 42 (fully reproducible)

### Features

| Feature | Description |
|---------|-------------|
| student_id | Unique student identifier |
| date, day_of_week | Date and weekday |
| hour, minute | Check-in time |
| location_id | Classroom/lab |
| ip_address, device_id | Network/device identifiers |
| previous_attendance_gap | Hours since last check-in |
| weekly_attendance_rate | Fraction of weekly classes attended |
| monthly_attendance_rate | Fraction of monthly classes attended |
| historical_deviation | Absolute deviation from long-term average |
| proxy_risk_indicator | Binary: 1 if proxy-attack pattern |
| **is_anomaly** | **Ground truth label (evaluation only)** |

### Anomaly Types Injected

1. **Proxy** – Shared device/IP + very short gap
2. **Time Anomaly** – Midnight to 5 AM check-ins
3. **Location Anomaly** – Different room than normal
4. **Behavioural Anomaly** – Sudden rate change
5. **Burst** – Multiple check-ins in seconds
6. **Historical Deviation** – Large long-term baseline deviation

---

## 7. Feature Engineering

Engineered features (in `src/feature_engineering.py`):

| Feature | Purpose |
|---------|---------|
| hour_sin, hour_cos | Cyclic hour encoding (preserves 23→0 continuity) |
| minute_sin, minute_cos | Cyclic minute encoding |
| hour_deviation | Absolute difference from student's median hour |
| gap_log | Log-transformed attendance gap (reduces skew) |
| absence_ratio | absence_count / total_classes |
| late_ratio | late_count / total_classes |
| attendance_rate_diff | \|weekly_rate - monthly_rate\| |
| device_student_count | How many students use this device |
| ip_student_count | How many students share this IP |
| device_frequency | Overall device occurrence rate |
| ip_frequency | Overall IP occurrence rate |

**Note:** `student_id` is never used directly as a numeric feature. Student-level statistics are derived and merged separately.

---

## 8. Experimental Methodology

### Data Split
- **60%** Training → model fitting
- **20%** Validation → hyperparameter / threshold / alpha selection
- **20%** Test → final unbiased evaluation (touched only at the end)

Stratified split preserves anomaly ratio in each partition.

### Hyperparameter Tuning (Validation Set)
| Algorithm | Parameter | Candidates |
|-----------|-----------|------------|
| Isolation Forest | n_estimators | [100, 200, 300] |
| Isolation Forest | contamination | [0.05, 0.08, 0.10] |
| LOF | n_neighbors | [10, 20, 30, 50] |
| LOF | contamination | [0.05, 0.08, 0.10] |

### Threshold Tuning
Candidates: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
Selection criterion: Maximise F1-score on validation set.

### Alpha Tuning (Hybrid)
Candidates: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
Selection criterion: Maximise F1-score on validation set.

---

## 9. Evaluation Metrics

| Metric | Why Used |
|--------|----------|
| **Accuracy** | Baseline overview (insufficient alone due to imbalance) |
| **Precision** | Fraction of flagged records that are real anomalies |
| **Recall** | Fraction of actual anomalies that were caught |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Threshold-independent discrimination ability |
| **PR-AUC** | Better than ROC-AUC for imbalanced datasets |

---

## 10. Results

*Generated from actual experiment execution. See `results/model_comparison.csv` for exact numbers.*

Results are in: `results/model_comparison.csv`

---

## 11. Algorithm Comparison

**Isolation Forest** performs well on global anomalies (time anomalies, burst events) but may miss locally anomalous records in dense clusters.

**LOF** excels at detecting local density anomalies but is more sensitive to the choice of `n_neighbors` and can be slower on large datasets.

**Hybrid Model** aims to combine both — whether it outperforms individual models is determined by experimental results (see `results/model_comparison.csv`).

Neither algorithm is universally superior. The best choice depends on the data characteristics and the type of anomalies present.

---

## 12. Student Innovation

The **Hybrid Attendance Anomaly Score** is entirely student-designed:

1. **Normalisation** of both model scores to the same range — prevents scale imbalance
2. **Weighted combination** with tunable alpha — data-driven, not arbitrary
3. **Dual tuning** (alpha + threshold) exclusively on validation set — no data leakage
4. **Honest evaluation** — improvement is verified (or honestly reported as not significant)

This goes beyond simply running two sklearn algorithms: the student designed the combination strategy, normalisation approach, and tuning methodology.

---

## 13. How to Install

```bash
# Clone the repository
git clone <repo-url>
cd Smart_Attendance_Anomaly_Detector

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 14. How to Run

### Option A: Full Pipeline (Recommended)

```bash
# Step 1: Generate dataset (creates data/attendance_dataset.csv and data/demo_attendance.csv)
python data/generate_dataset.py

# Step 2: Train all models, run experiments, generate results and plots
python run_project.py

# Step 3: Launch the web application
streamlit run app.py
```

### Option B: Manual Steps

```bash
python data/generate_dataset.py   # Generate data
python -m src.train               # Train models
streamlit run app.py              # Launch app
```

---

## 15. Demo

After running `python run_project.py`:

1. Open the Streamlit app: `streamlit run app.py`
2. Navigate to **Dashboard** → view detected anomalies and suspicious students
3. Go to **Upload & Detect** → click **Demo Data** tab → click **Run Anomaly Detection**
4. Explore **Student Analysis** → select any student to see their timeline
5. View **Algorithm Comparison** → compare IF, LOF, and Hybrid metrics
6. Browse **Visualisations** → see all generated plots

---

## 16. Example Output

After running the pipeline, the following files are generated:

```
results/
├── model_comparison.csv         # Final metric comparison
├── hyperparameter_results.csv   # HP search results
├── threshold_results.csv        # Threshold sweep
├── alpha_results.csv            # Hybrid alpha experiment
├── test_predictions.csv         # Per-record predictions + explanations
├── final_report.md              # Full narrative report
└── plots/
    ├── attendance_distribution.png
    ├── anomaly_type_breakdown.png
    ├── confusion_matrix_if.png
    ├── confusion_matrix_lof.png
    ├── confusion_matrix_hybrid.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── algorithm_comparison.png
    ├── anomaly_scores.png
    ├── suspicious_students.png
    ├── alpha_experiment.png
    ├── hyperparameter_results.png
    └── attendance_over_time.png
```

---

## 17. Project Structure

```
Smart_Attendance_Anomaly_Detector/
├── app.py                    # Streamlit web application
├── run_project.py            # Master pipeline script
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
│
├── data/
│   ├── generate_dataset.py   # Dataset generator (6 anomaly types)
│   ├── attendance_dataset.csv # Full dataset (10,000 records)
│   ├── demo_attendance.csv   # Demo slice (1,000 records)
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Load, clean, split
│   ├── feature_engineering.py # Feature extraction and scaling
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Metrics and explanations
│   ├── visualization.py      # All plotting functions
│   └── models/
│       ├── __init__.py
│       ├── isolation_forest_model.py
│       ├── lof_model.py
│       └── hybrid_model.py   # Student-Designed Innovation
│
├── results/                  # Generated after running pipeline
│   ├── *.csv
│   ├── final_report.md
│   └── plots/
│
├── models/                   # Saved trained models
│   ├── isolation_forest.pkl
│   ├── lof_model.pkl
│   ├── hybrid_model.pkl
│   ├── scaler.pkl
│   └── train_stats.pkl
│
└── docs/
    ├── VIVA.md               # 30 Q&A for viva preparation
    └── RUBRIC_MAPPING.md     # Marks rubric mapping
```

---

## 18. Limitations

- **Synthetic data** – Real college attendance may have different patterns
- **No biometric verification** – Cannot determine WHO actually submitted attendance
- **Anomaly ≠ fraud** – Every flag requires human verification before action
- **Threshold sensitivity** – Different thresholds give different precision/recall trade-offs
- **Static model** – Needs retraining if attendance behaviour patterns change significantly
- **Possible bias** – Feature engineering assumptions may not generalise across institutions

---

## 19. Future Work

- **Real-time monitoring** – Stream detection as records arrive
- **Face recognition integration** – Biometric attendance marking
- **GPS geolocation** – Verify student's physical location
- **Graph-based detection** – Model device-student-IP relationships as a network
- **LSTM autoencoders** – Temporal sequence anomaly detection
- **SHAP explanations** – Mathematically grounded feature importance
- **Federated learning** – Train across institutions without sharing raw data

---

## 20. Viva Questions

See [`docs/VIVA.md`](docs/VIVA.md) for 30 questions and answers.

---

## 21. Ethical Considerations

- All data in this project is **synthetic**. No real student data was used.
- Anomaly flags **do not prove misconduct**. Human review is mandatory.
- False positives are inevitable — the threshold controls this trade-off.
- Real deployments must comply with data protection laws (e.g., GDPR).
- Students must be informed that attendance systems use anomaly detection.
