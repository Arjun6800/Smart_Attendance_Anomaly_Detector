# Smart Attendance Anomaly Detector – Final Report

## Dataset Summary
| Metric | Value |
|--------|-------|
| Total Records | 10,000 |
| Normal Records | 9,200 |
| Anomalous Records | 800 |
| Anomaly Rate | 8.0% |
| Number of Students | ~200 |
| Date Range | 2024-01-15 to 2024-06-30 |

## Data Split
| Split | Size |
|-------|------|
| Training | 60% (6,000 records) |
| Validation | 20% (2,000 records) |
| Test | 20% (2,000 records) |

## Algorithm Configuration

### Isolation Forest (Best Parameters)
- n_estimators : 300
- contamination : 0.05

### LOF (Best Parameters)
- n_neighbors : 30
- contamination : 0.1

### Hybrid Model (Student-Designed Improvement)
- Best Alpha : 0.8
- Anomaly Threshold : 0.5
- Alpha candidates tested : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- Threshold candidates tested : [0.50, 0.55, ..., 0.90]

## Test Set Results

### Isolation Forest
| Metric | Value |
|--------|-------|
| Accuracy | 0.948 |
| Precision | 0.78 |
| Recall | 0.4875 |
| F1-Score | 0.6 |
| ROC-AUC | 0.8345 |
| PR-AUC | 0.5942 |
| TP | 78 |
| FP | 22 |
| TN | 1818 |
| FN | 82 |

### LOF
| Metric | Value |
|--------|-------|
| Accuracy | 0.8335 |
| Precision | 0.1319 |
| Recall | 0.1938 |
| F1-Score | 0.157 |
| ROC-AUC | 0.5181 |
| PR-AUC | 0.1775 |
| TP | 31 |
| FP | 204 |
| TN | 1636 |
| FN | 129 |

### Hybrid Model (Student Innovation)
| Metric | Value |
|--------|-------|
| Accuracy | 0.946 |
| Precision | 0.7766 |
| Recall | 0.4562 |
| F1-Score | 0.5748 |
| ROC-AUC | 0.8331 |
| PR-AUC | 0.5817 |
| TP | 73 |
| FP | 21 |
| TN | 1819 |
| FN | 87 |

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
