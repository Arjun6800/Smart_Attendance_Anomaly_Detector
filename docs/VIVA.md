# Viva Q&A – Smart Attendance Anomaly Detector
## 30 Questions and Answers for Undergraduate Viva Voce

---

### 1. What is anomaly detection?

Anomaly detection is the process of identifying data points that behave significantly differently from the expected or "normal" pattern. These unusual points are called anomalies or outliers. In our project, anomalies are attendance records that look suspicious compared to normal attendance behaviour.

---

### 2. Why is attendance anomaly detection useful?

Traditional attendance systems only record whether a student was present or absent. They cannot detect:
- Proxy attendance (friend marks attendance on behalf of another)
- Burst attendance fraud (manipulating the system multiple times in seconds)
- Unusual device/IP patterns suggesting system manipulation
- Students suddenly deviating from their historical attendance behaviour

An AI-based system can automatically flag these patterns for human review.

---

### 3. Why did you choose Isolation Forest?

Isolation Forest is excellent for:
- High-dimensional data (many features)
- Fast training: O(n log n)
- Detecting global outliers (points very different from the whole dataset)
- Not needing to define what "normal" looks like explicitly

It is well-suited to attendance data where anomalies can appear across many different feature dimensions simultaneously.

---

### 4. How does Isolation Forest work?

Isolation Forest builds many random **Isolation Trees**. In each tree:
1. A feature is randomly selected.
2. A random split value is chosen between the feature's min and max.
3. Data is recursively partitioned until each point is isolated.

**Key insight:** Anomalies are rare and unusual, so they end up in short, isolated branches. Normal points take longer to isolate because they cluster together. The anomaly score is based on the average path length across all trees.

---

### 5. What is contamination in Isolation Forest?

`contamination` tells the algorithm what fraction of the data we expect to be anomalous. For example, `contamination=0.08` means we expect 8% of records to be anomalies. sklearn uses this to set the decision threshold that separates anomalies from normal records in the `predict()` output.

---

### 6. Why did you choose Local Outlier Factor (LOF)?

LOF detects **local** anomalies — points that are unusual relative to their immediate neighbourhood, even if they would look normal from a global perspective. Attendance anomalies can be locally unusual (e.g., a student with a very different pattern from nearby similar students), which is where LOF excels.

---

### 7. How does LOF work?

LOF compares the **local density** of each point to the density of its k-nearest neighbours:
1. For each point, find k nearest neighbours.
2. Compute **Local Reachability Density (LRD)** — inverse of average reachability distance.
3. LOF score = ratio of neighbours' LRD to the point's LRD.

If LOF ≈ 1 → normal density (normal point). If LOF >> 1 → much sparser than neighbours → anomaly.

---

### 8. What is local density in LOF?

Local density measures how tightly packed the points are in the neighbourhood of a given point. A high local density means the point is surrounded by many close neighbours (normal region). A low local density means the point is in a sparse region (possible anomaly).

---

### 9. What is the difference between LOF and Isolation Forest?

| Property | Isolation Forest | LOF |
|----------|-----------------|-----|
| Type | Global, tree-based | Local, density-based |
| Complexity | O(n log n) | O(n²) |
| Detects | Global outliers | Local outliers |
| Novelty | Built-in (fit+predict) | Needs `novelty=True` |
| Best use | High-dim data | Dense clusters with local anomalies |

---

### 10. Why not use normal classification (e.g., logistic regression)?

Classification requires labelled training data (knowing which records are fraudulent beforehand). In a real attendance system, we rarely have reliable ground-truth fraud labels. Our project uses **unsupervised** anomaly detection — we only use labels for evaluation, not for training. This is more realistic.

---

### 11. Why is this an unsupervised problem?

Because in the real world:
- We do not know which attendance records are fraudulent before running the system.
- Labelling every record as fraud/not-fraud is expensive and error-prone.
- New fraud patterns may not exist in historical labelled data.

Unsupervised methods learn what "normal" looks like and flag deviations, without needing prior fraud labels.

---

### 12. How did you generate anomalies?

We created 6 types of injected anomalies:
1. **Proxy** – Same device/IP used by many students; very short gap between records
2. **Time Anomaly** – Attendance at midnight or 3 AM
3. **Location Anomaly** – Student attends in a completely different room than usual
4. **Behavioural Anomaly** – Sudden change in attendance rate from historical norm
5. **Burst** – Multiple check-ins within seconds
6. **Historical Deviation** – Large deviation from long-term attendance average

Each anomaly was randomly selected and injected with probability 8% (`ANOMALY_RATE = 0.08`).

---

### 13. What is data leakage?

Data leakage occurs when information from the test/validation set (or the target label) accidentally influences the model training. This makes performance metrics unrealistically optimistic because the model has "seen" the test data.

---

### 14. How did you prevent data leakage?

- The `is_anomaly` label column was **separated before any preprocessing** and never given to the ML models during training.
- **RobustScaler** was fitted only on training data; validation and test data were transformed using the training scaler.
- **Feature engineering statistics** (student profiles, device/IP sharing counts) were computed from training data only and applied to validation/test.
- **Hyperparameter tuning** used only the validation set, never the test set.
- **Alpha and threshold tuning** for the hybrid model also used only the validation set.

---

### 15. Why did you create a validation set?

The validation set serves two purposes:
1. **Hyperparameter selection** – choosing best `n_estimators`, `n_neighbors`, `contamination` without touching the test set.
2. **Threshold and alpha tuning** – selecting the best anomaly threshold and hybrid alpha value.

Using a separate validation set ensures the test set remains "unseen" for truly unbiased final evaluation.

---

### 16. Why is accuracy insufficient for anomaly detection?

Our dataset has ~8% anomalies. A naive model that classifies everything as "normal" would achieve 92% accuracy — but would detect 0 anomalies (completely useless). Therefore, we must use:
- **Precision** and **Recall** to separately measure false positives and false negatives.
- **F1-score** as a combined measure.
- **ROC-AUC** and **PR-AUC** for threshold-independent evaluation.

---

### 17. What is precision?

Precision = TP / (TP + FP)

Of all records **flagged as anomalies**, what fraction **actually are anomalies**?
High precision means fewer false alarms.

---

### 18. What is recall?

Recall = TP / (TP + FN)

Of all **actual anomalies**, what fraction did the model **correctly detect**?
High recall means fewer missed anomalies.

---

### 19. What is F1-score?

F1 = 2 × (Precision × Recall) / (Precision + Recall)

The harmonic mean of precision and recall. It gives a single balanced metric — useful when we want to balance both false positives and false negatives.

---

### 20. What is ROC-AUC?

ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at different thresholds. AUC (Area Under Curve) summarises the curve as a single number between 0 and 1:
- AUC = 1.0 → perfect classifier
- AUC = 0.5 → random classifier

ROC-AUC is threshold-independent and works well for balanced datasets.

---

### 21. What is PR-AUC?

PR-AUC (Precision-Recall Area Under Curve) is the area under the Precision-Recall curve. Unlike ROC-AUC, it is more informative for **imbalanced datasets** (like ours with 8% anomalies) because it focuses specifically on the positive (anomaly) class performance.

---

### 22. What is your student-designed improvement?

The **Hybrid Anomaly Score** is the student-designed innovation:

```
HybridScore = α × IF_score_norm + (1 - α) × LOF_score_norm
```

Both scores are first normalised to [0, 1] using training statistics. The weighting parameter α is tuned on the validation set to maximise F1-score. An optimal anomaly threshold is also tuned on the validation set. The idea is to combine the strengths of both algorithms.

---

### 23. Why combine Isolation Forest and LOF?

- IF is better at global anomalies (rare extreme points)
- LOF is better at local anomalies (dense clusters with local deviations)
- Many real attendance anomalies may be a mix of both types
- Combining models can reduce variance and improve robustness

This is similar to ensemble learning in supervised ML (e.g., Random Forest combining many trees).

---

### 24. How did you select alpha?

Alpha was selected by evaluating all candidate values [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] on the **validation set** and picking the value that maximised the F1-score. The test set was NOT used for this selection.

---

### 25. What are the limitations of your system?

1. **Synthetic dataset** – Results may differ on real attendance data.
2. **No biometric verification** – Cannot distinguish who actually marked attendance.
3. **Anomaly ≠ fraud** – False positives are always possible.
4. **Threshold sensitivity** – Performance changes with threshold; wrong threshold → too many/few flags.
5. **Static model** – The model does not update as attendance patterns change over time.
6. **Feature engineering assumptions** – Student profiles are based on historical averages which may not generalise.

---

### 26. What happens if the dataset changes (distribution shift)?

If new attendance patterns emerge that were not in the training data (e.g., COVID-related attendance changes, new campus locations), the model should be **retrained** on updated data. Otherwise, the anomaly scores may become miscalibrated. This is the problem of **distribution shift** or **concept drift**.

---

### 27. Can an anomaly always be considered cheating?

No. An anomaly flag indicates an unusual pattern, not proof of cheating. Possible legitimate reasons for anomalies:
- Student attended from a different room due to class rescheduling
- System error in recording time/location
- Student used a borrowed device
- Student moved to a new campus zone

Human review is always required before taking any action.

---

### 28. How would you deploy this system in a real college?

1. Connect to the college's attendance database (with proper data privacy measures).
2. Run the anomaly detection pipeline periodically (e.g., weekly).
3. Send flagged records to a human reviewer (attendance officer/faculty).
4. Reviewer investigates and confirms or dismisses each flag.
5. Confirmed fraud cases update a feedback loop for model improvement.
6. Ensure compliance with data protection regulations (e.g., GDPR, local laws).

---

### 29. What are possible false positives?

Examples of legitimate records that might be flagged:
- Student attending a make-up class at an unusual time (triggers time anomaly)
- Student using a shared lab computer (triggers device-sharing flag)
- Student on a different campus floor (triggers location anomaly)
- Student with unusually perfect attendance (triggers high-rate flag)

This is why human review is mandatory before any action.

---

### 30. What would you improve in the future?

1. **Real data** – Partner with an institution for anonymised real attendance data.
2. **Face recognition** – Add biometric verification during attendance marking.
3. **Geolocation** – Verify physical location via mobile GPS.
4. **Deep learning** – Use LSTM autoencoders for temporal sequence anomaly detection.
5. **Explainable AI (SHAP)** – Use SHAP values for mathematically grounded feature explanations.
6. **Graph-based analysis** – Model student-device-IP relationships as a graph to detect proxy networks.
7. **Real-time streaming** – Apply anomaly detection in real time as records are submitted.
8. **Adaptive thresholding** – Automatically adjust thresholds as data distribution changes.
