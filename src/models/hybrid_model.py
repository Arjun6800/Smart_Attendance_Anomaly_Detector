"""
src/models/hybrid_model.py
===========================
Student-Designed Improvement: Hybrid Attendance Anomaly Score
==============================================================

Motivation
----------
Both Isolation Forest (IF) and Local Outlier Factor (LOF) have complementary
strengths and weaknesses:

| Property              | Isolation Forest | LOF                  |
|-----------------------|-----------------|----------------------|
| Detects global outliers | Excellent      | Moderate             |
| Detects local outliers  | Moderate       | Excellent            |
| Robust to high dimensions | Good         | Degrades somewhat    |
| Sensitive to density  | No              | Yes                  |
| Training complexity   | O(n log n)      | O(n²)                |

Student-Designed Innovation
-----------------------------
Instead of selecting one algorithm, we COMBINE their normalised anomaly
scores using a weighted average:

    HybridScore = alpha * IF_score_norm + (1 - alpha) * LOF_score_norm

where:
  - IF_score_norm  = Isolation Forest score normalised to [0, 1]
  - LOF_score_norm = LOF score normalised to [0, 1]
  - alpha          = weighting parameter in [0, 1]

Rationale
---------
1. Averaging reduces the variance of each individual detector.
2. If IF misses a locally anomalous point, LOF may catch it (and vice-versa).
3. The alpha parameter allows data-driven weighting of each model.

Alpha Selection
---------------
Alpha is selected by MAXIMISING the F1-score on the VALIDATION SET only.
The test set is NEVER used during alpha selection (no data leakage).
Candidate values: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

Score Normalisation
-------------------
Raw scores from IF and LOF have different ranges and distributions.
We normalise each to [0, 1] using min-max scaling fitted on TRAINING data:

    score_norm = (score - min_train) / (max_train - min_train + ε)

The training min/max is saved and reused for val/test normalisation.

Threshold Selection
-------------------
After computing the hybrid score on the validation set, we try multiple
thresholds and select the one that maximises F1:

    prediction = 1 if HybridScore >= threshold else 0

Candidate thresholds: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

Limitations
-----------
1. The improvement is not guaranteed – on some datasets IF or LOF alone
   may outperform the hybrid.
2. Alpha tuning assumes the validation set is representative.
3. Equal feature representation must be ensured for a fair combination.
4. In high-noise scenarios, combining a bad model can degrade a good one.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# Candidate alpha values for tuning
ALPHA_CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Candidate threshold values for anomaly decision boundary
THRESHOLD_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


class HybridAnomalyDetector:
    """
    Student-Designed Hybrid Anomaly Detector.

    Combines Isolation Forest and LOF scores using a weighted average.
    Both alpha (weighting) and threshold are tuned on the validation set.

    Usage
    -----
    1. Call ``fit_transform_train(if_scores_train, lof_scores_train)``
       to normalise and store training score ranges.
    2. Call ``tune(if_scores_val, lof_scores_val, y_val)``
       to select best alpha and threshold using validation labels.
    3. Call ``predict(if_scores, lof_scores)`` on any split to get labels.
    4. Call ``score(if_scores, lof_scores)`` to get the raw hybrid score.
    """

    def __init__(self) -> None:
        self.alpha: float           = 0.5   # will be updated by tune()
        self.threshold: float       = 0.60  # will be updated by tune()
        self._if_min:   float       = 0.0
        self._if_max:   float       = 1.0
        self._lof_min:  float       = 0.0
        self._lof_max:  float       = 1.0
        self._alpha_results: list   = []
        self._threshold_results: list = []

    # ──────────────────────────────────────────────────────────────────────
    # Internal: Normalisation
    # ──────────────────────────────────────────────────────────────────────

    def _normalise_if(self, raw: np.ndarray) -> np.ndarray:
        """Normalise IF scores to [0, 1] using training min/max."""
        denom = (self._if_max - self._if_min) + 1e-9
        return np.clip((raw - self._if_min) / denom, 0.0, 1.0)

    def _normalise_lof(self, raw: np.ndarray) -> np.ndarray:
        """Normalise LOF scores to [0, 1] using training min/max."""
        denom = (self._lof_max - self._lof_min) + 1e-9
        return np.clip((raw - self._lof_min) / denom, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Fit normalisation range on training scores
    # ──────────────────────────────────────────────────────────────────────

    def fit_transform_train(
        self,
        if_scores_train:  np.ndarray,
        lof_scores_train: np.ndarray,
    ) -> np.ndarray:
        """
        Fit score normalisation on training data and return normalised hybrid scores.

        Parameters
        ----------
        if_scores_train  : Raw IF anomaly scores for training set.
        lof_scores_train : Raw LOF anomaly scores for training set.

        Returns
        -------
        np.ndarray
            Hybrid scores for training set (before threshold).
        """
        self._if_min  = float(if_scores_train.min())
        self._if_max  = float(if_scores_train.max())
        self._lof_min = float(lof_scores_train.min())
        self._lof_max = float(lof_scores_train.max())

        if_norm  = self._normalise_if(if_scores_train)
        lof_norm = self._normalise_lof(lof_scores_train)
        return self.alpha * if_norm + (1 - self.alpha) * lof_norm

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Tune alpha and threshold on validation set
    # ──────────────────────────────────────────────────────────────────────

    def tune(
        self,
        if_scores_val:  np.ndarray,
        lof_scores_val: np.ndarray,
        y_val:          np.ndarray,
    ) -> Tuple[float, float, list, list]:
        """
        Select the best alpha and threshold using VALIDATION labels only.

        No test data is used here – this is critical for a fair evaluation.

        Parameters
        ----------
        if_scores_val  : Raw IF anomaly scores on the validation set.
        lof_scores_val : Raw LOF anomaly scores on the validation set.
        y_val          : Ground-truth labels for validation set (0/1).

        Returns
        -------
        best_alpha     : float
        best_threshold : float
        alpha_results  : list of dicts with alpha / F1 pairs
        threshold_results : list of dicts with threshold / F1 pairs
        """
        if_norm  = self._normalise_if(if_scores_val)
        lof_norm = self._normalise_lof(lof_scores_val)

        # ── Alpha search ─────────────────────────────────────────────────
        alpha_results = []
        best_alpha    = 0.5
        best_f1_alpha = -1.0

        for alpha in ALPHA_CANDIDATES:
            hybrid = alpha * if_norm + (1 - alpha) * lof_norm
            # Use a fixed midpoint threshold for the alpha search
            preds = (hybrid >= 0.60).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            alpha_results.append({"alpha": alpha, "f1": round(f1, 4)})
            if f1 > best_f1_alpha:
                best_f1_alpha = f1
                best_alpha    = alpha

        self.alpha = best_alpha
        logger.info(
            "Best alpha selected: %.2f  (val F1 = %.4f)", best_alpha, best_f1_alpha
        )

        # ── Threshold search with best alpha ─────────────────────────────
        hybrid_val = best_alpha * if_norm + (1 - best_alpha) * lof_norm
        threshold_results = []
        best_threshold    = 0.60
        best_f1_thresh    = -1.0

        for thr in THRESHOLD_CANDIDATES:
            preds = (hybrid_val >= thr).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            threshold_results.append({"threshold": thr, "f1": round(f1, 4)})
            if f1 > best_f1_thresh:
                best_f1_thresh = f1
                best_threshold = thr

        self.threshold = best_threshold
        logger.info(
            "Best threshold selected: %.2f  (val F1 = %.4f)", best_threshold, best_f1_thresh
        )

        self._alpha_results     = alpha_results
        self._threshold_results = threshold_results
        return best_alpha, best_threshold, alpha_results, threshold_results

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Score / Predict on any set
    # ──────────────────────────────────────────────────────────────────────

    def score(
        self,
        if_scores:  np.ndarray,
        lof_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the hybrid anomaly score for a set of records.

        Uses the training normalisation ranges and the tuned alpha.

        Returns
        -------
        np.ndarray of float
            Hybrid scores in [0, 1].  Higher = more anomalous.
        """
        if_norm  = self._normalise_if(if_scores)
        lof_norm = self._normalise_lof(lof_scores)
        return self.alpha * if_norm + (1 - self.alpha) * lof_norm

    def predict(
        self,
        if_scores:  np.ndarray,
        lof_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Predict anomaly labels using the tuned alpha and threshold.

        Returns
        -------
        np.ndarray of int
            1 = anomaly,  0 = normal
        """
        hybrid = self.score(if_scores, lof_scores)
        return (hybrid >= self.threshold).astype(int)

    def predict_with_scores(
        self,
        if_scores:  np.ndarray,
        lof_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predictions, hybrid_scores)."""
        scores = self.score(if_scores, lof_scores)
        preds  = (scores >= self.threshold).astype(int)
        return preds, scores
