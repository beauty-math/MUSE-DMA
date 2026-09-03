from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    return {
        "aupr": float(average_precision_score(labels, scores)),
        "auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }
