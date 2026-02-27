"""
core/metrics.py — Evaluation metrics for binary segmentation.
Supplements your existing sklearn.metrics.jaccard_score usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SegmentationMetrics:
    """Container for all computed metrics."""
    iou: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray  # shape (2, 2): [[TN, FP], [FN, TP]]

    def to_dict(self) -> Dict[str, float]:
        return {
            "IoU": self.iou,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1-Score": self.f1,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> SegmentationMetrics:
    """
    Compute full segmentation metrics from two binary masks.
    Works with any shape — arrays are flattened internally.
    """
    yt = y_true.flatten().astype(np.uint8)
    yp = y_pred.flatten().astype(np.uint8)

    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    total = tp + fp + fn + tn

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    return SegmentationMetrics(
        iou=round(iou, 4),
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        confusion_matrix=cm,
    )