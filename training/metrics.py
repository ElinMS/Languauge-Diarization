"""
metrics.py
──────────
Evaluation metrics for language boundary detection.

Boundary Detection Metrics
──────────────────────────
  Precision / Recall / F1  — a predicted boundary is "correct" if it
  falls within ±tolerance seconds of a ground-truth boundary.

Frame-level Language ID Metrics
────────────────────────────────
  Accuracy, macro-F1 per language.

Diarization Error Rate (DER-inspired)
──────────────────────────────────────
  Language Error Rate (LER): fraction of frames with wrong language ID.
"""

from typing import List, Optional
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
#  Boundary F1
# ─────────────────────────────────────────────────────────────────────────────

def boundary_precision_recall_f1(
    pred_times:  List[float],
    gt_times:    List[float],
    tolerance_s: float = 0.2,
) -> dict:
    """
    Compute boundary Precision / Recall / F1 with a ±tolerance window.

    Parameters
    ----------
    pred_times   : list of predicted switch times (seconds)
    gt_times     : list of ground-truth switch times (seconds)
    tolerance_s  : match window in seconds (default 200 ms)

    Returns
    -------
    dict with precision, recall, f1
    """
    if not gt_times and not pred_times:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_times:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred_times:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    matched_gt   = set()
    matched_pred = set()

    for pi, pt in enumerate(pred_times):
        for gi, gt in enumerate(gt_times):
            if gi in matched_gt:
                continue
            if abs(pt - gt) <= tolerance_s:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    precision = tp / len(pred_times) if pred_times else 0.0
    recall    = tp / len(gt_times)   if gt_times   else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_boundary_metrics(
    all_pred: List[List[float]],
    all_gt:   List[List[float]],
    tolerance_s: float = 0.2,
) -> dict:
    """Average boundary metrics across a batch / epoch."""
    p_list, r_list, f_list = [], [], []
    for pred, gt in zip(all_pred, all_gt):
        m = boundary_precision_recall_f1(pred, gt, tolerance_s)
        p_list.append(m["precision"])
        r_list.append(m["recall"])
        f_list.append(m["f1"])
    return {
        "boundary_precision": float(np.mean(p_list)),
        "boundary_recall":    float(np.mean(r_list)),
        "boundary_f1":        float(np.mean(f_list)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Frame-level language accuracy and Language Error Rate
# ─────────────────────────────────────────────────────────────────────────────

def frame_language_metrics(
    pred_labels: np.ndarray,   # (N_frames,) predicted language IDs
    true_labels: np.ndarray,   # (N_frames,) ground-truth language IDs
    ignore_index: int = -1,
) -> dict:
    """
    Frame-level language identification metrics.

    Parameters
    ----------
    pred_labels  : predicted per-frame language IDs
    true_labels  : ground-truth per-frame language IDs
    ignore_index : frames with this label are ignored (padding)

    Returns
    -------
    dict with frame_accuracy, macro_f1, language_error_rate
    """
    valid = true_labels != ignore_index
    pred  = pred_labels[valid]
    true  = true_labels[valid]

    if len(true) == 0:
        return {"frame_accuracy": 0.0, "macro_f1": 0.0, "language_error_rate": 1.0}

    acc  = accuracy_score(true, pred)
    mf1  = f1_score(true, pred, average="macro", zero_division=0)
    ler  = 1.0 - acc   # Language Error Rate

    return {
        "frame_accuracy":      float(acc),
        "macro_f1":            float(mf1),
        "language_error_rate": float(ler),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Running meter for epoch aggregation
# ─────────────────────────────────────────────────────────────────────────────

class MetricMeter:
    """Accumulates scalar values and computes mean."""

    def __init__(self):
        self._sums:   dict = {}
        self._counts: dict = {}

    def update(self, values: dict, n: int = 1):
        for k, v in values.items():
            if k not in self._sums:
                self._sums[k]   = 0.0
                self._counts[k] = 0
            self._sums[k]   += float(v) * n
            self._counts[k] += n

    def mean(self) -> dict:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def reset(self):
        self._sums.clear()
        self._counts.clear()
