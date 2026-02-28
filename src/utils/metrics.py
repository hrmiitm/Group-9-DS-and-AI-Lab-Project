"""
utils/metrics.py — Evaluation metrics for fraud classification.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Mahfouz targets ───────────────────────────────────────────────────────────
TARGETS = {
    "f1_fraud"       : 0.91,
    "recall_fraud"   : 0.89,
    "precision_fraud": 0.93,
    "roc_auc"        : 0.95,
}


def compute_metrics(eval_pred) -> dict:
    """
    HuggingFace Trainer-compatible metrics function.

    Sweeps decision thresholds from 0.05 to 0.95 in steps of 0.01 and
    selects the threshold that maximises F1 on the fraud class. No hard
    precision/recall constraints are applied here so metrics are never
    all-zero during early training epochs.

    Args:
        eval_pred: EvalPrediction namedtuple with fields
                   (predictions, label_ids) as numpy arrays.

    Returns:
        Dict with keys: f1_fraud, recall_fraud, precision_fraud,
        best_threshold, roc_auc.
    """
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    best = {"f1": 0.0, "recall": 0.0, "precision": 0.0, "threshold": 0.5}

    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        prec  = precision_score(labels, preds, pos_label=1, zero_division=0)
        rec   = recall_score(labels,    preds, pos_label=1, zero_division=0)
        f1    = f1_score(labels,        preds, pos_label=1, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "f1"       : f1,
                "recall"   : rec,
                "precision": prec,
                "threshold": round(float(t), 2),
            }

    return {
        "f1_fraud"       : best["f1"],
        "recall_fraud"   : best["recall"],
        "precision_fraud": best["precision"],
        "best_threshold" : best["threshold"],
        "roc_auc"        : roc_auc_score(labels, probs),
    }


def sweep_thresholds(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Sweep decision thresholds and return a DataFrame of per-threshold
    metrics. Used after training to find the best operating point.

    Args:
        probs : Fraud class probabilities, shape [N].
        labels: Ground truth binary labels, shape [N].

    Returns:
        DataFrame with columns: threshold, precision, recall, f1.
        Sorted by threshold ascending.
    """
    results = []
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        results.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
            "recall"   : recall_score(labels,    preds, pos_label=1, zero_division=0),
            "f1"       : f1_score(labels,        preds, pos_label=1, zero_division=0),
        })
    return pd.DataFrame(results)


def print_target_summary(metrics: dict, threshold: float) -> None:
    """
    Print a formatted Mahfouz target summary table.

    Args:
        metrics  : Dict containing f1_fraud, recall_fraud, precision_fraud,
                   roc_auc, mcc, avg_precision.
        threshold: Decision threshold used to compute the metrics.
    """
    print(f"{'='*60}")
    print(f" MAHFOUZ TARGET SUMMARY")
    print(f"{'='*60}")
    for key, target in TARGETS.items():
        val    = metrics.get(key, float("nan"))
        status = "✅" if val >= target else "❌"
        print(f"  {key:20s}: {val:.4f}  {status}  (target ≥ {target})")
    print(f"  {'mcc':20s}: {metrics.get('mcc', float('nan')):.4f}")
    print(f"  {'avg_precision':20s}: {metrics.get('avg_precision', float('nan')):.4f}")
    print(f"  {'threshold':20s}: {threshold}")

    all_met = all(metrics.get(k, 0) >= v for k, v in TARGETS.items())
    print(f"\n  {'🎉 ALL TARGETS MET' if all_met else '⚠️  Some targets not met'}")
    print(f"{'='*60}")
