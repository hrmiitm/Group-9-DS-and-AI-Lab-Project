"""
tools/tool_roberta.py
RoBERTa fraud job classifier — wraps aditya963/fraud-job-classifier from HuggingFace Hub.
NEW TOOL — not present in the original web-app.

Model: fine-tuned RoBERTa-base, 125M params, trained on EMSCAD dataset.
Threshold: 0.87 (calibrated on validation set).
Device: CPU (HuggingFace Spaces free tier).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

MODEL_ID = os.getenv("ROBERTA_MODEL_ID", "aditya963/fraud-job-classifier")
FRAUD_THRESHOLD = float(os.getenv("ROBERTA_THRESHOLD", "0.87"))


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load model once, cache forever (lru_cache on module singleton)."""
    try:
        from transformers import pipeline as hf_pipeline
        clf = hf_pipeline(
            "text-classification",
            model=MODEL_ID,
            device=-1,           # CPU
            truncation=True,
            max_length=512,
            top_k=None,          # return all labels
        )
        return clf, None
    except Exception as e:
        return None, str(e)


def classify_job_roberta(job_text: str, threshold: Optional[float] = None) -> dict:
    """
    Run the fine-tuned RoBERTa fraud classifier on a job posting text.

    INPUT : job_text (raw string, ideally title [SEP] description)
            threshold (optional float override, default 0.87)
    OUTPUT: {ok, data: {label, fraud_probability, is_fraud, threshold_used, model_id}}
    """
    if not job_text or not job_text.strip():
        return {"ok": False, "error": "job_text is required"}

    used_threshold = threshold if threshold is not None else FRAUD_THRESHOLD

    clf, load_err = _load_pipeline()
    if clf is None:
        return {"ok": False, "error": f"Model failed to load: {load_err}"}

    try:
        # Truncate to avoid token length issues (rough word-level limit)
        text = job_text.strip()[:3000]
        raw = clf(text)

        # raw is list of lists when top_k=None: [[{label, score}, ...]]
        scores = raw[0] if isinstance(raw[0], list) else raw

        fraud_score = 0.0
        legit_score = 0.0
        for item in scores:
            lbl = item["label"].upper()
            if "FAKE" in lbl or "FRAUD" in lbl or lbl in {"LABEL_1", "1"}:
                fraud_score = item["score"]
            elif "REAL" in lbl or "LEGIT" in lbl or lbl in {"LABEL_0", "0"}:
                legit_score = item["score"]

        # Fallback: if labels unknown pattern, sort by score desc
        if fraud_score == 0.0 and legit_score == 0.0 and scores:
            sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
            # Higher score = fraud (model trained: 1=fraud)
            fraud_score = scores[-1]["score"]  # last label in list = often LABEL_1
            legit_score = scores[0]["score"]

        is_fraud = fraud_score >= used_threshold

        return {
            "ok": True,
            "data": {
                "model_id":         MODEL_ID,
                "threshold_used":   used_threshold,
                "fraud_probability": round(fraud_score, 4),
                "legit_probability": round(legit_score, 4),
                "is_fraud":         is_fraud,
                "label":            "FAKE" if is_fraud else "REAL",
                "confidence":       "high" if max(fraud_score, legit_score) > 0.9 else "medium",
                "raw_scores":       scores,
            },
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
