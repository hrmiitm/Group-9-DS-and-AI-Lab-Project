"""
eval.py — Fraud Job Posting Classifier Evaluation
===================================================
Evaluates a saved RoBERTa model on the held-out test set and supports
single-posting inference.

Usage:
    # Full test set evaluation with plots
    python src/eval.py \\
        --model_dir models/roberta-focal-best \\
        --data_path data/fake_job_postings.csv

    # Full evaluation without saving plots
    python src/eval.py \\
        --model_dir models/roberta-focal-best \\
        --data_path data/fake_job_postings.csv \\
        --no_plots

    # Single posting inference demo
    python src/eval.py \\
        --model_dir models/roberta-focal-best \\
        --infer
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Allow imports from src/utils when running as script ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    build_input_text,
    load_and_prepare_data,
    build_hf_datasets,
    sweep_thresholds,
    print_target_summary,
    TARGETS,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_dir: str):
    """
    Load model, tokenizer, and inference config from a saved model directory.

    Args:
        model_dir: Path to directory containing model.safetensors,
                   tokenizer files, and inference_config.json.

    Returns:
        Tuple of (model, tokenizer, config dict).

    Raises:
        FileNotFoundError: If inference_config.json is missing.
    """
    config_path = os.path.join(model_dir, "inference_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"inference_config.json not found in {model_dir}. "
            f"Ensure the model was saved with train.py."
        )

    with open(config_path) as f:
        cfg = json.load(f)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"✅ Model loaded from  : {model_dir}")
    print(f"   Final threshold    : {cfg['final_threshold']}")
    print(f"   Val F1             : {cfg['val_metrics']['f1_fraud']:.4f}")
    return model, tokenizer, cfg


# ── Inference ─────────────────────────────────────────────────────────────────
def get_predictions(model, dataset):
    """
    Run model inference on a HuggingFace Dataset using a lightweight
    Trainer wrapper (no training, predict only).

    Args:
        model  : Loaded AutoModelForSequenceClassification in eval mode.
        dataset: HuggingFace Dataset with input_ids, attention_mask, labels.

    Returns:
        Tuple of (probs, labels) as numpy arrays.
        probs  — fraud class probabilities, shape [N].
        labels — ground truth binary labels, shape [N].
    """
    tmp_args    = TrainingArguments(
        output_dir="/tmp/eval_tmp",
        report_to ="none",
        seed      =42,
    )
    tmp_trainer = Trainer(model=model, args=tmp_args)
    output      = tmp_trainer.predict(dataset)
    probs       = torch.softmax(
        torch.tensor(output.predictions), dim=-1,
    ).numpy()[:, 1]
    labels      = output.label_ids
    return probs, labels


def predict_fraud(job_posting: dict, model, tokenizer, cfg: dict) -> dict:
    """
    Predict fraud probability for a single job posting.

    Args:
        job_posting: Dict with keys matching TEXT_COLS and STRUCT_COLS
                     defined in utils/data.py. Missing keys default to ''.
        model      : Loaded AutoModelForSequenceClassification in eval mode.
        tokenizer  : Loaded AutoTokenizer.
        cfg        : Inference config dict (must contain 'final_threshold'
                     and 'max_seq_len').

    Returns:
        Dict with:
            fraud_probability (float): Model confidence the posting is fraud.
            prediction        (str)  : 'FRAUDULENT' or 'LEGITIMATE'.
            threshold_used    (float): Threshold applied to make the decision.
    """
    text   = build_input_text(job_posting)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["max_seq_len"],
        padding="max_length",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    prob_fraud = torch.softmax(logits, dim=-1)[0, 1].item()
    prediction = "FRAUDULENT" if prob_fraud >= cfg["final_threshold"] else "LEGITIMATE"

    return {
        "fraud_probability": round(prob_fraud, 4),
        "prediction"       : prediction,
        "threshold_used"   : cfg["final_threshold"],
    }


# ── Full evaluation ───────────────────────────────────────────────────────────
def evaluate(model_dir: str, data_path: str, save_plots: bool = True):
    """
    Run full test set evaluation, print the Mahfouz target summary,
    optionally save diagnostic plots, and write test_results.json.

    Args:
        model_dir : Path to saved model directory.
        data_path : Path to fake_job_postings.csv.
        save_plots: If True, save confusion matrix + ROC + PR curves.

    Returns:
        Tuple of (metrics dict, probs array, labels array).
    """
    # Load model
    model, tokenizer, cfg = load_model(model_dir)

    # Load test data — reproduce the same 70/15/15 split as training
    _, val_df, test_df = load_and_prepare_data(data_path)
    _, _, test_ds      = build_hf_datasets(val_df, val_df, test_df, tokenizer)
    # Note: train_df and val_df are placeholders here; only test_ds is used

    # Predictions
    probs, labels = get_predictions(model, test_ds)
    threshold     = cfg["final_threshold"]
    preds_final   = (probs >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "f1_fraud"       : f1_score(labels,        preds_final, pos_label=1, zero_division=0),
        "recall_fraud"   : recall_score(labels,    preds_final, pos_label=1, zero_division=0),
        "precision_fraud": precision_score(labels, preds_final, pos_label=1, zero_division=0),
        "roc_auc"        : roc_auc_score(labels,   probs),
        "mcc"            : matthews_corrcoef(labels, preds_final),
        "avg_precision"  : average_precision_score(labels, probs),
    }

    # ── Classification report ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" TEST RESULTS — Threshold {threshold}")
    print(f"{'='*60}")
    print(classification_report(
        labels, preds_final,
        target_names=["Legitimate", "Fraudulent"],
    ))

    # ── Mahfouz target summary ─────────────────────────────────────────────────
    print_target_summary(metrics, threshold)

    # ── Threshold sweep — check if any threshold meets all constraints ─────────
    sweep_df = sweep_thresholds(probs, labels)
    valid    = sweep_df[
        (sweep_df["precision"] >= TARGETS["precision_fraud"]) &
        (sweep_df["recall"]    >= TARGETS["recall_fraud"])
    ]
    if not valid.empty:
        best_row = valid.loc[valid["f1"].idxmax()]
        print(f"\n  ✅ Best threshold meeting ALL constraints: {best_row['threshold']}")
        print(f"     F1={best_row['f1']:.4f} | Recall={best_row['recall']:.4f} | "
              f"Precision={best_row['precision']:.4f}")
    else:
        print("\n  ⚠️  No single threshold satisfies all constraints on this test set.")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if save_plots:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Confusion matrix
        cm = confusion_matrix(labels, preds_final)
        ConfusionMatrixDisplay(
            cm, display_labels=["Legit", "Fraud"],
        ).plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title(f"Confusion Matrix (t={threshold})")

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        axes[1].plot(fpr, tpr, color="steelblue",
                     label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        # Precision-Recall curve
        prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
        axes[2].plot(rec_curve, prec_curve, color="coral",
                     label=f"AP = {metrics['avg_precision']:.4f}")
        axes[2].axhline(
            TARGETS["precision_fraud"], color="gray", linestyle="--",
            alpha=0.5, label=f"Precision target ({TARGETS['precision_fraud']})",
        )
        axes[2].axvline(
            TARGETS["recall_fraud"], color="gray", linestyle=":",
            alpha=0.5, label=f"Recall target ({TARGETS['recall_fraud']})",
        )
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve")
        axes[2].legend(fontsize=8)

        plt.suptitle("RoBERTa Fraud Classifier — Test Set", fontweight="bold")
        plt.tight_layout()
        plot_path = os.path.join(model_dir, "eval_plots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\n  📊 Plots saved to: {plot_path}")
        plt.show()

    # ── Save test results ──────────────────────────────────────────────────────
    results_path = os.path.join(model_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)
    print(f"  💾 Results saved to: {results_path}")

    return metrics, probs, labels


# ── Inference demo ────────────────────────────────────────────────────────────
def run_inference_demo(model_dir: str):
    """
    Load a saved model and run inference on a hardcoded sample posting
    that contains obvious fraud signals, for a quick sanity check.

    Args:
        model_dir: Path to saved model directory.
    """
    model, tokenizer, cfg = load_model(model_dir)

    sample_posting = {
        "title"              : "Work From Home Data Entry Specialist",
        "description"        : (
            "Earn $5000/week working from home. No experience needed. "
            "Send your bank details to get started immediately. "
            "Limited spots available. Apply now!"
        ),
        "requirements"       : "None required. Must have bank account.",
        "salary_range"       : "5000-20000",
        "employment_type"    : "Part-time",
        "required_experience": "Not Applicable",
        "company_profile"    : "",
        "location"           : "",
        "department"         : "",
        "benefits"           : "",
        "industry"           : "",
        "function"           : "",
        "required_education" : "",
    }

    result = predict_fraud(sample_posting, model, tokenizer, cfg)

    print(f"\n{'='*50}")
    print(f"  Job Posting Assessment (Inference Demo)")
    print(f"{'='*50}")
    print(f"  Fraud probability : {result['fraud_probability']:.1%}")
    print(f"  Prediction        : {result['prediction']}")
    print(f"  Threshold used    : {result['threshold_used']}")
    print(f"{'='*50}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RoBERTa fraud job posting classifier",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to saved model directory (output of train.py)",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to fake_job_postings.csv (required for full evaluation)",
    )
    parser.add_argument(
        "--infer", action="store_true",
        help="Run single-posting inference demo instead of full evaluation",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip generating and saving evaluation plots",
    )
    args = parser.parse_args()

    if args.infer:
        run_inference_demo(args.model_dir)
    elif args.data_path:
        evaluate(args.model_dir, args.data_path, save_plots=not args.no_plots)
    else:
        parser.error(
            "Provide --data_path for full evaluation or --infer for the demo."
        )
