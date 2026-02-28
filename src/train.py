"""
train.py — Fraud Job Posting Classifier
========================================
Full fine-tuning of RoBERTa-base with Focal Loss and cosine LR schedule.
Hyperparameters sourced from Optuna run-17 (trial 18, epoch 4).

Usage:
    python src/train.py \\
        --data_path  data/fake_job_postings.csv \\
        --output_dir models/roberta-focal-best

Val metrics (epoch 4): F1=0.920 | Recall=0.884 | Precision=0.958 | AUC=0.996
Test metrics (t=0.87): F1=0.907 | Recall=0.862 | Precision=0.957 | AUC=0.993
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ── Allow imports from src/utils when running as script ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_and_prepare_data,
    build_hf_datasets,
    get_last_checkpoint,
    FocalLossTrainer,
    compute_metrics,
)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Best hyperparameters from Optuna run-17 ───────────────────────────────────
BEST_LR           = 2.5897267430435147e-05
BEST_BATCH        = 16
BEST_WD           = 0.07017434328133583
BEST_WARMUP       = 0.15058177139073298
BEST_EPOCHS       = 9
BEST_GAMMA        = 1.6919871410013687
BEST_FRAUD_WEIGHT = 2.8251219104371517
FINAL_THRESHOLD   = 0.87


def train(args):
    print(f"\n{'='*60}")
    print(f"  Fraud Classifier — Training")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, _ = load_and_prepare_data(args.data_path)

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    print(f"\nClass weights → Legit: {class_weights[0]:.3f} | Fraud: {class_weights[1]:.3f}")

    # ── Tokeniser + HF datasets ────────────────────────────────────────────────
    tokenizer          = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, _ = build_hf_datasets(train_df, val_df, val_df, tokenizer)
    # Note: test split is not used during training; val_df passed twice as placeholder

    # ── Checkpoint recovery ────────────────────────────────────────────────────
    last_checkpoint = get_last_checkpoint(args.output_dir)

    # ── Model ──────────────────────────────────────────────────────────────────
    load_from = last_checkpoint if last_checkpoint else MODEL_NAME
    model = AutoModelForSequenceClassification.from_pretrained(
        load_from,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    if last_checkpoint:
        print(f"  ⚡ Resuming from checkpoint: {last_checkpoint}")
    else:
        print(f"  🚀 Starting fresh from {MODEL_NAME}")

    # ── Training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = BEST_EPOCHS,
        per_device_train_batch_size = BEST_BATCH,
        per_device_eval_batch_size  = 32,
        gradient_accumulation_steps = 2,
        learning_rate               = BEST_LR,
        weight_decay                = BEST_WD,
        warmup_ratio                = BEST_WARMUP,
        lr_scheduler_type           = "cosine",
        fp16                        = torch.cuda.is_available(),
        max_grad_norm               = 1.0,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_fraud",
        greater_is_better           = True,
        save_total_limit            = 3,
        logging_steps               = 50,
        report_to                   = "none",
        seed                        = 42,
    )
    # Inject focal params — read by FocalLossTrainer.compute_loss via getattr()
    training_args.focal_gamma        = BEST_GAMMA
    training_args.fraud_class_weight = BEST_FRAUD_WEIGHT

    print(f"\nHyperparameters:")
    print(f"  LR           : {BEST_LR:.2e}")
    print(f"  Batch        : {BEST_BATCH}")
    print(f"  Weight decay : {BEST_WD:.4f}")
    print(f"  Warmup ratio : {BEST_WARMUP:.2f}")
    print(f"  Epochs       : {BEST_EPOCHS}")
    print(f"  Focal gamma  : {BEST_GAMMA:.4f}")
    print(f"  Fraud weight : {BEST_FRAUD_WEIGHT:.4f}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = FocalLossTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.class_weights = class_weights  # consumed by FocalLossTrainer.compute_loss

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n✅ Trainer configured — starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("\n✅ Training complete")

    # ── Save model + tokeniser ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)      # saves best epoch (load_best_model_at_end=True)
    tokenizer.save_pretrained(args.output_dir)

    # ── Save inference config ──────────────────────────────────────────────────
    inference_config = {
        "model_name"         : MODEL_NAME,
        "fine_tuning"        : "full",
        "loss_function"      : "FocalLoss",
        "focal_gamma"        : BEST_GAMMA,
        "fraud_class_weight" : BEST_FRAUD_WEIGHT,
        "lr_scheduler"       : "cosine",
        "max_seq_len"        : 512,
        "final_threshold"    : FINAL_THRESHOLD,
        "val_metrics"        : {
            "f1_fraud"       : 0.9200,
            "recall_fraud"   : 0.8846,
            "precision_fraud": 0.9583,
            "roc_auc"        : 0.9962,
        },
        "test_metrics"       : {
            "f1_fraud"       : 0.9069,
            "recall_fraud"   : 0.8615,
            "precision_fraud": 0.9573,
            "roc_auc"        : 0.9930,
            "mcc"            : 0.8917,
        },
    }
    with open(os.path.join(args.output_dir, "inference_config.json"), "w") as f:
        json.dump(inference_config, f, indent=2)

    # ── Save training summary ──────────────────────────────────────────────────
    best_metric = trainer.state.best_metric
    best_epoch  = None
    for log in trainer.state.log_history:
        if log.get("eval_f1_fraud") == best_metric:
            best_epoch = log.get("epoch")
            break

    summary = {
        "saved_at"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir"      : args.output_dir,
        "best_epoch"     : best_epoch,
        "best_f1_val"    : best_metric,
        "hyperparameters": {
            "learning_rate"      : BEST_LR,
            "batch_size"         : BEST_BATCH,
            "weight_decay"       : BEST_WD,
            "warmup_ratio"       : BEST_WARMUP,
            "num_train_epochs"   : BEST_EPOCHS,
            "focal_gamma"        : BEST_GAMMA,
            "fraud_class_weight" : BEST_FRAUD_WEIGHT,
        },
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Artifacts saved to : {args.output_dir}")
    print(f"   Best epoch         : {best_epoch}")
    print(f"   Best F1 (val)      : {best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RoBERTa fraud job posting classifier",
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to fake_job_postings.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/roberta-focal-best",
        help="Directory to save model checkpoints and artifacts",
    )
    args = parser.parse_args()
    train(args)
