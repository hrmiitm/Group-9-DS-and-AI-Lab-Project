"""
utils/data.py — Data loading, feature engineering, and dataset preparation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# ── Column definitions ────────────────────────────────────────────────────────
TEXT_COLS = [
    "title", "description", "requirements", "company_profile", "benefits",
]
STRUCT_COLS = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function",
    "has_company_logo",
]

MAX_SEQ_LEN = 512


def build_input_text(row: dict) -> str:
    """
    Concatenate structured metadata and free-text fields into a single
    transformer input sequence, separated by [SEP] tokens.

    Structured fields are prefixed with their column name for context.
    Empty, null, and 'nan' values are skipped.

    Args:
        row: dict-like row from a pandas DataFrame.

    Returns:
        Single string ready for tokenisation.
    """
    parts = []
    for col in STRUCT_COLS:
        val = str(row.get(col, "") or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(f"{col.replace('_', ' ').title()}: {val}")
    for col in TEXT_COLS:
        val = str(row.get(col, "") or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return " [SEP] ".join(parts)


def load_and_prepare_data(data_path: str):
    """
    Load the raw CSV, engineer the input text feature, and produce
    stratified 70 / 15 / 15 train / val / test splits.

    Args:
        data_path: Path to fake_job_postings.csv.

    Returns:
        Tuple of (train_df, val_df, test_df) — each a pandas DataFrame
        with columns ['input_text', 'label'].
    """
    df = pd.read_csv(data_path)
    df["input_text"] = df.apply(build_input_text, axis=1)
    df["label"]      = df["fraudulent"].astype(int)
    df_clean = df[["input_text", "label"]].dropna().reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df_clean, test_size=0.30, stratify=df_clean["label"], random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42,
    )

    for split, name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        n_fraud = split["label"].sum()
        print(
            f"{name:5s}: {len(split):5,} samples | "
            f"fraud={n_fraud} ({n_fraud / len(split) * 100:.1f}%)"
        )

    return train_df, val_df, test_df


def build_hf_datasets(train_df, val_df, test_df, tokenizer):
    """
    Tokenise all three splits and convert them to HuggingFace Dataset
    objects with PyTorch tensor format.

    Args:
        train_df, val_df, test_df: pandas DataFrames with 'input_text' and 'label'.
        tokenizer: A HuggingFace AutoTokenizer instance.

    Returns:
        Tuple of (train_ds, val_ds, test_ds) as HuggingFace Datasets.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
        )

    def to_hf_dataset(df_split):
        ds = Dataset.from_pandas(df_split.reset_index(drop=True))
        ds = ds.map(tokenize_fn, batched=True, batch_size=128)
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    return to_hf_dataset(train_df), to_hf_dataset(val_df), to_hf_dataset(test_df)


def get_last_checkpoint(output_dir: str):
    """
    Scan output_dir for HuggingFace Trainer checkpoint folders.
    Returns the path to the most recently modified checkpoint, or None
    if no checkpoints exist (i.e. starting fresh).

    Args:
        output_dir: Directory to scan.

    Returns:
        Absolute path string or None.
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"  📂 Resuming from checkpoint: {latest}")
    return latest
