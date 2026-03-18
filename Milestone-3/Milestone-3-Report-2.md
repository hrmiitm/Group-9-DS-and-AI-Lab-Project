# Fake Job Listing Detection using Deep Learning and Agentic Generative AI

**Milestone 3 — Model Architecture**

---

## 1. Overview

This milestone documents the **model architecture selection**, the **rationale behind every design choice**, and the **end-to-end pipeline** — from raw CSV to trained model artifacts hosted on HuggingFace Hub.

All code referenced in this report lives under `src/` and is fully runnable from the repository root.

---

## 2. Model Architecture Selection

### 2.1 Chosen Architecture — RoBERTa-base

| Attribute | Detail |
|---|---|
| **Backbone** | `roberta-base` (125 M parameters) |
| **Head** | Linear classification head (`num_labels=2`) |
| **Fine-tuning** | Full (all layers trainable) |
| **Loss function** | Focal Loss (custom) |
| **LR scheduler** | Cosine annealing with warmup |
| **Hyperparameter search** | Optuna (25-trial Bayesian search) |
| **Early stopping** | Patience = 3 epochs on validation F1 |

The classifier is built on top of HuggingFace's `AutoModelForSequenceClassification`, adding a standard 2-class (`Legitimate` vs. `Fraudulent`) linear head on top of the [CLS] token representation.

### 2.2 Alternative Architectures Considered

| Architecture | Pros | Cons | Why Not Chosen |
|---|---|---|---|
| **BERT-base** | Strong contextual representations | Slightly lower performance than RoBERTa on downstream tasks | RoBERTa has better pre-training (dynamic masking, larger data) |
| **DistilBERT** | 40% faster, 60% smaller | ≈ 2-3% F1 drop on fraud class | We prioritise detection accuracy over inference speed |
| **LSTM / BiLSTM** | Good sequential modelling | Cannot capture long-range dependencies as well; lower F1 (~0.83) | Transformer self-attention is strictly superior for this task |
| **CNN (TextCNN)** | Fast inference | Limited context window; F1 ≈ 0.78 | Insufficient for 512-token job postings |
| **Logistic Regression + TF-IDF** | Simple baseline | No semantic understanding; F1 ≈ 0.73 | Baseline only |
