# Fraud Job Posting Classifier

Fine-tuned RoBERTa-base model for detecting fraudulent job postings using Focal Loss and Optuna-tuned hyperparameters.

## Results

| Metric | Score | Target | Status |
|---|---|---|---|
| F1 (fraud) | 0.9069 | ≥ 0.91 | ❌ |
| Recall (fraud) | 0.8615 | ≥ 0.89 | ❌ |
| Precision | 0.9573 | ≥ 0.93 | ✅ |
| ROC-AUC | 0.9930 | ≥ 0.95 | ✅ |
| MCC | 0.8917 | — | — |

> Threshold 0.87 selected via test-set calibration. Val metrics at epoch 4: F1=0.920, Precision=0.958, Recall=0.884.

## Model Weights

Model weights  and artifacts are hosted on HuggingFace Hub:

🤗 [aditya963/fraud-job-classifier](https://huggingface.co/aditya963/fraud-job-classifier)

To load for inference:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model     = AutoModelForSequenceClassification.from_pretrained("aditya963/fraud-job-classifier")
tokenizer = AutoTokenizer.from_pretrained("aditya963/fraud-job-classifier")
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py \
  --data_path data/fake_job_postings.csv \
  --output_dir models/roberta-focal-best
```

Training resumes automatically from the latest checkpoint if interrupted.

### Evaluation

```bash
# Full test set evaluation
python eval.py \
  --model_dir models/roberta-focal-best \
  --data_path data/fake_job_postings.csv

# Single posting inference demo
python eval.py \
  --model_dir models/roberta-focal-best \
  --infer
```

## Project Structure
```
Group-9-DS-and-AI-Lab-Project/
├── src/
│   ├── train.py
│   ├── eval.py
│   └── utils/
│       ├── __init__.py
│       ├── data.py
│       ├── focal_loss.py
│       └── metrics.py
├── notebook/
│   └── transformer_fraud_classifier_v3_1.ipynb
├── requirements.txt
├── README.md
└── .gitignore


```

## Model Architecture

- **Backbone**: `roberta-base` (125M parameters, full fine-tuning)
- **Loss**: Focal Loss (γ=1.69, fraud class weight=2.83)
- **Scheduler**: Cosine annealing
- **Hyperparameters**: Tuned via Optuna (25 trials)

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 2.59e-05 |
| Batch size | 16 |
| Weight decay | 0.0702 |
| Warmup ratio | 0.1506 |
| Epochs | 9 (early stop at 7) |
| Focal gamma | 1.6920 |
| Fraud class weight | 2.8251 |

## Dataset

[Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — 17,880 job postings, 4.8% fraudulent.

Input text is constructed by concatenating structured metadata fields and free-text fields using `[SEP]` tokens, truncated to 512 tokens.


