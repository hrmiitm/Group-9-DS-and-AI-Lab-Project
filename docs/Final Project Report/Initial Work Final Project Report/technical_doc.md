# Technical Documentation

**Project:** FraudGuard — Fake Job Listing Detection
**Version:** Final (Milestone 6)
**Last Updated:** 2026-04-15

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Summary](#4-training-summary)
5. [Evaluation Summary](#5-evaluation-summary)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Deployment Details](#7-deployment-details)
8. [System Design Considerations](#8-system-design-considerations)
9. [Error Handling and Monitoring](#9-error-handling-and-monitoring)
10. [Reproducibility Checklist](#10-reproducibility-checklist)

---

## 1. Environment Setup

### 1.1 Python and OS Requirements

| Requirement | Value |
|---|---|
| Python version | 3.11+ |
| Operating system | Linux (Ubuntu 20.04+ recommended); macOS supported for inference; Windows not tested |
| GPU (training) | CUDA-capable GPU, minimum 16GB VRAM (tested on Google Colab T4, 15GB) |
| GPU (inference) | Optional — CPU inference supported but slower |
| RAM | 16GB minimum; 32GB recommended for full training run |

### 1.2 Core Dependencies

```
torch==2.11.0
transformers==5.5.4
datasets==4.8.4
scikit-learn==1.8.0
optuna==4.8.0
flask==3.1.3
langchain-openai==1.1.12
langchain-community==0.4.1
pydantic==2.12.5
requests==2.33.1
beautifulsoup4==4.14.3
ddgs==9.12.0
trafilatura==2.0.0
python-whois==0.9.6
email-validator==2.3.0
phonenumbers==9.0.27
pandas==3.0.2
numpy==2.4.4
```

### 1.3 Installation

**Full environment (training + web-app, GPU required):**
```bash
pip install -r requirements.txt
```

**Web-app only (no GPU required):**
```bash
pip install flask markupsafe langchain-openai langchain-community pydantic \
            requests beautifulsoup4 ddgs trafilatura python-whois \
            email-validator phonenumbers
```

**Model inference only:**
```bash
pip install transformers torch
```

### 1.4 Environment Variables

The web-app requires three environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | `""` | LLM API key (set to AIPipe token for OpenRouter proxy) |
| `OPENAI_BASE_URL` | No | `https://aipipe.org/openrouter/v1` | LLM base URL |
| `LLM_MODEL` | No | `openai/gpt-4o-mini` | Any OpenRouter model slug |
| `FLASK_SECRET_KEY` | No | `dev-change-in-prod` | Flask session secret (change in production) |

Set them before launching:
```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://aipipe.org/openrouter/v1"
export LLM_MODEL="openai/gpt-4o-mini"
```

---

## 2. Data Pipeline

### 2.1 Dataset Source

| Attribute | Value |
|---|---|
| Name | Fake Job Postings (EMSCAD) |
| Source | Kaggle: `shivamb/real-or-fake-fake-jobposting-prediction` |
| Total samples | 17,880 |
| Legitimate | 17,014 (95.16%) |
| Fraudulent | 866 (4.84%) |
| Imbalance ratio | ~20:1 |
| Feature count | 18 columns (5 free-text, 9 structured, 3 binary/categorical) |
| License | CC BY-SA 4.0 |

**Deduplication:** Text-based duplicates (same title + description) were removed, reducing training data from 17,880 to 15,787 unique samples. This prevents data leakage between splits.

### 2.2 Preprocessing Steps

**Step 1 — Missing Value Handling**
- All `NaN` values in text fields are replaced with empty strings (`""`).
- Missing values are deliberately **not imputed** — their absence is a fraud signal (e.g., empty `company_profile` strongly correlates with fraud).

**Step 2 — Structured Field Formatting**
Metadata fields are converted to key-value pairs:
```
"Location: US, NY, New York"
"Employment Type: Full-time"
"Has Company Logo: 1"
```
This preserves field name context so the transformer can learn metadata semantics.

**Step 3 — Text Concatenation**
All fields are joined using `[SEP]` as a delimiter:
```
Location: US, NY, New York [SEP] Employment Type: Full-time [SEP]
Has Company Logo: 1 [SEP] Software Engineer [SEP] We are seeking
a talented professional... [SEP] Bachelor's degree required...
```
Structured metadata is placed first to protect it from sequence truncation.

**Step 4 — Tokenization**
- Tokenizer: RoBERTa BPE tokenizer
- `max_length = 512`
- `truncation = True`
- `padding = 'max_length'`
- ~11% of samples exceed 512 tokens and are truncated.

**Step 5 — Data Splits**

| Split | Proportion | Samples | Fraud Samples |
|---|---|---|---|
| Training | 70% | 12,516 | ~606 |
| Validation | 15% | 2,682 | ~130 |
| Test | 15% | 2,682 | ~130 |

Split uses `sklearn.model_selection.train_test_split` with `stratify=df['label']` and `random_state=42`.

### 2.3 Key Features Used

All 18 fields from the EMSCAD schema are used:

| Type | Fields |
|---|---|
| Free-text | `title`, `description`, `requirements`, `company_profile`, `benefits` |
| Structured | `location`, `department`, `salary_range`, `employment_type`, `required_experience`, `required_education`, `industry`, `function` |
| Binary/categorical | `telecommuting`, `has_company_logo`, `has_questions` |

---

## 3. Model Architecture

### 3.1 Overview

The final model (`v3_1`) is a **fully fine-tuned RoBERTa-base** transformer with a linear binary classification head.

| Component | Description |
|---|---|
| Backbone | `roberta-base` (HuggingFace) |
| Task | Binary sequence classification (Legitimate=0, Fraudulent=1) |
| Parameters | ~125.5M (all trainable) |
| Fine-tuning | Full fine-tuning (no LoRA/adapters) |

### 3.2 Detailed Architecture

| Layer | Detail | Output Shape | Parameters |
|---|---|---|---|
| Token & Position Embeddings | BPE vocab (50,265 tokens) | Seq × 768 | ~38.6M |
| Transformer Encoder (×12) | 12 self-attention heads, head_dim=64 | Seq × 768 | ~84.9M |
| Feed-Forward (per block) | 768 → 3,072 → 768, GELU | Seq × 768 | ~4.7M/block |
| Pooler | Dense(768,768), tanh | 1 × 768 | ~590K |
| Dropout | 0.1 (hidden + attention) | — | — |
| Classification Head | Linear(768 → 2) | 1 × 2 | ~1.5K |
| **Total** | | | **~125.5M** |

### 3.3 Key Design Choices

**Why RoBERTa over BERT?**
RoBERTa was trained with dynamic masking (random masking at each epoch vs. static), larger mini-batches, and more data (160GB vs. 16GB for BERT). It consistently outperforms BERT on NLP benchmarks.

**Why full fine-tuning over LoRA?**
LoRA was tested (early experiments) but produced lower validation F1. With only ~12,500 training samples and 866 fraud examples, full fine-tuning allows all 125M parameters to adapt to domain-specific fraud signals.

**Why Focal Loss?**
Standard cross-entropy on a 20:1 imbalance trains the model to predict "legitimate" for everything and still achieve 95% accuracy. Focal Loss:
- Assigns higher penalty to minority-class errors (`alpha` weighting).
- Down-weights easy correctly-classified examples (`gamma` focusing parameter), forcing the model to concentrate on hard fraud examples.

**Focal Loss formula:**
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
Where `γ = 1.6920` (Optuna-tuned) and `α = [auto_legit_weight, 2.8251]`.

---

## 4. Training Summary

### 4.1 Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Weight decay on non-bias/LayerNorm params |
| Learning rate | ~2.59e-5 | Log-uniform Optuna search: 1e-5 to 5e-5 |
| LR Scheduler | Cosine annealing | Warmup ratio 0.1506 (Optuna) |
| Batch size | 16 | + gradient accumulation steps=2 (effective batch=32) |
| Max epochs | 9 (early stop at epoch 7) | Early stopping patience=5 |
| Max sequence length | 512 tokens | RoBERTa tokenizer |
| Hardware | Google Colab T4 GPU | Mixed precision FP16 |
| Gradient clipping | max_grad_norm=1.0 | Applied throughout |
| Weight decay | 0.0702 | L2 penalty on model weights |
| Focal gamma | 1.6920 | Optuna-tuned |
| Fraud class weight | 2.8251 | Optuna-tuned |
| HPO method | Optuna (25 trials) | Bayesian optimization |
| HPO objective | Max F1 s.t. Recall≥0.89, Precision≥0.93 | Hard floors enforced |

### 4.2 Version History

| Version | Base Model | Loss | HPO | Key Change |
|---|---|---|---|---|
| v1 | roberta-base | Weighted CE | Manual | Baseline |
| v2 | roberta-base | Focal γ=2.0 | Manual | Focal loss introduced |
| v3 | roberta-base | Focal γ=2.0 | Optuna 15T | First automated HPO |
| **v3_1 (FINAL)** | **roberta-base** | **Focal γ=1.69** | **Optuna 25T** | **Dynamic γ + class weight via HPO** |
| v4 | deberta-v3-base | Focal | Optuna | DeBERTa experiment (discontinued) |
| v5 | roberta-base | Focal γ=3.0 | Optuna 20T | Cosine LR, recall-targeted threshold |
| v5_synth | roberta-base | Focal γ=3.0 | Optuna 25T | Synthetic LLM data augmentation |

### 4.3 Training Artifacts

Artifacts are saved to Google Drive during Colab training and archived on HuggingFace Hub:

| Artifact | Description |
|---|---|
| `model.safetensors` | Fine-tuned RoBERTa weights (~500MB) |
| `config.json` | HuggingFace model architecture config |
| `tokenizer.json` | BPE vocabulary and merge rules |
| `inference_config.json` | Best threshold, metrics, hyperparameter snapshot |
| `test_results.json` | Final test-set metrics |
| `probs_test.npy` | Raw probability scores on test set |
| `labels_test.npy` | True labels for test set |

---

## 5. Evaluation Summary

### 5.1 Final Model Performance (v3_1, threshold=0.87)

| Metric | Target | Test Result | Status |
|---|---|---|---|
| F1 (fraud class) | ≥ 0.91 | **0.9069** | Narrow Miss (−0.003) |
| Recall (fraud class) | ≥ 0.89 | **0.8615** | Miss (−0.029) |
| Precision (fraud class) | ≥ 0.93 | **0.9573** | ✅ Met (+0.027) |
| ROC-AUC | ≥ 0.95 | **0.9930** | ✅ Met (+0.043) |
| MCC | Reported | **0.8917** | — |

### 5.2 Comparison Across Models

| Model | F1 (fraud) | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.83 | ~0.80 | ~0.86 | ~0.94 |
| TF-IDF + Random Forest | ~0.82 | ~0.78 | ~0.85 | ~0.93 |
| RoBERTa v1 (Weighted CE) | 0.8745 | 0.8300 | 0.9200 | 0.9874 |
| **RoBERTa v3_1 (Final)** | **0.9069** | **0.8615** | **0.9573** | **0.9930** |

### 5.3 Key Insights

- **ROC-AUC 0.993** demonstrates near-perfect probabilistic separation between fraud and legitimate postings.
- **Precision gap vs. recall gap:** The model prioritizes precision (0.957 vs. target 0.93) at the cost of recall (0.862 vs. target 0.89). This is the correct trade-off for a user-facing tool — false alarms destroy user trust more than missed frauds.
- **Validation vs. Test gap:** Validation F1 at best epoch was 0.920 vs. test F1 0.907. This ~1.3% gap is primarily due to the small test fraud set (130 samples), where individual misclassifications have an outsized metric impact.
- **Threshold effect:** Moving from default 0.5 to optimized 0.87 threshold added ~2-4 percentage points of F1 without any retraining.
- **Confusion matrix narrative:** At threshold 0.87, the model correctly identifies ~86% of all fraudulent postings (recall) while generating false fraud alerts for only ~4.3% of legitimate postings (1 - precision).

---

## 6. Inference Pipeline

### 6.1 Step-by-Step Flow

```
Input: job posting (dict with 14–18 fields)
    ↓
Step 1: Missing value handling
    - Replace None/NaN with ""
    ↓
Step 2: Structured field formatting
    - "Location: {value}", "Has Company Logo: {0|1}", etc.
    ↓
Step 3: Text concatenation
    - Structured fields first, then free-text fields
    - Fields joined with " [SEP] "
    ↓
Step 4: Tokenization
    - RoBERTa BPE tokenizer
    - max_length=512, truncation=True, padding='max_length'
    - Outputs: input_ids [512], attention_mask [512]
    ↓
Step 5: Model forward pass
    - RoBERTa encoder: 12 transformer layers → [CLS] token (768-dim)
    - Dropout (0.1)
    - Classification head: 768 → 2 logits
    ↓
Step 6: Softmax + threshold
    - P(fraud) = softmax(logits)[1]
    - Prediction = FRAUDULENT if P(fraud) >= 0.87, else LEGITIMATE
    ↓
Output: {'fraud_probability': float, 'prediction': str, 'threshold_used': float}
```

### 6.2 Inference Code Snippet

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("aditya963/fraud-job-classifier")
tokenizer = AutoTokenizer.from_pretrained("aditya963/fraud-job-classifier")
model.eval()

# Load inference config (threshold)
# inference_config.json is included in the HuggingFace Hub repo
THRESHOLD = 0.87  # from inference_config.json


def build_input_text(job: dict) -> str:
    """Convert a job posting dict into the model's expected input text."""
    structured = {
        "Location": job.get("location", ""),
        "Department": job.get("department", ""),
        "Salary Range": job.get("salary_range", ""),
        "Employment Type": job.get("employment_type", ""),
        "Required Experience": job.get("required_experience", ""),
        "Required Education": job.get("required_education", ""),
        "Industry": job.get("industry", ""),
        "Function": job.get("function", ""),
        "Has Company Logo": str(job.get("has_company_logo", "")),
    }
    free_text = {
        "Job Title": job.get("title", ""),
        "Company Profile": job.get("company_profile", ""),
        "Description": job.get("description", ""),
        "Requirements": job.get("requirements", ""),
        "Benefits": job.get("benefits", ""),
    }
    parts = []
    for k, v in {**structured, **free_text}.items():
        if v and str(v).strip():
            parts.append(f"{k}: {str(v).strip()}")
    return " [SEP] ".join(parts)


def predict_fraud(job_posting: dict) -> dict:
    """
    Args:
        job_posting: dict with keys: title, description, requirements,
                     company_profile, benefits, location, department,
                     salary_range, employment_type, required_experience,
                     required_education, industry, function, has_company_logo
    Returns:
        {'fraud_probability': float, 'prediction': str, 'threshold_used': float}
    """
    input_text = build_input_text(job_posting)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_fraud = torch.softmax(logits, dim=-1)[0][1].item()
    prediction = "FRAUDULENT" if prob_fraud >= THRESHOLD else "LEGITIMATE"
    return {
        "fraud_probability": round(prob_fraud, 4),
        "prediction": prediction,
        "threshold_used": THRESHOLD,
    }


# Example usage
example = {
    "title": "Work From Home Data Entry Specialist",
    "description": "Earn $500/day working from home. No experience needed.",
    "requirements": "None",
    "company_profile": "",
    "benefits": "Unlimited earning potential!",
    "location": "Remote",
    "salary_range": "500-1000",
    "employment_type": "Part-time",
    "has_company_logo": 0,
}

result = predict_fraud(example)
print(result)
# {'fraud_probability': 0.92, 'prediction': 'FRAUDULENT', 'threshold_used': 0.87}
```

---

## 7. Deployment Details

### 7.1 Model Hosting

The fine-tuned model is hosted on **HuggingFace Hub**:
- Repository: `aditya963/fraud-job-classifier`
- Includes: model weights, tokenizer, `inference_config.json`

Load for inference:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("aditya963/fraud-job-classifier")
tokenizer = AutoTokenizer.from_pretrained("aditya963/fraud-job-classifier")
```

### 7.2 Web-App Deployment

The Flask web-app runs locally. Launch:
```bash
export OPENAI_API_KEY="your-key"
python web-app/app.py
# → http://localhost:5000
```

The app does NOT require the RoBERTa model at runtime — it uses a separate LLM-based agentic pipeline for fraud investigation. The RoBERTa model is used for the core classification probability only.

**API Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/` | Home page (input form) |
| POST | `/analyze` | Submit a job for analysis |
| GET | `/results/<job_id>` | View analysis results |
| GET | `/results` | History of all analyses |
| GET | `/api/settings` | Get current LLM settings |
| POST | `/api/settings` | Update LLM settings |
| GET | `/api/result/<job_id>` | Raw JSON result |

### 7.3 Chrome Extension Deployment

Load unpacked in Chrome:
1. Go to `chrome://extensions`
2. Enable Developer Mode
3. Click "Load unpacked" → select `web-extension/` folder
4. Set Gemini API key via extension popup

The extension runs entirely in the browser — no server required.

---

## 8. System Design Considerations

### 8.1 Architecture Decisions

**Separation of concerns:** The web-app (`web-app/`) is fully self-contained and imports nothing from `testing_work/src/`. This prevents coupling between the production app and training code.

**Tool isolation:** Each verification tool (`web-app/tools/`) is a pure function returning `{ok: bool, data: {...}}`. `safe_call()` wraps every tool invocation, so exceptions in individual tools never crash the analysis pipeline.

**Session-based LLM config:** Users can override LLM settings via the Settings modal. Flask session stores overrides; environment variables take precedence to prevent proxy bypass.

**Incremental result writing:** `services/analyzer.py` writes progress to `results/<job_id>.json` after each pipeline step. If a step fails, the partial result is preserved and the error is surfaced in the UI.

### 8.2 Scalability

The current deployment is single-threaded (Flask dev server). For production:
- Use Gunicorn with worker processes: `gunicorn -w 4 web-app.app:app`
- Add a task queue (Celery + Redis) to handle long-running analyses asynchronously
- Cache DuckDuckGo results to reduce latency
- Use HuggingFace Inference Endpoints for RoBERTa API calls

### 8.3 Web-App ↔ Extension Interaction

The web-app and Chrome extension are **independent systems**. They do not communicate with each other:
- Web-app uses OpenRouter/AIPipe LLM backend
- Extension uses Google Gemini API directly from the browser

A future integration path would have the extension call the web-app API endpoint instead of Gemini directly, enabling full 12-tool analysis from the browser.

---

## 9. Error Handling and Monitoring

### 9.1 Web-App Error Handling

```
Every tool call:
  safe_call(fn, *args) → returns {ok: False, error: "..."} on any exception

Every pipeline step:
  try/except with _save_patch() → partial result preserved, error field set

LLM calls:
  Failed calls return "Inconclusive — tool did not return data." as inference text

File uploads:
  Extension whitelist (.pdf, .docx, .doc, .txt, .html, .md) enforced in routes/main.py
  MAX_UPLOAD_BYTES = 10MB (defined but not yet enforced — see GAPS_FIXED.md)
```

### 9.2 Edge Cases

| Edge Case | Handling |
|---|---|
| Missing company name | `infer_company_name()` extracts from description; tools skipped if still empty |
| Empty job description | Pipeline continues with available fields; report notes sparse input |
| LinkedIn URL blocked | `linkedin.py` returns `None`; user shown error message |
| API key missing | 402 error from LLM; surfaced in results page with error field |
| Tool timeout | `REQUEST_TIMEOUT=20s` per tool; returns `{ok: False, error: "timeout"}` |
| Duplicate analysis | Each submission generates a new UUID; no deduplication |

### 9.3 Latency

A full analysis (12 tools + 12 LLM inferences + 1 final report) takes approximately **30–90 seconds** depending on:
- LLM model speed (gpt-4o-mini ≈ 30s; larger models ≈ 90s)
- Tool response times (WHOIS lookups, DuckDuckGo searches)
- Network latency to external APIs

The results page is served after all steps complete (synchronous pipeline).

---

## 10. Reproducibility Checklist

Follow these steps exactly to reproduce all results from scratch:

### 10.1 Environment Setup

- [ ] Create a Python 3.11 virtual environment: `python3.11 -m venv venv && source venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### 10.2 Data Preparation

- [ ] Download dataset from Kaggle: `shivamb/real-or-fake-fake-jobposting-prediction`
- [ ] Place file at `data/raw/fake_job_postings.csv`
- [ ] Verify: 17,880 rows, 18 columns

### 10.3 Training (Google Colab)

- [ ] Open `notebook/transformer_fraud_classifier_v3_1.ipynb` in Google Colab
- [ ] **Update `DATA_PATH`** to point to your uploaded dataset (e.g., `/content/drive/MyDrive/data/fake_job_postings.csv`)
- [ ] **Update `OUTPUT_DIR`** to your desired model output path on Drive
- [ ] Ensure T4 GPU runtime is selected (Runtime → Change runtime type → T4 GPU)
- [ ] Mount Google Drive: run the drive mount cell
- [ ] Set `random_state=42` (already set in notebook)
- [ ] Run all cells sequentially
- [ ] Expected training time: ~2–4 hours for 9 epochs with 25 Optuna trials

### 10.4 Verification of Results

- [ ] Check `test_results.json` in model output directory
- [ ] Expected: F1 ≈ 0.907, Precision ≈ 0.957, Recall ≈ 0.862, AUC ≈ 0.993
- [ ] Threshold should be ~0.87 (stored in `inference_config.json`)
- [ ] If results differ significantly, verify `random_state=42` and Optuna seed settings

### 10.5 Loading Published Model

For inference without retraining:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("aditya963/fraud-job-classifier")
tokenizer = AutoTokenizer.from_pretrained("aditya963/fraud-job-classifier")
```

### 10.6 Running the Web-App

- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Run: `python web-app/app.py`
- [ ] Navigate to `http://localhost:5000`
- [ ] Test with a sample job description (paste any LinkedIn job description)

### 10.7 Fixed Seeds Used

| Component | Seed | Where |
|---|---|---|
| Train/val/test split | `random_state=42` | `utils/data.py` |
| Optuna sampler | Default (no fixed seed — trials vary) | Notebook HPO section |
| IsolationForest | `random_state=42` | `tools/metadata_detector/anomaly_model.py` |
