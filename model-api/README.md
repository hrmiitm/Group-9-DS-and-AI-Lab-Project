---
title: FraudGuard RoBERTa API
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# FraudGuard — RoBERTa Fraud Job Classifier API

> FastAPI service wrapping the fine-tuned `aditya963/fraud-job-classifier` model.
> Accepts a job posting, returns a fraud probability score and verdict.

---

## Model Performance

| Metric | Score | Target |
|---|---|---|
| F1 (fraud class) | 0.9069 | ≥ 0.91 |
| Precision | 0.9573 | ≥ 0.93 |
| Recall | 0.8615 | ≥ 0.89 |
| ROC-AUC | **0.9930** | ≥ 0.95 |
| MCC | 0.8917 | — |

Operating threshold: **0.87** (calibrated on validation set)

---

## API Flow

```
POST /predict
─────────────────────────────────────────────────────────────────
JobPosting JSON (title, description, location, salary_range ...)
        │
        ▼
  build_input_text()
  ┌─────────────────────────────────────────────────────────┐
  │  Structured fields first (key: value)                   │
  │  "Location: New York [SEP] Salary Range: 80000-100000   │
  │   [SEP] Employment Type: Full-time [SEP] ..."           │
  │  Then free-text fields (title, description, etc.)       │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  RoBERTa BPE Tokenizer  →  input_ids [512]  +  attention_mask [512]
        │
        ▼
  RoBERTa-base Encoder  (12 layers × 12 heads × 768 dim)
        │
        ▼
  [CLS] token  →  Dropout(0.1)  →  Linear(768→2)  →  Softmax
        │
        ▼
  P(fraud)  ──  threshold 0.87  ──▶  FRAUDULENT / LEGITIMATE
        │
        ▼
  { fraud_probability, verdict, confidence, latency_ms }
─────────────────────────────────────────────────────────────────
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check — model status, device, threshold |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |
| `POST` | `/predict` | Single job posting → prediction |
| `POST` | `/predict/batch` | Up to 16 postings → list of predictions |

---

## Quick Examples

### Health check

```bash
curl https://YOUR-SPACE.hf.space/
```

```json
{
  "status": "ok",
  "model_id": "aditya963/fraud-job-classifier",
  "threshold": 0.87,
  "device": "cpu",
  "version": "1.0.0"
}
```

### Single prediction

```bash
curl -X POST https://YOUR-SPACE.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Work From Home Data Entry Specialist",
    "description": "Earn $500/day. No experience needed. Send bank details.",
    "company_profile": "",
    "location": "Remote",
    "salary_range": "500-1000",
    "employment_type": "Part-time",
    "has_company_logo": 0
  }'
```

```json
{
  "fraud_probability": 0.9247,
  "fraud_percent": 92.5,
  "verdict": "FRAUDULENT",
  "confidence": "HIGH",
  "threshold": 0.87,
  "model_id": "aditya963/fraud-job-classifier",
  "latency_ms": 43.2
}
```

### Batch prediction

```bash
curl -X POST https://YOUR-SPACE.hf.space/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "postings": [
      {"title": "Senior Engineer", "description": "5 years Python required", "has_company_logo": 1},
      {"title": "Easy money from home", "description": "No experience needed. Pay upfront.", "has_company_logo": 0}
    ]
  }'
```

---

## Running Locally with Docker

```bash
# 1. Clone or copy the model-api/ folder
cd model-api

# 2. Build the image (downloads model weights into the image layer)
docker build -t fraudguard-api .

# 3. Run
docker run -p 7860:7860 fraudguard-api

# 4. Open Swagger UI
open http://localhost:7860/docs
```

---

## Deploying to HuggingFace Spaces

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Set **SDK** to **Docker**
3. Name it `fraudguard-api` (or any name)
4. Visibility: Public or Private

### Step 2 — Push this folder

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Clone your new Space repo
git clone https://huggingface.co/spaces/YOUR-USERNAME/fraudguard-api
cd fraudguard-api

# Copy the three files from model-api/
cp /path/to/model-api/app.py .
cp /path/to/model-api/requirements.txt .
cp /path/to/model-api/Dockerfile .
cp /path/to/model-api/README.md .

# Push
git add .
git commit -m "Deploy FraudGuard RoBERTa API"
git push
```

### Step 3 — Watch the build

- Go to your Space URL on HuggingFace
- Click **"Factory reboot"** if needed
- Build takes ~3–5 minutes (model weights are downloaded during Docker build)
- Once green: your API is live at `https://YOUR-USERNAME-fraudguard-api.hf.space`

### Step 4 — Update your extension

In [web-extension/tools/roberta-tool.js](../web-extension/tools/roberta-tool.js), replace the HuggingFace Inference API URL with your Space URL:

```js
// Before (uses HF Inference API — requires HF token, rate limited):
const HF_MODEL_URL = "https://api-inference.huggingface.co/models/aditya963/fraud-job-classifier";

// After (your own Space — no token needed, no rate limit):
const HF_MODEL_URL = "https://YOUR-USERNAME-fraudguard-api.hf.space/predict";
```

Also change the request body format:

```js
// Before (HF Inference API format):
body: JSON.stringify({ inputs: standardizedText })

// After (this API's format):
body: JSON.stringify({ description: standardizedText })
// Or pass all fields individually as JobPosting fields
```

---

## Environment Variables

Set these in your HuggingFace Space settings under **"Variables and secrets"**:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `aditya963/fraud-job-classifier` | HuggingFace model repo to load |
| `FRAUD_THRESHOLD` | `0.87` | Classification threshold (0–1) |
| `MAX_LENGTH` | `512` | Tokenizer max sequence length |
| `MAX_BATCH_SIZE` | `16` | Maximum postings per batch request |

---

## Input Schema Reference

All fields are optional. The more fields provided, the more accurate the prediction.

| Field | Type | Example |
|---|---|---|
| `title` | string | `"Software Engineer"` |
| `description` | string | `"We are looking for..."` |
| `requirements` | string | `"5 years Python"` |
| `company_profile` | string | `"A leading tech firm..."` |
| `benefits` | string | `"Health insurance, WFH"` |
| `location` | string | `"New York, NY, US"` |
| `salary_range` | string | `"80000-100000"` |
| `employment_type` | string | `"Full-time"` |
| `required_experience` | string | `"Mid-Senior level"` |
| `required_education` | string | `"Bachelor's Degree"` |
| `department` | string | `"Engineering"` |
| `industry` | string | `"Information Technology"` |
| `function` | string | `"Engineering"` |
| `has_company_logo` | int (0 or 1) | `1` |
| `telecommuting` | int (0 or 1) | `0` |
| `has_questions` | int (0 or 1) | `1` |
