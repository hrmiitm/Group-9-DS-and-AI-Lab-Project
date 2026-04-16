# Future Work and Known Limitations

**Project:** FraudGuard — Fake Job Listing Detection
**Last Updated:** 2026-04-15

---

## 1. Known Limitations

### 1.1 Model Performance Limitations

| Limitation | Impact | Severity |
|---|---|---|
| **Recall miss (0.862 vs. target 0.89)** | ~14% of fraudulent postings are missed (false negatives) at the operating threshold | Medium |
| **512-token sequence limit** | ~11% of job postings exceed 512 tokens and are truncated; fraud signals buried in long descriptions may be lost | Medium |
| **English-only** | The model was trained exclusively on English-language postings; non-English fraud is undetected | High |
| **Dataset age** | The EMSCAD dataset was collected around 2014–2017; newer fraud patterns (AI-generated listings, deepfake company profiles) may not be represented | Medium |
| **Class imbalance (4.84% fraud)** | Despite Focal Loss, the 20:1 imbalance means the model is less confident on edge-case fraud patterns | Medium |

### 1.2 Web-App Limitations

| Limitation | Impact | Severity |
|---|---|---|
| **Company registry tool is a stub** | `tool_company_registry.py` always returns `{ok: false}` — no actual company registration check | Medium |
| **File upload size not enforced** | `MAX_UPLOAD_BYTES=10MB` is defined but not enforced in `routes/main.py` | Low |
| **Synchronous pipeline** | Long analyses (60–90s) block the server process; no async/queue support | Medium |
| **No automated tests** | Regressions may go undetected without a pytest test suite | Medium |
| **LinkedIn scraping blocked** | LinkedIn increasingly blocks automated scraping; the URL input may silently return empty | Low |
| **Session-based config** | LLM API key is stored in an HTTP cookie session — not appropriate for multi-user production | Low |

### 1.3 Chrome Extension Limitations

| Limitation | Impact | Severity |
|---|---|---|
| **LinkedIn only** | Extension only activates on `linkedin.com`; does not work on Indeed, Naukri, or other job boards | Medium |
| **Gemini API dependency** | Extension requires a Google Gemini API key; free tier has rate limits | Low |
| **No 12-tool verification** | Extension only uses a single LLM call (Gemini), not the full 12-tool pipeline available in the web-app | High |
| **LinkedIn DOM fragility** | LinkedIn A/B tests their UI constantly; class name changes may break scraping | Medium |

### 1.4 Infrastructure Limitations

| Limitation | Impact | Severity |
|---|---|---|
| **No production deployment** | App runs locally only; no cloud hosting, authentication, or HTTPS | High |
| **No model serving API** | RoBERTa model must be loaded locally; no REST API wrapper around the model | Medium |
| **Google Colab dependency for training** | Full reproducibility requires a T4 GPU; local CPU training is feasible but slow | Low |

---

## 2. Possible Extensions

### 2.1 Model Improvements

**Longer sequence handling:**
Approximately 11% of samples are truncated at 512 tokens. Two approaches:
- **Sliding window:** Split the input into overlapping 512-token windows and aggregate predictions (e.g., max probability).
- **Hierarchical encoding:** Use a sentence encoder (e.g., `sentence-transformers`) on paragraphs, then a lightweight classifier on the aggregated embeddings.

**Multilingual support:**
Fine-tune `xlm-roberta-base` (trained on 100 languages) on translated or natively multilingual fraud datasets to detect Hindi, Bangla, and other South Asian language fraud postings — particularly relevant for the Indian market.

**Model ensembling:**
Combine RoBERTa and DeBERTa-v3 probability scores (e.g., averaging) to push ROC-AUC and F1 beyond individual model ceilings.

**Synthetic data augmentation (quality-filtered):**
The `v5_synth` experiment showed promise for LLM-generated fraud postings. A quality filtering step (semantic diversity scoring, perplexity filter) before augmentation would improve this approach.

**Continual learning pipeline:**
As new fraud patterns emerge (AI-generated listings, synthetic job ads), a periodic re-fine-tuning pipeline on newly labeled data would keep the model current.

### 2.2 Agentic Pipeline Extensions

**Implement company registry verification:**
`tool_company_registry.py` is currently a stub. Implement actual company registration checks using:
- [Companies House API](https://developer.company-information.service.gov.uk/) (UK)
- [Ministry of Corporate Affairs API](https://www.mca.gov.in/content/mca/global/en/home.html) (India)
- OpenCorporates API (global)

**Add RoBERTa as a verification tool:**
Integrate the fine-tuned RoBERTa model directly into the web-app's tool pipeline as `tool_roberta_classifier`. Currently the web-app uses only the LLM-based pipeline; adding the ML model would provide a second independent signal.

**Expand to other job boards:**
Add specialized scrapers and content handlers for Indeed, Naukri, Glassdoor, and Internshala.

**Real-time alert system:**
Add a monitoring mode that periodically re-checks saved analyses for newly discovered fraud complaints (new web search results) and sends alerts.

### 2.3 Chrome Extension Extensions

**Full 12-tool pipeline in extension:**
Replace the Gemini API call with a call to the web-app's `/analyze` endpoint, enabling the complete evidence-backed analysis from within the browser.

**Support more job boards:**
Extend `content.js` to activate on Indeed, Naukri, Glassdoor, and other major platforms.

**Extension-local RoBERTa:**
Use HuggingFace Transformers.js (WebAssembly) to run a quantized version of the RoBERTa model directly in the extension without any API calls.

### 2.4 Deployment and Infrastructure

**Cloud deployment:**
- Package web-app as a Docker container
- Deploy to Render, Railway, or Google Cloud Run
- Add HTTPS via Let's Encrypt

**Model serving API:**
Wrap the fine-tuned RoBERTa model in a FastAPI service with the inference endpoint:
```
POST /predict
Body: {"text": "job description text"}
Response: {"fraud_probability": 0.92, "prediction": "FRAUDULENT"}
```

**Authentication and multi-user support:**
Add user registration and per-user analysis history for public deployment.

**Test suite:**
Add a pytest test suite covering:
- Unit tests for each tool function
- Integration tests for the analysis pipeline
- Regression tests for specific known fraud patterns

---

## 3. How to Retrain or Update the Model with New Data

### Adding New Training Data

1. **Collect new labeled samples:**
   - Label new job postings as 0 (legitimate) or 1 (fraudulent)
   - Aim for at least 500+ new fraud samples to avoid catastrophic forgetting
   - Store in the same CSV format as `fake_job_postings.csv` (18 columns)

2. **Merge with existing dataset:**
   ```python
   import pandas as pd
   old = pd.read_csv("data/raw/fake_job_postings.csv")
   new = pd.read_csv("data/raw/new_samples.csv")
   combined = pd.concat([old, new]).drop_duplicates(subset=["title", "description"])
   combined.to_csv("data/raw/fake_job_postings_v2.csv", index=False)
   ```

3. **Update `DATA_PATH` in the notebook** and re-run the training pipeline.

4. **Re-run Optuna HPO** (at least 15 trials) to recalibrate hyperparameters for the new data distribution.

5. **Upload new model to HuggingFace Hub:**
   ```python
   model.push_to_hub("your-username/fraud-job-classifier-v2")
   tokenizer.push_to_hub("your-username/fraud-job-classifier-v2")
   ```

### Updating the Operating Threshold

After retraining, re-run the threshold calibration sweep on the new validation set:
```python
from sklearn.metrics import f1_score, precision_score, recall_score
best_f1, best_threshold = 0, 0.5
for t in [i/100 for i in range(5, 95)]:
    preds = (probs_val >= t).astype(int)
    f1 = f1_score(labels_val, preds)
    recall = recall_score(labels_val, preds)
    precision = precision_score(labels_val, preds)
    if recall >= 0.89 and precision >= 0.93 and f1 > best_f1:
        best_f1, best_threshold = f1, t
```

Update `inference_config.json` with the new threshold.

---

## 4. Maintainer Contacts

<!-- TEAM NAMES HERE -->

For questions about this project, contact the team:
- **Arun Dutta** — <!-- EMAIL -->
- **Hritik Roshan Maurya** — <!-- EMAIL -->
- **Vivek Bajaj** — <!-- EMAIL -->
- **Vishwas Mehta** — <!-- EMAIL -->

**GitHub Repository:** [github.com/hrmiitm/Group-9-DS-and-AI-Lab-Project](https://github.com/hrmiitm/Group-9-DS-and-AI-Lab-Project)

**HuggingFace Model:** [huggingface.co/aditya963/fraud-job-classifier](https://huggingface.co/aditya963/fraud-job-classifier)
