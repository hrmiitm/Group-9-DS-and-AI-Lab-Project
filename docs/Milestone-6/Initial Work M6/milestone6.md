# Milestone 6 — Deployment & Documentation

**Project:** FraudGuard — Fake Job Listing Detection using Deep Learning and Agentic AI
**Team:** Group 9 — Arun Dutta · Hritik Roshan Maurya · Vivek Bajaj · Vishwas Mehta
**Course:** DS & AI Lab Project
**Submission Date:** April 2026

---

## Overview

Milestone 6 turns FraudGuard from a research notebook into a working, documented, reproducible product. This file is the **single-point summary** of everything delivered, mapped section-by-section against the Milestone 6 Deliverable Structure PDF.

---

## Deliverable 1 — Deployment

### What Is Deployed

| Component | Platform | Access |
|---|---|---|
| RoBERTa fraud classifier (weights) | HuggingFace Hub | `from_pretrained("aditya963/fraud-job-classifier")` |
| Model REST API | HuggingFace Spaces (Docker SDK) | `https://hrmhrmhrm-roberta-model.hf.space` |
| Flask web application | Local dev server | `python web-app/app.py` → `http://localhost:5000` |
| Chrome extension | Browser (load unpacked) | `chrome://extensions` → Load unpacked → `web-extension/` |
| Training notebooks | Google Colab (T4 GPU) | `notebook/` |

### Deployment Architecture

```
                    ┌───────────────────────────────────────┐
                    │         User Entry Points              │
                    ├─────────────────┬─────────────────────┤
                    │  Flask Web-App  │   Chrome Extension   │
                    │  localhost:5000 │   LinkedIn pages      │
                    └────────┬────────┴──────────┬──────────┘
                             │                   │
                             ▼                   ▼
                    ┌────────────────┐  ┌─────────────────┐
                    │  12-Tool Agent │  │   Gemini API    │
                    │  (LangChain +  │  │   (Gemini Pro)  │
                    │  OpenRouter)   │  └─────────────────┘
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────────────────────────────┐
                    │  RoBERTa Model API  (HuggingFace Spaces)│
                    │  POST /predict  →  fraud_probability    │
                    │  aditya963/fraud-job-classifier         │
                    └────────────────────────────────────────┘
```

### How to Run Locally

```bash
# 1. Clone repo
git clone <repo-url>
cd Group-9-DS-and-AI-Lab-Project

# 2. Create and activate environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp web-app/.env.example web-app/.env
# Edit web-app/.env: add OPENROUTER_API_KEY

# 5. Run Flask web-app
python web-app/app.py
# Open: http://localhost:5000

# Chrome extension: chrome://extensions → Load unpacked → web-extension/
# Add Gemini API key in extension popup settings
```

### Model REST API (HuggingFace Spaces)

**Base URL:** `https://hrmhrmhrm-roberta-model.hf.spacehttps://hrmhrmhrm-roberta-model.hf.space`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — model status, device, threshold |
| `POST` | `/predict` | Single job posting → fraud probability + verdict |
| `POST` | `/predict/batch` | Up to 16 postings → list of predictions |

**Example request:**

```bash
curl -X POST https://hrmhrmhrm-roberta-model.hf.spacehttps://hrmhrmhrm-roberta-model.hf.spacehttps://hrmhrmhrm-roberta-model.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Work From Home Data Entry",
    "description": "Earn $500/day, no experience needed. Send bank details.",
    "location": "Remote",
    "has_company_logo": 0
  }'
```

**Example response:**

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

Full API reference: [docs/api_doc.md](../api_doc.md)

### Inputs and Outputs

**Web-app inputs:**
- Free-text job description (paste)
- File upload: `.pdf`, `.docx`, `.txt`, `.html`, `.csv`
- LinkedIn job URL (scrapes and analyzes)

**Web-app output:** Full investigation report with:
- Verdict: `SAFE` / `SUSPICIOUS` / `LIKELY_FAKE`
- Per-tool evidence cards (12 tools)
- LLM-written narrative report
- RoBERTa fraud probability score

**Chrome extension inputs:** Any LinkedIn job listing page (auto-scraped DOM)

**Chrome extension output:** Color-coded overlay on the page (`🟢 LEGITIMATE` / `🟡 SUSPICIOUS` / `🔴 FRAUDULENT`) with key reasons

---

## Deliverable 2 — Comprehensive Documentation

All documentation lives in `/docs/` and maps to the required sections as follows:

### A. Overview

**File:** [docs/overview.md](../overview.md)

| Required Item | Status | Location |
|---|---|---|
| Purpose & problem statement | ✅ Done | overview.md §1 |
| Architecture summary with data flow diagram | ✅ Done | overview.md §2, §4 |
| Deployed components list | ✅ Done | overview.md §3 |

### B. Technical Documentation

**File:** [docs/technical_doc.md](../technical_doc.md)

| Required Section | Status | Location |
|---|---|---|
| 1. Environment Setup | ✅ Done | technical_doc.md §1 |
| 2. Data Pipeline | ✅ Done | technical_doc.md §2 |
| 3. Model Architecture | ✅ Done | technical_doc.md §3 |
| 4. Training Summary | ✅ Done | technical_doc.md §4 |
| 5. Evaluation Summary | ✅ Done | technical_doc.md §5 |
| 6. Inference Pipeline | ✅ Done | technical_doc.md §6 |
| 7. Deployment Details | ✅ Done | technical_doc.md §7 |
| 8. System Design Considerations | ✅ Done | technical_doc.md §8 |
| 9. Error Handling & Monitoring | ✅ Done | technical_doc.md §9 |
| 10. Reproducibility Checklist | ✅ Done | technical_doc.md §10 |

### C. User Documentation

**File:** [docs/user_guide.md](../user_guide.md)

| Required Item | Status | Location |
|---|---|---|
| App Overview & use cases | ✅ Done | user_guide.md §1 |
| Input description | ✅ Done | user_guide.md §2–3 |
| Output description | ✅ Done | user_guide.md §4 |
| Step-by-step: web-app launch | ✅ Done | user_guide.md §2 |
| Step-by-step: extension install | ✅ Done | user_guide.md §3 |
| Example queries | ✅ Done | user_guide.md §5 |
| Troubleshooting | ✅ Done | user_guide.md §6 |
| Screenshots | ⚠️ Placeholders only | user_guide.md — add screenshots manually |

### D. API Documentation

**File:** [docs/api_doc.md](../api_doc.md)

| Required Item | Status | Location |
|---|---|---|
| Base URL | ✅ Done | api_doc.md Part 1 |
| POST /predict — input format + response | ✅ Done | api_doc.md §1.3 |
| GET / — health check | ✅ Done | api_doc.md §1.3 |
| Example curl requests | ✅ Done | api_doc.md §1.4 |
| Response format / JSON keys | ✅ Done | api_doc.md §1.2 |
| Web-app endpoints | ✅ Done | api_doc.md Part 2 |

### E. Licensing & Dataset References

**File:** [docs/licenses.md](../licenses.md)

| Required Item | Status |
|---|---|
| Code license (MIT) | ✅ Done |
| Dataset license (CC BY 4.0 — EMSCAD) | ✅ Done |
| Model sources and citations | ✅ Done |

### F. Future Work / Maintenance Notes

**File:** [docs/future_work.md](../future_work.md)

| Required Item | Status |
|---|---|
| Possible extensions | ✅ Done |
| Known limitations | ✅ Done |
| How to retrain / update the model | ✅ Done |
| Contacts / maintainers | ⚠️ Email addresses are placeholders |

---

## Deliverable 3 — Final Project Report

**File:** [docs/Final_Project_Report.md](../Final_Project_Report.md)
**Script to generate Google Doc:** [create_google_doc.py](../../create_google_doc.py)

The report follows the required structure:

| Required Section | Status | Notes |
|---|---|---|
| 1. Title Page | ✅ | Group 9, April 2026 |
| 2. Abstract | ✅ | ~200 words |
| 3. Introduction | ✅ | Problem, motivation, goals |
| 4. Literature Review (Milestone 1) | ✅ | 5 approaches, gap analysis |
| 5. Dataset and Methodology (Milestone 2–3) | ✅ | EMSCAD, preprocessing, splits |
| 6. Model Development & Hyperparameter Tuning (Milestone 4) | ✅ | v1→v3_1 progression, Optuna |
| 7. Evaluation & Analysis (Milestone 5) | ✅ | All metrics, confusion matrix, comparisons |
| 8. Deployment & Documentation (Milestone 6) | ✅ | This milestone |
| 9. Conclusion and Future Work | ✅ | |
| 10. References and Appendix | ✅ | 12 citations, 5 appendices |

**Mermaid diagrams in report:** 20 diagrams covering system architecture, preprocessing flow, model architecture, HPO process, training curves, metric comparisons, confusion matrix, pipeline flows, milestone timeline, future roadmap.

---

## Additional Nice-to-Have Items

| Item | Status | Notes |
|---|---|---|
| Demo video (10-20 min) | ⚠️ Not recorded | Record and embed YouTube link in README |
| Quick Start in README | ✅ Done | README.md has Quick Start section |
| Resource constraints documented | ✅ Done | technical_doc.md §1 — Google Colab T4, CPU inference |

---

## File Map — Complete Deliverable Tree

```
Group-9-DS-and-AI-Lab-Project/
├── README.md                          ✅ Rewritten with badges, Quick Start, architecture
├── requirements.txt                   ✅ All dependencies pinned
├── create_google_doc.py               ✅ Generates Google Doc from Final_Project_Report.md
│
├── model-api/                         ✅ HuggingFace Spaces Docker deployment
│   ├── app.py                         ✅ FastAPI service (health, /predict, /predict/batch)
│   ├── Dockerfile                     ✅ HF Spaces compliant (port 7860, UID 1000)
│   ├── requirements.txt               ✅ Pinned ML stack
│   ├── download_model.py              ✅ Build-time weight pre-download
│   └── README.md                      ✅ HF Spaces YAML + full API docs
│
├── web-app/                           ✅ Flask application
├── web-extension/                     ✅ Chrome extension (MV3)
├── notebook/                          ✅ Training notebooks
│
└── docs/
    ├── overview.md                    ✅ A. Overview
    ├── technical_doc.md               ✅ B. Technical Documentation (10 sections)
    ├── user_guide.md                  ✅ C. User Documentation
    ├── api_doc.md                     ✅ D. API Documentation (Model API + Web-App API)
    ├── licenses.md                    ✅ E. Licensing & Dataset References
    ├── future_work.md                 ✅ F. Future Work / Maintenance Notes
    ├── contribution_summary.md        ✅ Team contributions
    ├── Final_Project_Report.md        ✅ Full academic report (20 Mermaid diagrams)
    └── Milestone-6/
        ├── milestone6.md              ✅ This file — deliverable summary
        ├── team_contribution.md       ✅ Milestone 6 specific contributions
        ├── notebooklm_slides_prompt.md ✅ Slide generation prompt
        └── gaps.md                    ✅ Known gaps and how to fix them
```
