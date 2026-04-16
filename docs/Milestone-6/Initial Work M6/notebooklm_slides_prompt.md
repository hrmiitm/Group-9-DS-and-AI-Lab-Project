# NotebookLM Slide Generation Prompt — FraudGuard

**How to use this file:**
1. Open [notebooklm.google.com](https://notebooklm.google.com)
2. Create a new notebook
3. Upload these sources:
   - `docs/Final_Project_Report.md`
   - `docs/technical_doc.md`
   - `docs/overview.md`
   - `docs/api_doc.md`
   - `README.md`
4. Open the **Studio** panel → select **Presentation**
5. Paste the prompt below into the instructions field
6. Click Generate

---

## Slide Generation Prompt

```
Generate a professional academic presentation for the FraudGuard final project report.
The audience is course instructors and peer reviewers evaluating a Data Science and AI
Lab project at IIT Madras. The tone should be clear, confident, and data-driven.

Create exactly 12 slides following this structure:

---

SLIDE 1 — Title Slide
Title: FraudGuard: Fake Job Listing Detection using Deep Learning and Agentic AI
Subtitle: Data Science and AI Lab Project — Final Presentation
Group: Group 9 — Arun Dutta · Hritik Roshan Maurya · Vivek Bajaj · Vishwas Mehta
Date: April 2026

---

SLIDE 2 — Problem Statement & Motivation
Heading: The Problem: Fake Job Listings at Scale
- 3–5% of all online job postings are fraudulent
- Modern fraud postings mimic legitimate ads with linguistic sophistication — keyword filters fail
- Victims lose money, personal data, and time — real-world impact on Indian job seekers
- No existing tool combines deep NLP + external verification + human-readable explanation
End with one bold hook sentence about what FraudGuard does differently.

---

SLIDE 3 — System Architecture
Heading: FraudGuard — Three-Layer Architecture
Describe three layers with a clear visual flow:
  Layer 1: RoBERTa-base classifier (125M params, EMSCAD dataset, threshold 0.87)
  Layer 2: 12-Tool Agentic Verification Pipeline (LangChain + OpenRouter LLM)
  Layer 3: Two user interfaces — Flask web-app + Chrome extension (LinkedIn)
Use arrows to show data flow: Input → Extract → Verify → Report → User

---

SLIDE 4 — Dataset & Preprocessing
Heading: EMSCAD Dataset — 17,880 Job Postings
Key facts:
- 17,880 samples, 4.84% fraudulent (866 fraud, 17,014 legitimate) — 20:1 imbalance
- 18 features: 5 free-text, 9 structured metadata, 3 binary
- After deduplication: 15,787 unique samples
- Split: 70% train / 15% validation / 15% test (stratified)
Preprocessing innovation: structured metadata fields placed FIRST in the [SEP]-joined
input string — protects critical signals from 512-token truncation.
Show the input format: "Location: New York [SEP] Salary: 500-1000 [SEP] title text [SEP] description..."

---

SLIDE 5 — Model Architecture & Training Challenge
Heading: RoBERTa-base — Why and How
Architecture bullet points:
- roberta-base: 12 transformer layers, 12 attention heads, 768-dim hidden state, 125.5M params
- Classification head: Dropout(0.1) → Linear(768→2) → Softmax
- Full fine-tuning (not LoRA) — small dataset needs all parameters to adapt
Challenge: 20:1 class imbalance
Solution: Focal Loss (γ=1.6920, fraud class weight=2.8251) — down-weights easy negatives,
          amplifies gradient signal on hard fraud examples

---

SLIDE 6 — Hyperparameter Optimization
Heading: 25-Trial Optuna HPO — Systematic Search with Hard Floors
Show the search space table:
  lr: 2.59e-5 | warmup: 0.1506 | batch: 16 | weight_decay: 0.0702
  focal_gamma: 1.6920 | fraud_class_weight: 2.8251 | epochs: 9

Key innovation: Trials are PRUNED unless Recall ≥ 0.89 AND Precision ≥ 0.93 simultaneously.
This prevents optimizing F1 at the expense of either metric.

Show model version progression:
v1 (CE) → 0.875 F1 | v2 (Focal) → 0.910 | v3_1 Final → 0.9069 F1, AUC 0.993

---

SLIDE 7 — Evaluation Results
Heading: Results — Near-Perfect AUC, High Precision
Present final metrics as a prominent table:
  Metric       | Target | Achieved | Status
  F1 (fraud)   | ≥ 0.91 | 0.9069   | Narrow miss (−0.3%)
  Recall       | ≥ 0.89 | 0.8615   | Miss (−2.9%)
  Precision    | ≥ 0.93 | 0.9573   | ✅ Met (+2.7%)
  ROC-AUC      | ≥ 0.95 | 0.9930   | ✅ Met (+4.3%)
  MCC          | —      | 0.8917   | —

Headline result: AUC 0.993 — the model's probability scores cleanly separate classes
in 99.3% of cases. This is the most important metric for a fraud scoring system.

Threshold calibration: moving threshold from 0.50 → 0.87 added 3–4 F1 points at zero training cost.

---

SLIDE 8 — Agentic Verification Pipeline
Heading: Beyond Classification — 12-Tool Evidence Pipeline
Explain why a classifier alone is not enough: it cannot verify external facts.
List the 12 tools:
  email_verify · domain_reputation · website_verify · website_content
  company_wikipedia · company_web_search · company_news · social_profiles
  job_boards · phone_check · scam_signals · company_registry

Pipeline flow:
  Input → LLM extracts 16 fields → DuckDuckGo fills gaps → 12 tools run in parallel
  → LLM summarizes each tool → Final LLM report → SAFE / SUSPICIOUS / LIKELY_FAKE

Highlight: This produces human-readable evidence, not just a probability score.

---

SLIDE 9 — Web Application Demo
Heading: Flask Web-App — Three Input Modes
Describe the three input paths:
  1. Paste text → job description entered directly
  2. File upload → PDF, DOCX, TXT, HTML, CSV
  3. LinkedIn URL → scrapes and analyzes live posting

Show what the user sees:
  - Verdict banner (SAFE / SUSPICIOUS / LIKELY_FAKE) with confidence
  - 12 tool result cards, each with 2–4 sentence LLM summary
  - Full LLM-written investigation report
  - RoBERTa fraud probability score

Include a note: analysis takes 30–90 seconds (LLM calls + 12 tool executions)

---

SLIDE 10 — Chrome Extension Demo
Heading: Chrome Extension — Real-Time LinkedIn Analysis
Flow:
  1. User visits any LinkedIn job listing
  2. Extension injects "🔍 Analyze Job" floating button
  3. Click → extension scrapes DOM (title, company, description, requirements)
  4. Gemini Pro API called with structured fraud-analysis prompt
  5. Color-coded overlay appears within 3–5 seconds

Verdicts: 🟢 LEGITIMATE · 🟡 SUSPICIOUS · 🔴 FRAUDULENT
Plus: key fraud indicators, confidence score, actionable tips

Contrast with web-app: extension is fast (3–5s, Gemini only) vs deep (30–90s, 12 tools).
Both are needed for different user contexts.

---

SLIDE 11 — Deployment & Reproducibility
Heading: What Was Shipped — Fully Reproducible Stack
Three artifacts:
  1. Model weights: HuggingFace Hub — aditya963/fraud-job-classifier
     → loadable in 2 lines: from_pretrained("aditya963/fraud-job-classifier")
  2. Model REST API: HuggingFace Spaces (Docker) — FastAPI, /predict, /predict/batch
     → any client can call it with a single HTTP request, no PyTorch needed
  3. Applications: Flask web-app (localhost) + Chrome extension (load unpacked)

Documentation stack:
  docs/overview.md · technical_doc.md · user_guide.md · api_doc.md · licenses.md · future_work.md
  Final_Project_Report.md (20 Mermaid diagrams) · README.md (Quick Start)

---

SLIDE 12 — Conclusion & Future Work
Heading: What We Built and What Comes Next
Summary (3 bullet points):
- FraudGuard achieves AUC 0.993 and Precision 0.957 — state-of-the-art on EMSCAD
- Unique combination: deep NLP classifier + 12-tool external verification + LLM narrative
- Deployed as a REST API, Flask web-app, and Chrome extension — accessible to all users

Top 3 future improvements:
1. Multilingual support — xlm-roberta-base for Hindi and regional language fraud
2. Sliding-window encoding — recover fraud signals from >512 token postings
3. Production deployment — Celery + Redis async queue, Docker + HTTPS, company registry tool

Closing line: FraudGuard shows that combining transformer fine-tuning with agentic AI
produces qualitatively superior fraud detection — one that not only detects, but explains.

---

END OF PROMPT
```

---

## Formatting Notes for NotebookLM

- Ask it to use **bullet points** for technical slides and **narrative sentences** for intro/conclusion slides
- For the metrics table on Slide 7, explicitly ask: *"present as a formatted table with emoji status indicators"*
- If slides feel too dense: *"split slide X into two slides, one for context and one for data"*
- For architecture slides: *"describe as a numbered step-by-step flow with arrows"*

## Alternative: Google Slides Manual Template

If you prefer to build slides manually, follow this colour scheme (matching the sample PDF):

| Element | Value |
|---|---|
| Background | White `#FFFFFF` |
| H1 heading colour | Blue `#4472C4` |
| Body text | Black `#000000`, 12pt Calibri |
| Accent / highlight | Orange `#E67E22` |
| Metric "good" colour | Green `#27AE60` |
| Metric "miss" colour | Red `#E74C3C` |
| Code font | Courier New, 10pt, grey background `#F2F2F2` |
