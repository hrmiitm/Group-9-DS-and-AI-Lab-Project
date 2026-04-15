# Team Contribution — Milestone 6

**Project:** FraudGuard — Fake Job Listing Detection using Deep Learning and Agentic AI
**Milestone:** 6 — Deployment & Documentation
**Team:** Group 9
**Date:** April 2026

---

## Milestone 6 Contribution Summary

| Deliverable | Owner | Status |
|---|---|---|
| Model REST API (`model-api/`) — FastAPI + Dockerfile | Vivek Bajaj | ✅ Complete |
| HuggingFace Spaces deployment | Vivek Bajaj | ✅ Deployed |
| API Documentation (`docs/api_doc.md`) | Hritik Roshan Maurya | ✅ Complete |
| Technical Documentation (`docs/technical_doc.md`) | Hritik Roshan Maurya + Vivek Bajaj | ✅ Complete |
| System Overview (`docs/overview.md`) | Hritik Roshan Maurya | ✅ Complete |
| User Guide (`docs/user_guide.md`) | Vishwas Mehta | ✅ Complete |
| Chrome Extension documentation | Vishwas Mehta | ✅ Complete |
| Final Project Report (`docs/Final_Project_Report.md`) | Arun Dutta | ✅ Complete |
| Contribution Summary (`docs/contribution_summary.md`) | Arun Dutta | ✅ Complete |
| Licenses & Dataset References (`docs/licenses.md`) | Arun Dutta | ✅ Complete |
| Future Work (`docs/future_work.md`) | Arun Dutta | ✅ Complete |
| README.md rewrite | All (led by Arun Dutta) | ✅ Complete |
| Repo Audit & .gitignore (`docs/REPO_AUDIT.md`) | All | ✅ Complete |
| Milestone 6 folder (`docs/Milestone-6/`) | All | ✅ Complete |
| Google Doc generation script (`create_google_doc.py`) | Vishwas Mehta | ✅ Complete |

---

## Individual Contributions — Milestone 6 Detail

### Arun Dutta
**Role: Documentation Lead & Report Author**

- Authored `docs/Final_Project_Report.md` — the complete academic-style final report (~600 lines) including abstract, all 8 body sections, 12 references, and 5 appendices with code samples
- Added 20 Mermaid diagrams to the report covering system architecture, preprocessing flow, model architecture, Optuna HPO process, metric comparisons, confusion matrix, deployment flows, milestone Gantt chart, and future roadmap
- Wrote `docs/contribution_summary.md` with full per-milestone, per-member breakdown (all 4 milestones × 4 members)
- Wrote `docs/licenses.md` with code license, EMSCAD dataset attribution, model card citations
- Wrote `docs/future_work.md` with short/medium/long-term roadmap, known limitations, and maintenance contacts
- Co-led final README.md rewrite with badges, Quick Start, architecture diagram, results table, and annotated folder tree
- Coordinated repo audit and gap analysis (`docs/REPO_AUDIT.md`, `docs/GAPS_FIXED.md`)

---

### Hritik Roshan Maurya
**Role: API & Deployment Documentation Lead**

- Authored `docs/api_doc.md` — restructured into two parts (RoBERTa Model API + Web-App API) with ASCII architecture diagrams, Mermaid deployment flow diagrams, full endpoint documentation, schema tables, Python client examples, curl examples, and error code reference
- Authored `docs/overview.md` with system architecture narrative, component interaction ASCII diagram, Mermaid sequence diagram, and key design decisions table
- Authored `docs/technical_doc.md` — all 10 required sections:
  - §1 Environment Setup (Python 3.10+, CUDA, pip install instructions)
  - §2 Data Pipeline (EMSCAD preprocessing, tokenization, split strategy)
  - §3 Model Architecture (RoBERTa-base, 125M params, classification head)
  - §4 Training Summary (Focal Loss, 25-trial Optuna, training artifacts)
  - §5 Evaluation Summary (all metrics, threshold calibration)
  - §6 Inference Pipeline (code snippet, data flow)
  - §7 Deployment Details (HF Hub, HF Spaces, local Flask)
  - §8 System Design Considerations (async future, modularity)
  - §9 Error Handling & Monitoring (safe_call wrappers, latency tracking)
  - §10 Reproducibility Checklist (seeds, configs, artifacts)
- Documented the model-api `roberta-tool.js` update instructions (3 specific code change diffs)

---

### Vivek Bajaj
**Role: Model API Engineering Lead**

- Designed and implemented the complete `model-api/` deployment package:
  - `model-api/app.py` — FastAPI service with 3 endpoints (`GET /`, `POST /predict`, `POST /predict/batch`), Pydantic validation, CORS middleware, confidence bands, batch inference
  - `model-api/Dockerfile` — HuggingFace Spaces compliant: `python:3.11-slim`, non-root UID 1000, port 7860, model pre-downloaded at build time via `download_model.py`
  - `model-api/download_model.py` — isolated build-time weight pre-download script
  - `model-api/requirements.txt` — pinned to `transformers==4.47.1`, `tokenizers==0.21.1` to resolve the tokenizer format incompatibility bug (root cause: tokenizer.json serialized with newer tokenizers than originally pinned `0.19.1`)
  - `model-api/README.md` — HuggingFace Spaces YAML frontmatter, ASCII inference pipeline diagram, Mermaid deployment diagram, all endpoint docs, 6-step deployment guide
- Fixed Dockerfile tokenizer error: upgraded `tokenizers` from `0.19.1` to `0.21.1` and replaced inline heredoc `python -c "..."` (which broke Dockerfile linter) with separate `download_model.py` script
- Deployed the model API to HuggingFace Spaces
- Co-authored `docs/technical_doc.md` §3–5 (model architecture, training, evaluation)

---

### Vishwas Mehta
**Role: User Documentation & Tooling Lead**

- Authored `docs/user_guide.md` — complete user-facing guide covering:
  - Web-app: 5-step launch instructions, 3 input modes, result interpretation
  - Chrome extension: 5-step install guide, Gemini API key setup, LinkedIn usage walkthrough
  - 5 annotated example use cases (fraudulent WFH posting, legitimate SE role, overseas scam, startup ambiguity, government phishing)
  - Troubleshooting tables for both web-app and extension
  - Screenshot placeholders (to be filled manually)
- Wrote `create_google_doc.py` — Python script using Google Docs API to generate a styled Google Doc from `Final_Project_Report.md`, matching the sample report format (blue headings, justified body, team table, cover page, TOC, bordered data tables, code blocks in Courier New)
- Wrote Milestone 6 folder files: `milestone6.md`, `team_contribution.md` (this file), `notebooklm_slides_prompt.md`, `gaps.md`
- Updated `web-extension/SETUP.md` with Milestone 6 deployment instructions

---

## Effort Distribution

| Member | Milestone 6 Focus Area | Estimated Share |
|---|---|---|
| Arun Dutta | Final Report, Licenses, Future Work, Contribution Summary | ~25% |
| Hritik Roshan Maurya | API Docs, Technical Docs, Overview | ~25% |
| Vivek Bajaj | Model API engineering, HF Spaces deployment | ~25% |
| Vishwas Mehta | User Guide, Google Doc script, M6 folder files | ~25% |

---

## How Milestone 6 Closed the Loop

Milestones 1–5 produced research artifacts: notebooks, model weights, evaluation reports. Milestone 6 converted those artifacts into a product by:

1. **Wrapping the model** in a REST API (FastAPI + Docker → HuggingFace Spaces) so any client can call it without installing PyTorch
2. **Documenting everything** at three levels: technical (for developers), user-facing (for non-technical users), and API (for integrators)
3. **Writing the final report** in a format suitable for academic submission and external review
4. **Auditing the repo** for security, reproducibility, and consistency issues

The result is a project that can be reproduced, extended, or evaluated by anyone — the three core goals of Milestone 6.
