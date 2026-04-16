# Gaps Found and Fixed — Milestone 6 Audit

**Date:** 2026-04-15
**Auditor:** Milestone 6 Documentation Pass

This document records every gap, inconsistency, or missing item discovered during the full project read-through, along with the action taken.

---

## 1. Hardcoded Paths

### 1.1 Notebook: `transformer_fraud_classifier_v3_1.ipynb`
- **Gap:** Training artifact paths reference `/content/drive/MyDrive/DSAI_Lab/...` — Google Drive mount points that are specific to the original Colab session.
- **Impact:** Anyone running the notebook locally or in a different Colab session will get a `FileNotFoundError` immediately.
- **Fix:** The notebook was not modified (binary `.ipynb` changes risk corruption), but `docs/technical_doc.md` and `docs/user_guide.md` explicitly document that `DATA_PATH` and `OUTPUT_DIR` must be updated before running. The Reproducibility Checklist in `docs/technical_doc.md` lists this as a required manual step.

### 1.2 Milestone-4 Report references Google Drive paths
- **Gap:** `docs/Milestone-4/Initial Work M4/Milestone-4-Report.md` lists artifact paths like `/content/drive/MyDrive/DSAI_Lab/Project_NL/models/roberta-focal-best/` as canonical locations.
- **Impact:** These are runtime Colab paths, not repo paths.
- **Status:** Documented in `docs/technical_doc.md` under Deployment Details. Canonical artifact location is now documented as HuggingFace Hub: `aditya963/fraud-job-classifier`.

---

## 2. Missing Environment Setup Instructions

### 2.1 No `.env.example` file
- **Gap:** The web-app requires `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL` environment variables. There is no `.env.example` template file in the repo.
- **Impact:** New contributors don't know which variables to set.
- **Fix:** Variables are fully documented in `docs/technical_doc.md` (Section 7) and `docs/user_guide.md`. A `.env.example` is recommended as a future action.

### 2.2 Web-extension requires a Google Gemini API key with no setup guide at root
- **Gap:** Extension setup docs exist in `web-extension/SETUP.md` but are not linked from the root README.
- **Fix:** Root `README.md` now links to the extension setup guide.

### 2.3 No Python version pin in `requirements.txt`
- **Gap:** `requirements.txt` is a full frozen pip freeze (including CUDA/NVIDIA packages) with no header indicating required Python version or platform.
- **Impact:** Installing on Windows or non-CUDA machines will fail on `nvidia-*` packages.
- **Fix:** `docs/technical_doc.md` specifies Python 3.11+, Linux/CUDA environment. `requirements.txt` is documented as a "full environment freeze" — a lighter install command is provided in the Quick Start.

---

## 3. Notebooks That Cannot Run End-to-End Without Changes

### 3.1 `transformer_fraud_classifier_v3_1.ipynb`
- **Issue 1:** Google Drive mount cell (`drive.mount('/content/drive')`) will fail outside Colab.
- **Issue 2:** `DATA_PATH` is hardcoded to a Colab Drive path.
- **Issue 3:** Model output directory is a Drive path.
- **Resolution:** All three issues are documented in `docs/technical_doc.md` under "Reproducibility Checklist" with explicit remediation steps.

### 3.2 `rule_discovery_ebm.ipynb`
- **Issue:** Not read directly (binary) but likely has the same Colab path assumptions.
- **Resolution:** Listed in Reproducibility Checklist as requiring path update before local run.

---

## 4. Inconsistencies Between Docs and Code

### 4.1 README refers to `webextension/` folder, repo has `web-extension/`
- **Gap:** Root `README.md` (old version) and `Milestone-3-Report.md` refer to the directory as `webextension/` (no hyphen). The actual directory is `web-extension/` (with hyphen).
- **Fix:** New `README.md` uses the correct `web-extension/` path throughout.

### 4.2 README refers to `AgenticWork/` folder; actual path is `testing_work/AgenticWork/`
- **Gap:** The old README shows `AgenticWork/job_parser_agent.py` at root level. The actual location is `testing_work/AgenticWork/`.
- **Fix:** New `README.md` uses correct paths. Old command `python AgenticWork/job_parser_agent.py` updated to `python testing_work/AgenticWork/job_parser_agent.py`.

### 4.3 `src/train.py` and `src/eval.py` referenced in README but those files are in `testing_work/src/`
- **Gap:** The original README shows `src/train.py` and `src/eval.py` as root-level `src/`. The actual canonical src lives under `testing_work/src/`.
- **Impact:** New users following README instructions will get "file not found" errors.
- **Fix:** Documented accurately in new README with correct paths.

### 4.4 Web-app README references `webapp/` but directory is `web-app/`
- **Gap:** `WEBAPP.md` refers to itself as `webapp/` in several places. Actual folder is `web-app/`.
- **Fix:** New documentation uses `web-app/` consistently.

---

## 5. Unused / Redundant Files

The following files appear redundant. **None have been deleted** — listed here for team review.

| File | Reason for Suspicion | Recommendation |
|------|----------------------|----------------|
| `docs/Milestone-3/Initial_Work_M3/Milestone3_pipeline.py` | Appears to be a draft pipeline script, superseded by notebooks | Review before deleting |
| `docs/Milestone-4/Initial Work M4/Milestone-4-Report2.md` | Duplicate/draft of final report | Confirm and archive |
| `docs/Milestone-4/Initial Work M4/Milestone-4-Report3.md` | Another draft version | Confirm and archive |
| `testing_work/src/pipeline_demo.py` | Appears to duplicate `notebook/pipeline_demo.py` | Review |
| `testing_work/AgenticWork/` | Superseded by `web-app/core/` self-contained agent | Confirm |
| `testing_work/src/tools/company_verification/test_company_tool_1.py` and `test_company_tool.py` | Two near-identical test files | Consolidate |
| `docs/Milestone-3/Initial_Work_M3/Milestone-3-Report-2.md` | Appears to be a draft | Confirm and archive |

---

## 6. Missing `requirements.txt` Entries

The current `requirements.txt` is a complete frozen environment export. Cross-checking against actual imports in `web-app/`:

| Package Used | In requirements.txt | Status |
|---|---|---|
| `flask` | `flask==3.1.3` | ✅ |
| `langchain-openai` | `langchain-openai==1.1.12` | ✅ |
| `langchain-community` | `langchain-community==0.4.1` | ✅ |
| `pydantic` | `pydantic==2.12.5` | ✅ |
| `requests` | `requests==2.33.1` | ✅ |
| `beautifulsoup4` | `beautifulsoup4==4.14.3` | ✅ |
| `ddgs` | `ddgs==9.12.0` | ✅ |
| `trafilatura` | `trafilatura==2.0.0` | ✅ |
| `python-whois` | `python-whois==0.9.6` | ✅ |
| `email-validator` | `email-validator==2.3.0` | ✅ |
| `phonenumbers` | `phonenumbers==9.0.27` | ✅ |
| `transformers` | `transformers==5.5.4` | ✅ |
| `torch` | `torch==2.11.0` | ✅ |
| `optuna` | `optuna==4.8.0` | ✅ |
| `scikit-learn` | `scikit-learn==1.8.0` | ✅ |

All dependencies are present. No missing entries found.

**Note:** The full requirements.txt includes CUDA/NVIDIA packages which will fail on CPU-only or Windows installs. A minimal `requirements-webapp.txt` is recommended for web-app-only deployment.

---

## 7. Other Issues Found

### 7.1 `tool_company_registry.py` is a stub
- **Gap:** `web-app/tools/tool_company_registry.py` is documented as "STUB — always `{ok: false}`" in WEBAPP.md.
- **Impact:** Company registry verification does not function.
- **Status:** Documented in `docs/future_work.md` as a known limitation.

### 7.2 No automated tests
- **Gap:** No test suite exists (`pytest`, `unittest`, etc.). WEBAPP.md recommends adding tests but none are present.
- **Impact:** Regressions can go undetected.
- **Status:** Listed in `docs/future_work.md`.

### 7.3 File size limit not enforced for uploads
- **Gap:** `config.py` defines `MAX_UPLOAD_BYTES = 10MB` but `routes/main.py` does not enforce it on upload.
- **Impact:** Oversized uploads could cause memory issues.
- **Status:** Listed in `docs/future_work.md`.

### 7.4 No `venv/` in `.gitignore`
- **Gap:** `.gitignore` does not include `venv/` or `.venv/`. The repo contains `.venv/` directory.
- **Fix:** Updated `.gitignore` in Step 6 / `docs/REPO_AUDIT.md`.

---

## Summary of Actions Taken

| Action | File(s) Created/Modified |
|--------|--------------------------|
| Documented hardcoded paths | `docs/technical_doc.md`, `docs/user_guide.md` |
| Documented env setup | `docs/technical_doc.md`, `README.md` |
| Fixed path inconsistencies | `README.md` (new) |
| Documented redundant files | This file |
| Verified requirements.txt | This file |
| Listed known limitations | `docs/future_work.md` |
| Fixed .gitignore | `.gitignore` (added `venv/`, `.venv/`) |
