# Repository Structure Audit

**Project:** FraudGuard — Fake Job Listing Detection
**Audit Date:** 2026-04-15
**Auditor:** Milestone 6 Documentation Pass

---

## 1. Recommended vs. Actual Folder Structure

The recommended layout for a DS/AI project repo is:

```
project/
├── app/         ← Web application
├── api/         ← REST API (separate)
├── src/         ← Model source code (training, evaluation, utilities)
├── notebooks/   ← Jupyter notebooks
├── data/        ← Datasets (gitignored)
├── models/      ← Model checkpoints (gitignored)
├── docs/        ← Documentation
└── tests/       ← Test suite
```

**Actual structure vs. recommended:**

| Recommended | Actual | Status | Notes |
|---|---|---|---|
| `app/` | `web-app/` | ✅ Present (renamed) | `web-app/` is self-contained Flask app |
| `api/` | (within `web-app/routes/api.py`) | ⚠️ No separate `api/` dir | API is embedded in web-app, not separate |
| `src/` | `testing_work/src/` | ⚠️ Nested | Training code lives under `testing_work/src/`, not root `src/` |
| `notebooks/` | `notebook/` | ✅ Present (singular) | Contains the 2 key notebooks |
| `data/` | Not in repo (gitignored) | ✅ Correct | Dataset too large; gitignored |
| `models/` | Not in repo (gitignored) | ✅ Correct | Model on HuggingFace Hub |
| `docs/` | `docs/` | ✅ Present | Comprehensive milestone docs |
| `tests/` | None | ❌ Missing | No test suite exists |
| `web-extension/` | `web-extension/` | ✅ Present | Chrome extension |

---

## 2. What Was Found

### Root Level Files

| File | Status | Notes |
|---|---|---|
| `README.md` | ✅ Updated (Milestone 6) | Rewritten with full documentation |
| `requirements.txt` | ✅ Present | Full frozen pip environment (~186 packages) |
| `.gitignore` | ✅ Updated (Milestone 6) | Added `venv/`, `.venv/`, `web-app/results/`, `web-app/uploads/`, OS files |
| `Milestone 6 Deliverable Structure.pdf` | ⚠️ Binary in root | Should be moved to `docs/Milestone-6/` |
| `.github/workflows/main.yml` | ✅ Present | CI/CD workflow exists |

### `docs/` Folder

| Document | Status |
|---|---|
| `Milestone-0/` through `Milestone-5/` | ✅ All present |
| `overview.md` | ✅ Created (Milestone 6) |
| `technical_doc.md` | ✅ Created (Milestone 6) |
| `user_guide.md` | ✅ Created (Milestone 6) |
| `api_doc.md` | ✅ Created (Milestone 6) |
| `licenses.md` | ✅ Created (Milestone 6) |
| `future_work.md` | ✅ Created (Milestone 6) |
| `contribution_summary.md` | ✅ Created (Milestone 6) |
| `Final_Project_Report.md` | ✅ Created (Milestone 6) |
| `notebooklm_prompt.md` | ✅ Created (Milestone 6) |
| `GAPS_FIXED.md` | ✅ Created (Milestone 6) |
| `REPO_AUDIT.md` | ✅ This file |

### `notebook/` Folder

| File | Status | Notes |
|---|---|---|
| `transformer_fraud_classifier_v3_1.ipynb` | ✅ Present | Main training notebook |
| `rule_discovery_ebm.ipynb` | ✅ Present | EBM rule discovery notebook |
| `pipeline_demo.py` | ✅ Present | Pipeline smoke-test script |

### `web-app/` Folder

| Component | Status | Notes |
|---|---|---|
| `app.py` | ✅ | Flask factory with absolute template paths |
| `config.py` | ✅ | All env vars centralized |
| `core/` | ✅ | JobPosting schema + helpers |
| `routes/` | ✅ | main.py + api.py |
| `services/` | ✅ | analyzer.py, tool_runner.py, job_extractor.py, linkedin.py |
| `tools/` | ✅ | 12 tools (1 stub: company_registry) |
| `templates/` | ✅ | Full Jinja2 template set |
| `static/` | ✅ | CSS + JS |
| `WEBAPP.md` | ✅ | Detailed web-app documentation |
| `results/` | ⚠️ | Auto-created; now gitignored |
| `uploads/` | ⚠️ | Auto-created; now gitignored |

### `web-extension/` Folder

| Component | Status | Notes |
|---|---|---|
| `manifest.json` | ✅ | MV3, ES modules |
| `background.js` | ✅ | Pipeline orchestrator |
| `content.js` | ✅ | LinkedIn DOM scraping + overlay |
| `popup.html/css/js` | ✅ | API key settings UI |
| `lib/` | ✅ | LangChain-inspired JS framework |
| `tools/` | ✅ | 4 pipeline tools |
| `icons/` | ✅ | Extension icons |
| `README.md`, `SETUP.md`, `ARCHITECTURE.md`, `CHAIN_DOCS.md` | ✅ | Comprehensive docs |

### `testing_work/` Folder

| Component | Status | Notes |
|---|---|---|
| `AgenticWork/` | ✅ | CLI job parser agent (superseded by web-app/core) |
| `src/eval.py` | ✅ | Production evaluation script |
| `src/utils/` | ✅ | data.py, focal_loss.py, metrics.py |
| `src/tools/metadata_detector/` | ✅ | IsolationForest + rules engine |
| `src/job_analyzer/` | ✅ | Structured job parsing tools |
| `src/pipeline_demo.py` | ⚠️ | May duplicate `notebook/pipeline_demo.py` |
| `src/__pycache__/` | ⚠️ | Should not be in repo; now gitignored |

---

## 3. What Was Fixed (Milestone 6 Actions)

| Issue | Fix Applied |
|---|---|
| `.gitignore` missing `venv/`, `.venv/`, OS files | Updated `.gitignore` |
| `web-app/results/` and `uploads/` not gitignored | Added to `.gitignore` |
| `__pycache__/` in `testing_work/src/` | `.gitignore` already covers `__pycache__/` — tracked files can be removed with `git rm -r --cached testing_work/src/__pycache__/` |
| No Milestone 6 documentation | Created 10 new docs files |
| README.md outdated and incomplete | Rewritten from scratch |

---

## 4. What Still Needs Manual Action

The following items require manual action by the team and cannot be automated:

| Item | Priority | Action Required |
|---|---|---|
| **Remove `__pycache__` from git tracking** | High | Run: `git rm -r --cached testing_work/src/__pycache__/` then commit |
| **Move `Milestone 6 Deliverable Structure.pdf` to docs/** | Low | `mv "Milestone 6 Deliverable Structure.pdf" docs/Milestone-6/` |
| **Add `.venv/` removal from git tracking** | High | Run: `git rm -r --cached .venv/` (if it was ever tracked) |
| **Insert screenshots in `docs/user_guide.md`** | High | Take screenshots of web-app and extension; add to appropriate `<!-- INSERT SCREENSHOT -->` placeholders |
| **Insert demo video link in `README.md`** | Medium | Record a demo video; replace `<!-- INSERT DEMO VIDEO LINK HERE -->` |
| **Fill in team email addresses in `docs/future_work.md`** | Low | Replace `<!-- EMAIL -->` placeholders |
| **Implement `tool_company_registry.py`** | Medium | See `docs/future_work.md` Section 2.2 |
| **Add pytest test suite** | Medium | See `docs/future_work.md` Section 2.4 |
| **Create a minimal `requirements-webapp.txt`** | Low | List only webapp dependencies without CUDA packages |
| **Update `docs/Milestone-6/` folder** | Low | Create folder, move any M6 artifacts |

---

## 5. `requirements.txt` Audit

The current `requirements.txt` is a **complete frozen pip environment** exported from the development machine. It includes:
- CUDA/NVIDIA packages (`cuda-bindings`, `nvidia-*`) — will fail on non-CUDA machines
- All dependencies (training + web-app + development tools)

**Verified present:** All core dependencies for both training pipeline and web-app confirmed present (see `docs/GAPS_FIXED.md` Section 6 for full verification table).

**Recommendation:** Create a separate `requirements-webapp.txt` with only web-app dependencies for lightweight deployment:
```
flask>=3.0
markupsafe>=3.0
langchain-openai>=1.0
langchain-community>=0.4
pydantic>=2.0
requests>=2.30
beautifulsoup4>=4.14
ddgs>=9.0
trafilatura>=2.0
python-whois>=0.9
email-validator>=2.3
phonenumbers>=9.0
```

---

## 6. CI/CD Status

The `.github/workflows/main.yml` CI workflow exists. Its current configuration was not audited in detail — the team should verify it runs successfully on push to main.

---

## 7. Summary Scorecard

| Category | Score | Notes |
|---|---|---|
| Documentation coverage | 9/10 | All 10 M6 docs created; screenshots pending |
| Code organization | 7/10 | `testing_work/src/` nesting is non-standard; no `tests/` |
| `.gitignore` completeness | 9/10 | Updated; tracked binary files may need manual removal |
| `requirements.txt` | 7/10 | Complete but heavy; no lightweight webapp variant |
| Model deployment | 10/10 | Published to HuggingFace Hub |
| Web-app quality | 9/10 | Self-contained, well-documented; company registry stub |
| Extension quality | 9/10 | Fully functional; LinkedIn-only |
| Test coverage | 0/10 | No test suite |
| **Overall** | **7.5/10** | Solid project with clear improvement paths |
