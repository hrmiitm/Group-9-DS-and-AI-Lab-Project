# Milestone 6 — Gap Analysis & Fix Guide

**Purpose:** Documents everything in the Milestone 6 Deliverable Structure PDF that is not yet 100% complete, along with exact steps to close each gap.

---

## Summary

| # | Gap | Severity | Fix Time |
|---|---|---|---|
| 1 | Demo video not recorded | Medium | 2–3 hours |
| 2 | Screenshots missing from user_guide.md | Medium | 30 min |
| 3 | Demo video link missing from README.md | Low | 5 min |
| 4 | Team email addresses are placeholders in future_work.md | Low | 5 min |
| 5 | Institution name missing from Final_Project_Report.md | Low | 2 min |
| 6 | HuggingFace Spaces URL not updated in roberta-tool.js | High | 10 min |
| 7 | model-api base URL placeholder in api_doc.md | Medium | 5 min |
| 8 | company_registry tool is an unimplemented stub | Low | 4–8 hours |
| 9 | .venv/ and __pycache__ may still be tracked in git | Medium | 15 min |
| 10 | No automated tests | Low | Optional |
| 11 | Upload size limit not enforced in routes/main.py | Low | 15 min |

---

## Detailed Fix Instructions

---

### Gap 1 — Demo video not recorded

**Requirement (PDF §5):** Record a short demo video (10–20 minutes) and embed the YouTube or HuggingFace link in the README.

**What's missing:** No video has been recorded.

**How to fix:**
1. Start a screen recording (OBS, Loom, or built-in system recorder)
2. Record the following walkthrough (~10–15 min):
   - Open the Flask web-app at `localhost:5000`
   - Demo input 1: paste a fraudulent job posting (WFH, "no experience needed")
   - Demo input 2: paste a legitimate job posting (Senior Engineer, named company)
   - Demo input 3: upload a PDF job posting
   - Show the verdict page, tool cards, and LLM report for each
   - Open LinkedIn in Chrome, click the extension button, show the overlay verdict
   - Briefly show `http://localhost:5000` → show the model probability score
3. Upload to YouTube (unlisted is fine) or HuggingFace Spaces demo video
4. In `README.md`, find the line `<!-- DEMO VIDEO LINK HERE -->` and replace with:
   ```markdown
   ## Demo
   [![FraudGuard Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtu.be/YOUR_VIDEO_ID)
   ```

---

### Gap 2 — Screenshots missing from user_guide.md

**Requirement (PDF §2C):** Screenshots showing example interactions.

**What's missing:** `docs/user_guide.md` contains screenshot placeholders like `![Screenshot: ...](screenshots/...)` but no actual image files exist.

**How to fix:**
1. Create the folder:
   ```bash
   mkdir -p docs/screenshots
   ```
2. Take screenshots of:
   - Flask web-app home page (`localhost:5000`)
   - A results page showing LIKELY_FAKE verdict with tool cards
   - A results page showing SAFE verdict
   - Chrome extension "Analyze Job" button on a LinkedIn page
   - Chrome extension FRAUDULENT overlay
   - Chrome extension LEGITIMATE overlay
3. Save as PNG files named to match the placeholders in `user_guide.md`
4. Replace placeholder alt text with actual image paths

---

### Gap 3 — Demo video link missing from README.md

**Requirement (PDF §1):** Optional demo video embedded in README.

**How to fix:** After recording the demo video (Gap 1), add it to `README.md`:
```markdown
## Demo Video
> [Watch the 12-minute walkthrough on YouTube](https://youtu.be/YOUR_ID)
```
Add this section after the Quick Start section in `README.md`.

---

### Gap 4 — Team email addresses are placeholders in future_work.md

**Requirement (PDF §2F):** Contacts / maintainers with real contact information.

**What's missing:** `docs/future_work.md` contains placeholder text for maintainer emails.

**How to fix:**
Open `docs/future_work.md` and find the Contacts section. Replace the placeholder lines with real contact details:

```markdown
## Contacts / Maintainers

| Name | Role | Contact |
|---|---|---|
| Vivek Bajaj | Model training & API | [your email] |
| Hritik Roshan Maurya | Web-app backend | hrmiitm@example.com |
| Vishwas Mehta | Chrome extension | mehtavishwas989@gmail.com |
| Arun Dutta | Documentation | [your email] |
```

---

### Gap 5 — Institution name missing from Final_Project_Report.md

**What's missing:** Line 12 of `docs/Final_Project_Report.md`:
```
| **Institution** | <!-- INSTITUTION NAME HERE --> |
```

**How to fix:** Replace with:
```markdown
| **Institution** | Indian Institute of Technology Madras |
```

---

### Gap 6 — HuggingFace Spaces URL not updated in roberta-tool.js *(HIGH PRIORITY)*

**What's missing:** `web-extension/tools/roberta-tool.js` line 13 still points to the HuggingFace Inference API:
```js
const HF_MODEL_URL = "https://api-inference.huggingface.co/models/aditya963/fraud-job-classifier";
```

Now that `model-api/` is deployed to HuggingFace Spaces, this should point to the new endpoint.

**How to fix:**

Step 1 — Update the URL:
```js
// Before:
const HF_MODEL_URL = "https://api-inference.huggingface.co/models/aditya963/fraud-job-classifier";

// After (replace YOUR-USERNAME with your HF username):
const HF_MODEL_URL = "https://YOUR-USERNAME-fraudguard-api.hf.space/predict";
```

Step 2 — Update the request body format (HF Inference API → JobPosting format):
```js
// Before (HF Inference API format):
body: JSON.stringify({ inputs: standardizedText })

// After (model-api JobPosting format):
body: JSON.stringify({ description: standardizedText })
```

Step 3 — Update the response parsing (different response structure):
```js
// Before (HF Inference API returns [[{label, score}, {label, score}]]):
const score = data[0].find(x => x.label === "LABEL_1")?.score ?? 0;

// After (model-api returns {fraud_probability, verdict, confidence, ...}):
const score = data.fraud_probability ?? 0;
const verdict = data.verdict;  // "FRAUDULENT" or "LEGITIMATE"
```

---

### Gap 7 — model-api base URL placeholder in api_doc.md

**What's missing:** `docs/api_doc.md` contains placeholder `YOUR-USERNAME-fraudguard-api.hf.space` in all curl examples.

**How to fix:** After deploying to HuggingFace Spaces, do a global find-and-replace:
```bash
# In docs/api_doc.md and model-api/README.md:
sed -i 's/YOUR-USERNAME-fraudguard-api.hf.space/ACTUAL-SPACE-URL/g' docs/api_doc.md model-api/README.md
```
Or manually replace in the files.

---

### Gap 8 — company_registry tool is an unimplemented stub

**What's missing:** `web-app/tools/tool_company_registry.py` returns placeholder data. This is documented in `docs/REPO_AUDIT.md`.

**How to fix (if time permits):**
```python
# In web-app/tools/tool_company_registry.py
# Implement using one of:
# 1. MCA (Ministry of Corporate Affairs) India API — company registration lookup
# 2. WHOIS lookup for company domain registration date
# 3. OpenCorporates API (free tier) — https://api.opencorporates.com

import requests

def check_company_registry(company_name: str) -> dict:
    # OpenCorporates free API:
    url = f"https://api.opencorporates.com/companies/search?q={company_name}&jurisdiction_code=in"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        companies = data.get("results", {}).get("companies", [])
        return {
            "found": len(companies) > 0,
            "count": len(companies),
            "top_result": companies[0] if companies else None,
        }
    return {"found": False, "error": "API unavailable"}
```

**Severity:** Low — the other 11 tools still run. The stub returns a neutral result that does not affect the final verdict.

---

### Gap 9 — .venv/ and __pycache__ may still be tracked in git

**What's missing:** Even though `.gitignore` now includes `venv/`, `.venv/`, `__pycache__/`, if these were committed before the `.gitignore` update, they are still tracked.

**How to fix:**
```bash
# Remove from git tracking (does NOT delete the files locally)
git rm -r --cached .venv/ 2>/dev/null
git rm -r --cached __pycache__/ 2>/dev/null
git rm -r --cached web-app/__pycache__/ 2>/dev/null

# Commit the removal
git add .gitignore
git commit -m "Remove tracked venv and pycache from git"
```

---

### Gap 10 — No automated tests

**Requirement (PDF §1):** Not explicitly required, but good practice.

**What's missing:** No `tests/` directory or pytest suite.

**How to fix (optional, if time permits):**
```bash
mkdir tests
# Create tests/test_api.py with:
# - test that build_input_text() produces correct [SEP]-joined string
# - test that /predict returns 200 with fraud_probability field
# - test batch endpoint with 2 postings
# - test health endpoint returns status: "ok"
```

**Priority:** Optional. Skip if time is short — the repo audit notes this as a known gap.

---

### Gap 11 — Upload size limit not enforced in routes/main.py

**What's missing:** `web-app/routes/main.py` does not check the file size of uploaded PDFs/DOCX/HTML files before processing. A large file could cause a memory error or timeout.

**How to fix:**
```python
# In web-app/routes/main.py, inside the file upload handler:
MAX_FILE_SIZE_MB = 5
file = request.files.get("file")
file.seek(0, 2)  # seek to end
file_size_mb = file.tell() / (1024 * 1024)
file.seek(0)     # reset
if file_size_mb > MAX_FILE_SIZE_MB:
    return jsonify({"error": f"File too large ({file_size_mb:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB."}), 413
```

**Alternatively**, set Flask's `MAX_CONTENT_LENGTH` in `web-app/app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB
```

---

## Not Gaps — Already Done

These items were required by the PDF and ARE complete:

| Requirement | File | Done |
|---|---|---|
| README with Quick Start | `README.md` | ✅ |
| Environment setup instructions | `docs/technical_doc.md §1` | ✅ |
| Data pipeline documentation | `docs/technical_doc.md §2` | ✅ |
| Model architecture + hyperparameters | `docs/technical_doc.md §3–4` | ✅ |
| Evaluation metrics | `docs/technical_doc.md §5` | ✅ |
| Inference pipeline with code snippet | `docs/technical_doc.md §6` | ✅ |
| Deployment platform details | `docs/technical_doc.md §7` | ✅ |
| REST API documentation | `docs/api_doc.md` | ✅ |
| User guide (non-technical) | `docs/user_guide.md` | ✅ |
| Licensing | `docs/licenses.md` | ✅ |
| Future work | `docs/future_work.md` | ✅ |
| Final Project Report | `docs/Final_Project_Report.md` | ✅ |
| Team contributions | `docs/contribution_summary.md` + this folder | ✅ |
| requirements.txt | `requirements.txt`, `model-api/requirements.txt` | ✅ |
| .gitignore updated | `.gitignore` | ✅ |
| Reproducibility checklist | `docs/technical_doc.md §10` | ✅ |
