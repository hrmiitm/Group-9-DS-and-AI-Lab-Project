# FraudGuard — Web Extension Reference

> Two-part reference for the FraudGuard Chrome extension. Each section is self-contained and structured for slide generation.

---

# Part 1 — Web Extension (v1)

## What It Is

FraudGuard v1 is a **Chrome Extension (Manifest V3)** that detects fraudulent LinkedIn job listings in real time using a **LangChain-inspired tool pipeline** running inside the browser. It uses no external backend — all AI calls go directly from the extension to HuggingFace (RoBERTa) and Google Gemini APIs.

---

## Architecture Overview

```
LinkedIn Page
     │
     ▼
[content.js]  ←── Injected automatically on linkedin.com
   • Scrapes job title, company, description, links
   • Injects the FraudGuard button + results overlay
   • Sends job data to background.js via chrome.runtime.sendMessage
     │
     ▼
[background.js]  ←── MV3 Service Worker (Orchestrator)
   • Holds the ToolRegistry
   • Builds and runs the analysis pipeline
   • Sends result back to content.js
     │
     ├──► [RoBERTaTool]          → HuggingFace Inference API
     ├──► [DetectLinksTool]      → Finds company/social links in job text
     ├──► [LinkScraperTool]      → Fetches and reads each discovered link
     ├──► [ContentAggregatorTool]→ Merges scraped content into one context
     └──► [JobAnalyzerTool]      → Google Gemini 2.5-Flash (final verdict)
```

---

## Step-by-Step Flow

### Step 1 — User Opens a LinkedIn Job Page
LinkedIn is a **Single Page Application (SPA)**. A `MutationObserver` in `content.js` watches for DOM changes and re-injects the FraudGuard button whenever the job detail panel loads.

### Step 2 — DOM Scraping (`content.js`)
`scrapeJobData()` uses multi-selector fallback arrays to extract:
- **Job title** — tries 5 CSS selectors, then scans all `<h1>` tags
- **Company name** — tries company-name selectors, then `/company/` href links
- **Description** — tries description containers, then TreeWalker text search
- **Location, salary, employment type** — additional metadata fields

> **Examiner Q:** *Why use multiple selectors?* LinkedIn frequently changes its CSS class names. Fallback arrays ensure the scraper doesn't break on UI updates.

### Step 3 — Message Passing to Service Worker
```
content.js  →  chrome.runtime.sendMessage({ type: "ANALYZE_JOB", data: jobData })
background.js  →  chrome.tabs.sendMessage(tabId, result)  →  content.js
```
`return true` is used in the message listener to keep the channel open for the async response.

> **Examiner Q:** *Why does background.js need `return true`?* Chrome closes the message channel after the listener returns. `return true` signals that the response will be sent asynchronously.

### Step 4 — Parallel Pipeline (`background.js`)

Two branches run **in parallel** using `Promise.allSettled`:

**Branch A — ML Scoring**
```
RoBERTaTool
  → POST to HuggingFace Inference API
  → Model: aditya963/fraud-job-classifier (fine-tuned RoBERTa-base)
  → Returns: fraud_probability (0-1), threshold = 0.87
```

**Branch B — Link Evidence**
```
DetectLinksTool  →  finds URLs in job description
LinkScraperTool  →  fetches each URL, extracts visible text
ContentAggregatorTool  →  combines all scraped text into one blob
```

Both complete, then:

**Step 5 — LLM Analysis (`JobAnalyzerTool`)**
```
Input: job text + RoBERTa score + scraped link content
Model: Google Gemini 2.5-Flash (via generativelanguage.googleapis.com)
Output: verdict (SAFE / SUSPICIOUS / LIKELY_FAKE) + detailed reasoning
```

### Step 6 — Results Rendered in Overlay
`content.js` receives the verdict and injects a **glassmorphism overlay panel** directly into the LinkedIn page DOM with:
- Verdict badge (color-coded)
- RoBERTa probability score
- LLM reasoning report
- Scraped link evidence

---

## LangChain-Inspired Architecture

v1 implements its own lightweight agent framework in `lib/`:

| Component | Role |
|-----------|------|
| `ToolRegistry` | Registers all tools by name + category; looks them up by name |
| `PipelineBuilder` | Chains tools together in a defined execution order |
| `PipelineConfig` | Holds runtime config (API keys, mode flags) |
| `ContentAggregatorTool` | Merges outputs from multiple tools into a single context |

> **Examiner Q:** *Why build a custom framework instead of using LangChain directly?* Chrome extensions cannot run Node.js. LangChain JS requires a Node environment. The custom framework replicates the tool-chaining pattern in pure browser-compatible ES modules.

---

## Analysis Modes

| Mode | Trigger | What Runs |
|------|---------|-----------|
| **Full Analysis** | "Analyze Job" button | RoBERTa + Link scraping + Gemini |
| **Quick Analysis** | "Quick Check" button | RoBERTa + Gemini only (no link scraping) |

---

## Key Files (v1)

| File | Purpose |
|------|---------|
| `manifest.json` | MV3 config, permissions, service worker declaration |
| `content.js` | LinkedIn scraper + UI injection |
| `background.js` | Tool registry + pipeline orchestration |
| `tools/roberta-tool.js` | HuggingFace API caller |
| `tools/link-detector.js` | URL extraction from job text |
| `tools/link-scraper.js` | External URL content fetcher |
| `tools/job-analyzer-tool.js` | Gemini LLM integration |
| `lib/langchain-core.js` | ToolRegistry implementation |
| `lib/pipeline.js` | PipelineBuilder + ContentAggregatorTool |
| `popup.html/js` | Settings UI for API key storage |

---

## Storage & API Keys

API keys are stored in `chrome.storage.local` (sandboxed per extension, not accessible by web pages). The popup UI lets the user enter their Gemini API key, which is then read by the service worker on each analysis.

> **Examiner Q:** *Is it safe to store API keys in chrome.storage.local?* It is isolated from web pages and other extensions. It is not end-to-end encrypted — a risk mitigated by the key being user-owned and the extension being local only.

---

# Part 2 — Web Extension v2

## What Changed and Why

v1 had a critical bug: Chrome's MV3 service worker is **terminated after ~30 seconds of inactivity**. Any in-flight API call from the service worker would silently fail, causing the "Analyzing..." spinner to hang forever.

**Fix:** Move ALL API calls into `content.js`. Content scripts run in the page's context — they are never terminated by Chrome. The service worker in v2 is a 5-line stub.

---

## Architecture Overview (v2)

```
LinkedIn Page
     │
     ▼
[content.js]  ←── Does EVERYTHING (scraping + all API calls + UI)
     │
     ├─ Step 1: Health check → GET /health
     ├─ Step 2: Batch tools  → POST /api/v1/run-batch  (13 tools in one call)
     ├─ Step 3: LLM summary  → POST /api/v1/llm/final-summary
     └─ Step 4: Render results overlay

[background.js]  ←── Minimal stub (ping only, no API calls)

[popup.js]  ←── Settings: backend URL, optional LLM API key
```

---

## Deployed Services

| Service | URL | Technology |
|---------|-----|-----------|
| **Backend API** | `https://hrmhrmhrm-company-backend-api.hf.space` | FastAPI on HuggingFace Spaces |
| **RoBERTa Model** | `https://hrmhrmhrm-roberta-model.hf.space` | HuggingFace Inference, `aditya963/fraud-job-classifier` |
| **Frontend App** | `https://hrmhrmhrm-company-frontend-app.hf.space` | React/Vite |

The extension talks **only to the Backend API**. The backend internally calls the RoBERTa model and LLM.

---

## Step-by-Step Flow (v2)

### Step 1 — Health Check
```
GET /health  (timeout: 12s, retries once with 50s for HF cold start)
```
HuggingFace Spaces "sleep" when idle. The health check wakes them and confirms connectivity before proceeding.

### Step 2 — Batch Tool Call (13 Tools in One Request)
```
POST /api/v1/run-batch
Body: [ { tool_name: "...", ...kwargs }, ... ]
Timeout: 90 seconds
```

All 13 tools are dispatched in a **single HTTP call**. The backend runs them and returns all results at once.

#### The 13 Tools

| # | Tool Name | Input | What It Checks |
|---|-----------|-------|----------------|
| 1 | `roberta_classifier` | job_text | ML fraud probability (fine-tuned RoBERTa) |
| 2 | `scam_signals` | job_text | Keyword patterns: "wire transfer", "advance fee", "no experience" |
| 3 | `domain_reputation` | company website URL | Domain age, registrar risk score |
| 4 | `website_verify` | company website URL | Is site live? Valid SSL? |
| 5 | `social_profiles` | company_name | LinkedIn, Twitter, Facebook presence count |
| 6 | `job_boards` | job_title, company_name | Cross-reference Indeed, Glassdoor, etc. |
| 7 | `company_wikipedia` | company_name | Wikipedia presence (legitimacy signal) |
| 8 | `company_news` | company_name | News article count (media presence) |
| 9 | `contact_info` | job_text | Suspicious contact patterns (personal Gmail, etc.) |
| 10 | `salary_analysis` | job_text | Unrealistic pay claims |
| 11 | `requirements_analysis` | job_text | Vague / no-experience-required red flags |
| 12 | `company_registration` | company_name | Business registry lookup |
| 13 | `location_verify` | location, company | Address plausibility check |

### Step 3 — LLM Final Summary
```
POST /api/v1/llm/final-summary
Body: {
  job_dict: { title, company_name, location, salary_range, description },
  tool_results: { ...all 13 tool outputs },
  llm_config: { api_key?, base_url?, model? }  ← optional, from popup settings
}
Timeout: 50 seconds
```

The LLM (default: `openai/gpt-4.1-mini` via AIPipe/OpenRouter) receives ALL tool evidence and writes a human-readable fraud report with a final verdict.

### Step 4 — Heuristic Fallback (if LLM fails)
If the LLM call times out or fails, `heuristicVerdict()` computes a **weighted fraud score** using all tool results:

| Tool | Max Points | Trigger |
|------|-----------|---------|
| `roberta_classifier` | 40 pts | fraud_probability × 40 |
| `scam_signals` | 25 pts | HIGH risk = 25, MEDIUM = 12 |
| `domain_reputation` | 15 pts | domain < 30 days old = 15, < 180 days = 8 |
| `website_verify` | 8 pts | site offline = 8, no SSL = 4 |
| `social_profiles` | 8 pts | 0 platforms found = 8, 1 found = 3 |
| `job_boards` | 6 pts | not listed anywhere = 6 |
| `company_wikipedia` | 4 pts | no Wikipedia page = 4 |
| `company_news` | 4 pts | 0 news articles = 4 |

**Verdict thresholds:**
- Score ≥ 55 → `LIKELY_FAKE`
- Score ≥ 30 → `SUSPICIOUS`
- Score < 30 → `SAFE`

> **Examiner Q:** *What happens if the LLM is unavailable?* The extension degrades gracefully — the weighted heuristic uses all 13 tool outputs to produce a verdict, and the UI shows a signal-by-signal breakdown instead of the LLM narrative.

---

## LinkedIn DOM Scraping (v2)

`scrapeJobData()` uses a **multi-selector fallback system** for each field:

```
Job Title  → tries 5 CSS selectors → scans all <h1> → parses document.title
Company    → tries 5 CSS selectors → scans <a href*="/company/"> links
Description→ tries 3 containers → TreeWalker for "About the Job" headings
```

`waitForJobContent()` retries scraping up to 8 times (500ms apart) to handle LinkedIn's lazy-loaded content.

> **Examiner Q:** *Why retry scraping?* LinkedIn is an SPA. The job detail panel may render after the initial DOM is ready. Retrying with delays catches content that loads asynchronously.

---

## MutationObserver (SPA Navigation)

```javascript
new MutationObserver(debounce(onDomChange, 500))
  .observe(document.body, { childList: true })
```

When LinkedIn navigates to a new job (without a full page reload), the observer fires, re-checks the URL, and re-injects the FraudGuard button if it was removed.

**Debounce to 500ms + `childList` only** (not `subtree`) prevents performance degradation from LinkedIn's high-frequency DOM mutations.

---

## Key Files (v2)

| File | Purpose |
|------|---------|
| `manifest.json` | MV3, `host_permissions: https://*/*` for cross-origin fetch |
| `content.js` | Everything: scraper + API calls + heuristic + UI overlay |
| `background.js` | 5-line stub (ping response only) |
| `content.css` | Glassmorphism overlay styles |
| `popup.html/js/css` | Backend URL + LLM key settings UI |

---

## Error Handling

| Error | Cause | User-Facing Message |
|-------|-------|---------------------|
| Health check fails (12s) | HF Space sleeping | Retries with 50s cold-start timeout |
| Health check fails (50s) | Backend down | "Cannot reach backend — verify URL in popup" |
| Batch call fails | Network / server error | Falls back to individual core tool calls |
| LLM summary fails | LLM quota / timeout | Falls back to heuristic verdict |
| Extension context invalidated | Extension reloaded mid-session | "Please refresh this page (Cmd+R)" |

---

## v1 vs v2 — Key Differences

| Aspect | v1 | v2 |
|--------|----|----|
| API call location | `background.js` (service worker) | `content.js` (page context) |
| Backend | None — direct HF + Gemini calls | FastAPI at HuggingFace Spaces |
| Number of tools | 5 (client-side) | 13 (server-side) |
| LLM | Gemini 2.5-Flash (direct API) | OpenRouter/AIPipe (configurable) |
| MV3 termination bug | Present — hangs after 30s | Fixed — content scripts never terminate |
| Verdict logic | Binary RoBERTa threshold + Gemini | Weighted 13-signal heuristic + LLM summary |
| Link scraping | Yes (client-side fetch) | Handled by backend tools |
