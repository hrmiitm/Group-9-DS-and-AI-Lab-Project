# Webapp — Fraud Job Detector

Self-contained Flask application for detecting fraudulent job postings.
**Fully independent** — all code lives inside `webapp/`. No imports from `src/`.

## How to Run

```bash
# From project root:
export OPENAI_API_KEY=$AIPIPE_TOKEN
export OPENAI_BASE_URL=https://aipipe.org/openrouter/v1
export LLM_MODEL=openai/gpt-4o-mini   # any OpenRouter model slug

python webapp/app.py
# → http://localhost:5000
```

Switch models without code changes — just change `LLM_MODEL`:
```bash
LLM_MODEL=anthropic/claude-3-5-sonnet  python webapp/app.py
LLM_MODEL=google/gemini-2.5-flash      python webapp/app.py
```

---

## Directory Structure

```
webapp/
│
├── app.py               Flask factory. Adds webapp/ to sys.path, registers blueprints,
│                        sets up Jinja2 filters (markdown, pretty_json).
│                        Uses absolute template/static paths for import-safe app startup.
│
├── config.py            All env vars and paths. Single source of truth.
│                        Exports: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL,
│                                 RESULTS_DIR, UPLOAD_DIR, TOOL_LABELS, TOOL_ICONS,
│                                 get_llm(), simple_markdown()
│
├── core/                Shared domain objects — no Flask, no HTTP.
│   ├── job_parser_agent.py   JobPosting (Pydantic, 16 fields) + load_document()
│   │                         Copied/trimmed from src/job_analyzer/job_parser_agent.py
│   └── helpers.py            safe_call(), normalize_website(), infer_company_name()
│                             Copied/trimmed from src/job_analyzer/run.py
│
├── tools/               12 investigative tools — each is a standalone pure function.
│   ├── tools_config.py       REQUEST_TIMEOUT=20, REQUEST_HEADERS, DEFAULT_PHONE_REGION="IN"
│   ├── tool_scam_signals.py  detect_scam_signals(raw_text) → score 0-100, risk_level
│   ├── tool_email_verify.py  verify_email(email) → DNS MX, disposable domain check
│   ├── tool_domain_reputation.py  check_domain_reputation(domain_or_email) → WHOIS, age, risk
│   ├── tool_company_wikipedia.py  get_company_wikipedia(company) → extract + Wikipedia URL
│   ├── tool_company_web_search.py search_company_web(company) → 5 DuckDuckGo angles
│   ├── tool_company_news.py       search_company_news(company) → DDG News articles
│   ├── tool_website_verify.py     verify_website(url) → HTTP status, SSL, redirects
│   ├── tool_website_content.py    extract_website_content(url) → trafilatura text + metadata
│   ├── tool_social_profiles.py    check_social_profiles(company) → 7 platforms via DDG
│   ├── tool_job_boards.py         check_job_boards(title, company, location) → 8 boards
│   ├── tool_phone_check.py        check_phone_number(phone) → phonenumbers library
│   └── tool_company_registry.py  get_company_registry() → STUB (not yet implemented)
│
├── services/            Business logic — orchestrates tools and LLM calls.
│   ├── job_extractor.py     extract_from_text / extract_from_file / extract_from_linkedin
│   │                        → (JobPosting, raw_text)
│   │                        Uses: core/job_parser_agent + langchain_openai
│   ├── tool_runner.py       run_all_tools(job, raw_text, field_overrides=None)
│   │                        Uses: all 12 tools from tools/ + core/helpers
│   ├── analyzer.py          run_analysis(job_id, ...) — extraction + deep research + tools + report
│   │                        Includes: deep_research_missing_fields() + run_candidate_tool_checks()
│   │                        Writes progress to results/<job_id>.json after each step
│   └── linkedin.py          scrape_linkedin_job(url) → str | None
│                            requests + BeautifulSoup, gracefully returns None if blocked
│
├── routes/
│   ├── main.py    GET /        → index.html (3-tab input form)
│   │              POST /analyze → runs full pipeline → redirect to /results/<id>
│   │              GET /results/<id> → results.html
│   │              GET /results → history.html
│   └── api.py     POST /api/settings → save api_key/model/base_url to Flask session
│                  GET  /api/settings → return current effective settings
│                  GET  /api/result/<id> → raw JSON result
│
├── templates/
│   ├── base.html            Layout, navbar, settings modal
│   ├── index.html           3-tab form: Paste Text | Upload File | LinkedIn URL
│   ├── results.html         Results page with verdict banner, tool grid, report
│   ├── history.html         Table of all past analyses
│   └── components/
│       ├── job_info.html    JD + enriched values with source tags (JD vs Deep Research)
│       ├── deep_research.html   Deep research summary, missing fields, evidence sources
│       ├── candidate_tools.html Candidate-level tool outputs for all discovered emails/phones/websites
│       ├── tool_block.html  Per-tool card: status badge + inference + collapsible raw JSON
│       └── report.html      Final markdown report + web source chips + copy button
│
├── static/
│   ├── css/style.css        Enriched glassmorphism-like light theme + responsive data cards
│   └── js/main.js           Settings modal, file drop UX, flash auto-dismiss
│
├── results/                 Auto-created. UUID-keyed JSON files (one per analysis).
└── uploads/                 Auto-created. Temporary uploaded files.
```

---

## Maintenance Log

### 2026-04-15 — Issues Found and Fixed

1. **Results page crash when scam signals exist**
  - **Symptom:** `/results/<job_id>` returned HTTP 500 with `TypeError: 'int' object is not subscriptable`.
  - **Root cause:** `templates/components/tool_block.html` sliced `signals_found[:6]` assuming a list, but `tools/tool_scam_signals.py` returned `signals_found` as an integer count.
  - **Fix:**
    - Updated `tools/tool_scam_signals.py` to return:
     - `signals_found` as a list of matched rule names
     - `signals_count` as numeric count
    - Updated `templates/components/tool_block.html` to be backward-compatible with old saved results (`signals_found` int) by falling back to `matched_signals` keys.

2. **Template/static resolution fragile in non-standard import contexts**
  - **Symptom:** app factory could fail to resolve templates when imported via custom module loading/testing contexts.
  - **Root cause:** relative template/static paths depended on Flask import resolution context.
  - **Fix:** `app.py` now passes absolute paths for `template_folder` and `static_folder`.

3. **Proxy routing could be bypassed by stale session settings (causing 402)**
  - **Symptom:** LLM calls could fail with `Error code: 402` from direct OpenRouter credits even when proxy env vars were exported.
  - **Root cause:** runtime resolution previously preferred Flask session settings over env vars; also empty Settings values did not clear existing session overrides.
  - **Fix:**
    - `routes/main.py` now resolves LLM settings with explicit env var precedence (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`).
    - `routes/api.py` now allows clearing session overrides by sending empty values.
    - `GET /api/settings` now returns the effective settings source (`env` / `session` / `config`).

4. **Missing JD data now enriched via Deep Research + multi-candidate tool checks**
  - **Symptom:** many postings lacked email/phone/website in source text, causing weak investigations.
  - **Fix:**
    - Added external deep research stage in `services/analyzer.py` to search and recover likely missing contacts/websites.
    - Added candidate-level verification for all discovered emails/phones/websites (including multiple values) via tool calls.
    - Added explicit source labels in UI so users can distinguish JD-provided vs deep-research-found data.
    - Added enriched JSON fields: `job_posting_enriched`, `deep_research`, `candidate_tool_results`.

5. **UI redesign for readability and richer insight presentation**
  - Improved layout, typography, cards, chips, data hierarchy, and raw payload presentation.
  - Added dedicated sections for:
    - deep research evidence and applied overrides
    - multi-candidate tool results with per-item status
    - source transparency tags (`From JD` vs `Deep Research`).

### Change Ledger (Memory-Friendly)

| Date | Area | What changed | Key files |
|------|------|--------------|-----------|
| 2026-04-15 | Reliability | Fixed scam signal rendering mismatch (`signals_found` int vs list) and made template backward-compatible | `tools/tool_scam_signals.py`, `templates/components/tool_block.html` |
| 2026-04-15 | Runtime robustness | Forced absolute template/static paths in Flask factory to avoid import-context issues | `app.py` |
| 2026-04-15 | LLM routing | Added env-first precedence for proxy routing and session-clear behavior for stale overrides | `routes/main.py`, `routes/api.py`, `static/js/main.js` |
| 2026-04-15 | Data completeness | Added deep research enrichment stage for missing email/phone/website and applied safe overrides | `services/analyzer.py` |
| 2026-04-15 | Investigation depth | Added candidate-level tool execution for all discovered emails/phones/websites (JD + deep research) | `services/analyzer.py`, `services/tool_runner.py` |
| 2026-04-15 | UX/Presentation | Added dedicated deep research + candidate verification sections and redesigned visual system | `templates/results.html`, `templates/components/deep_research.html`, `templates/components/candidate_tools.html`, `templates/components/job_info.html`, `templates/base.html`, `static/css/style.css` |
| 2026-04-15 | Documentation | Updated architecture, enhanced pipeline, and expanded results schema for new fields | `WEBAPP.md` |

### Validation Snapshot

Last verified after the above changes:

1. Python compile check passed for `webapp/`.
2. Flask render smoke tests passed for `/` and `/results/<id>` with enriched payload.
3. Results page confirmed to render deep research and candidate-level sections.

### Recommended Next Steps

1. Implement `tools/tool_company_registry.py` (currently stubbed).
2. Add lightweight automated tests for:
  - `GET /` returns 200
  - `GET /results/<job_id>` renders successfully for both old and new `scam_signals` payload shapes
  - deep research + candidate sections render when `deep_research` and `candidate_tool_results` are present
3. Enforce `MAX_UPLOAD_BYTES` in upload flow (`routes/main.py`) to prevent oversized uploads.
4. Add guardrails for deep-research latency (timeouts/result caps already present, but add per-step timing metrics in saved JSON).

---

## Analysis Pipeline (Enhanced)

```
User Input (text / file / LinkedIn URL)
        ↓
Step 1 — EXTRACT
  services/job_extractor.py
  LLM reads raw text → fills JobPosting Pydantic model (16 fields)
  Uses: core/job_parser_agent.JobPosting + langchain_openai.ChatOpenAI.with_structured_output()
        ↓
Step 1b — DEEP RESEARCH ENRICHMENT
  services/analyzer.py → deep_research_missing_fields(job_dict, raw_text)
  Searches external web sources (DuckDuckGo) for missing email/phone/website
  Produces candidate lists + applied overrides for missing JD fields
        ↓
Step 2 — INVESTIGATE PRIMARY PATH (12 tools)
  services/tool_runner.py → run_all_tools(job, raw_text, field_overrides)
  Each tool returns: {ok: bool, data: {...}} or {ok: false, error: "..."}
  Tools skipped if required input (email/website/phone/company) not found
        ↓
Step 2b — CANDIDATE TOOL CHECKS (multi-value)
  services/analyzer.py → run_candidate_tool_checks(...)
  Runs tools for all discovered emails/phones/websites (JD + deep research)
        ↓
Step 3 — INFER (one LLM call per tool)
  services/analyzer.py → infer_tool_result(tool_name, result, job_dict)
  System prompt: "Summarize what this tool found and what it means for fraud risk"
  Returns 2-4 sentences per tool
        ↓
Step 4 — REPORT
  services/analyzer.py → web_search_fraud_signals() + generate_final_report()
  DuckDuckGo: 3 targeted queries for fraud complaints (12 results max)
  LLM: structured markdown report with verdict SAFE / SUSPICIOUS / LIKELY_FAKE
```

---

## Results JSON Schema

Each analysis is stored at `results/<uuid>.json`:

```json
{
  "job_id":             "uuid",
  "status":             "processing | complete | error",
  "created_at":         "ISO8601",
  "input_type":         "text | file | url",
  "raw_text":           "...",
  "job_posting":        { "title": "...", "company_name": "...", ... },
  "job_posting_enriched": { "contact_email": "...", "contact_phone": "...", ... },
  "deep_research": {
    "missing_from_jd": ["contact_email", "contact_phone", "company_website"],
    "candidates": { "emails": [...], "phones": [...], "websites": [...] },
    "applied_overrides": { "contact_email": "...", "contact_phone": "..." }
  },
  "tool_results":       { "scam_signals": { "ok": true, "data": {...} }, ... },
  "candidate_tool_results": {
    "emails":   [ { "value": "...", "source": "jd|deep_research", "results": {...} } ],
    "phones":   [ { "value": "...", "source": "jd|deep_research", "results": {...} } ],
    "websites": [ { "value": "...", "source": "jd|deep_research", "results": {...} } ]
  },
  "tool_inferences":    { "scam_signals": "2-4 sentence inference", ... },
  "web_search_results": [ { "query": "...", "title": "...", "url": "...", "snippet": "..." } ],
  "final_report":       "## Fraud Risk Assessment\n...",
  "verdict":            "SAFE | SUSPICIOUS | LIKELY_FAKE",
  "error":              null
}
```

---

## Tool Output Contracts

All tools return `{ok: bool, data: {...}}` on success or `{ok: false, error: "..."}` on failure.
`core/helpers.safe_call()` wraps every tool call so exceptions never crash the pipeline.

| Tool | Key output fields |
|------|-------------------|
| `scam_signals` | `scam_score` (0-100), `risk_level` (low/medium/high), `signals_found` (list), `signals_count` (int), `matched_signals` |
| `email_verify` | `is_deliverable`, `is_disposable`, `is_role_account`, `overall_status` |
| `domain_reputation` | `domain_age_days`, `risk_level`, `is_live`, `registrar` |
| `website_verify` | `is_live`, `ssl_valid`, `status_code`, `redirect_count` |
| `website_content` | `extracted_text`, `word_count`, `metadata` (title, description, sitename) |
| `company_wikipedia` | `title`, `extract`, `wikipedia_url` |
| `company_web_search` | `searches` (5 angles: general, reviews, scam, glassdoor, linkedin) |
| `company_news` | `articles` (date, title, url, source, snippet) |
| `social_profiles` | `platforms_found` (0-7), `profiles` per platform (found, links) |
| `job_boards` | `boards_found` (0-8), `verdict` (strong/moderate/not_found) |
| `phone_check` | `e164`, `is_valid`, `region_code`, `carrier`, `location` |
| `company_registry` | STUB — always `{ok: false}` |

---

## How to Add a New Tool

1. Create `webapp/tools/tool_<name>.py` with a single exported function
2. Follow the contract: return `{"ok": True, "data": {...}}` or `{"ok": False, "error": "..."}`
3. Add it to `services/tool_runner.py` → `run_all_tools()` with appropriate skip logic
4. Add display name to `TOOL_LABELS` in `config.py` and icon to `TOOL_ICONS`
5. Add to the `tool_order` list in `templates/results.html` to control display order

---

## LLM Configuration

All LLM calls go through `config.get_llm()` which reads three env vars:

| Env var | Default | Purpose |
|---------|---------|---------|
| `OPENAI_API_KEY` | `""` | API key (set to `$AIPIPE_TOKEN` for aipipe) |
| `OPENAI_BASE_URL` | `https://aipipe.org/openrouter/v1` | Proxy base URL |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Any OpenRouter model slug |

User can also override per-session via the Settings modal (stored in Flask session).

### Effective Precedence (Important)

Runtime uses this precedence:

1. Explicit environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`)
2. Session overrides (Settings modal)
3. `config.py` defaults

If `OPENAI_API_KEY` is exported, the app uses env-based routing first to avoid accidental bypass of the proxy.
Also, saving empty values in Settings now clears any stale session override.

---

## Dependencies (from project requirements.txt)

```
flask, markupsafe                    # Web framework
langchain-openai, langchain-core     # LLM calls
langchain-community                  # Document loaders (PDF, DOCX, etc.)
pydantic                             # JobPosting schema
requests, beautifulsoup4             # HTTP + LinkedIn scraping
ddgs                                 # DuckDuckGo search (all research tools)
trafilatura                          # Website content extraction
python-whois                         # Domain WHOIS lookup
email-validator                      # Email syntax + MX check
phonenumbers                         # Phone validation
```
