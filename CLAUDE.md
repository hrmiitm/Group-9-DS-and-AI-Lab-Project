# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Webetention is a two-track fraud job posting detection system:
1. **Python backend**: Fine-tuned RoBERTa-base model + 12-tool free evidence pipeline
2. **JavaScript frontend**: Chrome MV3 extension with LangChain-inspired pipeline + Gemini AI

The model weights are published on HuggingFace as `aditya963/fraud-job-classifier`.

## Commands

### Python ML Pipeline

```bash
# Install Python dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Required for spaCy NER

# Train the RoBERTa model
python src/train.py --data_path data/fake_job_postings.csv --output_dir models/roberta-focal-best

# Evaluate / single inference
python src/eval.py --model_dir models/roberta-focal-best --data_path data/fake_job_postings.csv
python src/eval.py --model_dir models/roberta-focal-best --infer

# Run the free evidence pipeline on a job document (from project root)
python src/job_analyzer/run.py path/to/job_posting.pdf
# Outputs: src/job_analyzer/outputs/<stem>_summary_<timestamp>.txt
#           src/job_analyzer/outputs/<stem>_tool_evidence_<timestamp>.txt

# Test individual evidence tools
python src/job_analyzer/tools/tool_email_verify.py
python src/job_analyzer/tools/tool_domain_reputation.py
```

The evidence pipeline requires `OPENAI_API_KEY` by default (for `job_parser_agent.py`). Switch providers via env vars — see [LangChain ReAct Agent](#langchain-react-agent) section.

### Chrome Extension

No build step — the extension uses vanilla ES modules loaded directly by the browser.

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `webextension/` directory
4. Open the extension popup and paste a Gemini API key

## Architecture

### Python ML Stack

- **Data** (`src/utils/data.py`): Loads the Kaggle Fake Job Postings CSV (17,880 samples, 4.8% fraud). Features are built by concatenating metadata + free-text fields with `[SEP]` tokens. Stratified 70/15/15 split.
- **Model** (`src/train.py`): `roberta-base` fine-tuned with Focal Loss (gamma=1.69, fraud weight=2.83), AdamW + cosine LR scheduler. Hyperparameters Optuna-tuned over 25 trials. Best threshold: 0.87 (calibrated on test set).
- **Evaluation** (`src/eval.py`): Generates F1/Precision/Recall/ROC-AUC/MCC metrics, threshold sweep, confusion matrix, and ROC/PR curve plots.

### Free Evidence Pipeline (`src/job_analyzer/`)

12-tool chain that requires **no API keys** (except `job_parser_agent.py` for extraction):

| Stage | Tools | Method |
|-------|-------|--------|
| Extract | `job_parser_agent.py` | LangChain ReAct + LLM |
| Verify email & domain | `tool_email_verify`, `tool_domain_reputation` | DNS MX + WHOIS |
| Company research | `tool_company_wikipedia`, `tool_company_web_search`, `tool_company_news` | Wikipedia API + DuckDuckGo |
| Website | `_tool_website_verify`, `_tool_website_content` | HTTP + trafilatura |
| Social/boards | `tool_social_profiles`, `tool_job_boards` | DuckDuckGo |
| Signal scoring | `tool_phone_check`, `_tool_scam_signals` | phonenumbers + keyword scoring |
| Registry | `_tool_company_registry` | DuckDuckGo company lookup |

Tools prefixed with `_` are internal (not standalone-runnable). `run.py` orchestrates all tools via `build_context()` → `run_all_tools()` → `write_reports()` and writes two timestamped `.txt` files to `src/job_analyzer/outputs/`.

### Chrome Extension (`webextension/`)

**Data flow (parallel architecture):**

```
LinkedIn DOM → content.js scrapes job data
                        ↓
              background.js dispatches two branches in parallel:
                ├── RoBERTaTool → HuggingFace Inference API (aditya963/fraud-job-classifier)
                └── DetectLinksTool → LinkScraperTool → ContentAggregatorTool
                        ↓ (both branches complete)
              JobAnalyzerTool → Gemini (gemini-2.5-flash)
              receives: RoBERTa fraud score + scraped link evidence
                        ↓
              content.js ← verdict overlay injected
```

**Custom framework** (`webextension/lib/`):
- `langchain-core.js`: `BaseTool` (timing, caching, validation), `ToolRegistry`, `Chain`, `ToolResult`
- `pipeline.js`: `PipelineConfig` (Standard/Quick/Deep modes), `PipelineBuilder`, `ContentAggregatorTool`

**Tool pipeline** (`webextension/tools/`):
1. `RoBERTaTool` — calls HuggingFace Inference API; builds `[SEP]`-delimited input matching Python training format; threshold=0.87
2. `DetectLinksTool` — regex URL extraction + categorization (job board / career / social / form)
3. `LinkScraperTool` — parallel fetch (concurrency=3) with retry/backoff
4. `TextExtractor` — HTML → clean text, JSON-LD extraction
5. `JobAnalyzerTool` — builds prompt with 30-point red flag taxonomy + RoBERTa score, calls Gemini, parses JSON verdict

**Analysis modes** (configured in `background.js`):
- **Quick**: No link scraping, brief prompt
- **Standard**: Scrape 5 links, thorough prompt
- **Deep**: Scrape 10 links, exhaustive prompt

**Verdicts**: `SAFE` / `SUSPICIOUS` / `LIKELY_FAKE` with confidence score, summary, key findings, and actionable tips rendered as a slide-in overlay panel.

**Settings**: Gemini API key stored via `chrome.storage.local`, managed in `popup.html/js`.

### LangChain ReAct Agent

`src/job_analyzer/job_parser_agent.py` and `AgenticWork/job_parser_agent.py` extract 18 structured features from multi-format job documents (PDF, DOCX, HTML, MD, TXT) using LangChain ReAct. The version in `src/job_analyzer/` is imported as a library by `run.py`.

Supports multiple LLM providers via env vars — no code changes needed:

```bash
# Default
export OPENAI_API_KEY="sk-..."                                     # OpenAI gpt-4.1-nano

# Alternatives
AGENT_PROVIDER=anthropic AGENT_MODEL=claude-3-5-sonnet-20241022
AGENT_PROVIDER=google    AGENT_MODEL=gemini-2.0-flash
AGENT_PROVIDER=ollama    AGENT_MODEL=llama3.1

python AgenticWork/job_parser_agent.py path/to/job_description.pdf
```

## Key Configuration

- `src/job_analyzer/config.py`: `REQUEST_TIMEOUT=20`, `DEFAULT_PHONE_REGION="IN"`, User-Agent header
- `webextension/manifest.json`: MV3, host permissions include `https://*/*` for link scraping and `https://generativelanguage.googleapis.com/*` for Gemini
- `webextension/tools/roberta-tool.js`: `FRAUD_THRESHOLD=0.87`, `HF_MODEL_URL` points to `aditya963/fraud-job-classifier`
- Model hyperparameters (LR=2.59e-05, batch=16, threshold=0.87) are in `src/train.py` — sourced from Optuna trial 18
