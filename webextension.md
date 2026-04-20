# FraudGuard Chrome Extension — Complete Viva Guide

> A comprehensive reference covering architecture, workflow, concepts, and Q&A prep for the FraudGuard LinkedIn Job Fraud Detection Chrome Extension.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Chrome Extension Fundamentals](#3-chrome-extension-fundamentals)
4. [Manifest V3 Deep Dive](#4-manifest-v3-deep-dive)
5. [Extension Components & Their Roles](#5-extension-components--their-roles)
6. [Complete Data Flow — Step by Step](#6-complete-data-flow--step-by-step)
7. [LinkedIn DOM Scraping](#7-linkedin-dom-scraping)
8. [The ML Pipeline (RoBERTa)](#8-the-ml-pipeline-roberta)
9. [The LLM Analysis Pipeline (Gemini)](#9-the-llm-analysis-pipeline-gemini)
10. [Link Scraping Pipeline](#10-link-scraping-pipeline)
11. [Analysis Modes](#11-analysis-modes)
12. [The LangChain-Inspired Framework](#12-the-langchain-inspired-framework)
13. [Tool Ecosystem (Extension Tools)](#13-tool-ecosystem-extension-tools)
14. [Message Passing Architecture](#14-message-passing-architecture)
15. [Storage & API Key Management](#15-storage--api-key-management)
16. [UI Injection & Overlay System](#16-ui-injection--overlay-system)
17. [Error Handling & Fault Tolerance](#17-error-handling--fault-tolerance)
18. [Security & Privacy Design](#18-security--privacy-design)
19. [Full System Context (Web-App + Backend)](#19-full-system-context-web-app--backend)
20. [Key Algorithms Explained](#20-key-algorithms-explained)
21. [Common Viva Questions & Answers](#21-common-viva-questions--answers)

---

## 1. Project Overview

**FraudGuard** is an AI-powered Chrome extension that detects fraudulent job listings on LinkedIn in real time. It protects job seekers from:

- **Advance-fee scams** (asking you to pay for training/equipment)
- **Phishing attacks** (stealing personal information)
- **Identity theft** (requesting government ID/bank details upfront)
- **Fake companies** (non-existent employers)

### The Core Problem

Online job fraud costs victims billions annually. Fraudulent listings are designed to look legitimate — they mimic real companies, use professional language, and target desperate job seekers. Manual detection is unreliable and slow.

### The Solution: Dual-Layer AI Detection

FraudGuard uses **two independent AI systems** working in parallel:

| Layer | Technology | What It Detects |
|-------|-----------|-----------------|
| Layer 1 | Fine-tuned RoBERTa (125M params) | Statistical fraud patterns from training data |
| Layer 2 | Gemini 2.5-Flash LLM | Contextual red flags from job + external links |

Both layers feed into a **final verdict**: `SAFE`, `SUSPICIOUS`, or `LIKELY_FAKE`.

---

## 2. System Architecture

FraudGuard is a multi-platform system. The Chrome extension is one component of a larger ecosystem:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRAUDGUARD SYSTEM                            │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  CHROME EXTENSION    │    │       WEB APP (Flask)            │  │
│  │                      │    │                                  │  │
│  │  content.js          │    │  Job Upload → Parser Agent →     │  │
│  │  (LinkedIn Scraper)  │    │  12 Investigation Tools →        │  │
│  │         ↓            │    │  LLM Reports                     │  │
│  │  background.js       │    └──────────────────────────────────┘  │
│  │  (Orchestrator)      │                                           │
│  │         ↓            │    ┌──────────────────────────────────┐  │
│  │  ┌──────┴──────┐     │    │     REACT FRONTEND (Vite)        │  │
│  │  │  RoBERTa  │Gemini│    │     Tool Grid + Report View      │  │
│  │  │  Tool     │Tool  │    └──────────────────────────────────┘  │
│  │  └───────────┴──────┘     │                                   │  │
│  └──────────────────────┘    ┌──────────────────────────────────┐  │
│           │                  │    FASTAPI BACKEND               │  │
│           │                  │    Tool Registry + LLM APIs      │  │
│           ↓                  └──────────────────────────────────┘  │
│  ┌────────────────────┐      ┌──────────────────────────────────┐  │
│  │  HuggingFace       │      │    MODEL API (FastAPI)           │  │
│  │  Inference API     │      │    RoBERTa Prediction Service    │  │
│  │  (RoBERTa)         │      │    Deployed on HF Spaces         │  │
│  └────────────────────┘      └──────────────────────────────────┘  │
│           │                                                         │
│  ┌────────────────────┐                                             │
│  │  Gemini 2.5-Flash  │                                             │
│  │  (Google AI API)   │                                             │
│  └────────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Why a Chrome Extension Specifically?

- **In-context analysis**: Works where the user already is (LinkedIn)
- **Zero friction**: No copy-pasting, no separate tab
- **Real-time protection**: Warns before the user applies
- **Access to rendered DOM**: Can read dynamically-loaded LinkedIn content

---

## 3. Chrome Extension Fundamentals

Before understanding FraudGuard, you need to understand how Chrome extensions work.

### What Is a Chrome Extension?

A Chrome extension is a **web application** (HTML/CSS/JS) that runs inside the Chrome browser with special privileges. It can:
- Inject scripts into web pages
- Access browser APIs (storage, tabs, network)
- Run persistent background code
- Communicate with external APIs

### Extension Architecture (Manifest V3)

Modern Chrome extensions (MV3) consist of several distinct contexts:

```
┌─────────────────────────────────────────────────────┐
│                   CHROME BROWSER                    │
│                                                     │
│  ┌─────────────────────┐  ┌────────────────────┐   │
│  │   WEB PAGE          │  │  EXTENSION POPUP   │   │
│  │   (linkedin.com)    │  │  (popup.html)      │   │
│  │                     │  │                    │   │
│  │  content.js runs    │  │  Rendered when     │   │
│  │  IN this page's     │  │  user clicks icon  │   │
│  │  context            │  └────────────────────┘   │
│  └─────────────────────┘                           │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │         SERVICE WORKER (background.js)      │   │
│  │                                             │   │
│  │  Persistent background process              │   │
│  │  Handles API calls, business logic          │   │
│  │  Cannot access the DOM                      │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Isolation Principle

These three contexts **cannot directly share variables**. They communicate via **message passing** — sending JSON messages through the Chrome runtime API. This is a fundamental constraint that shapes the entire extension architecture.

---

## 4. Manifest V3 Deep Dive

The `manifest.json` is the extension's configuration file — it tells Chrome everything about the extension.

```json
{
  "manifest_version": 3,
  "name": "LinkedIn Job Predictor",
  "version": "2.0.0",
  "description": "AI-powered fraud detection for LinkedIn job listings",

  "permissions": [
    "activeTab",
    "storage"
  ],

  "host_permissions": [
    "https://generativelanguage.googleapis.com/*",
    "https://*/*",
    "http://*/*"
  ],

  "background": {
    "service_worker": "background.js",
    "type": "module"
  },

  "content_scripts": [
    {
      "matches": ["*://*.linkedin.com/*"],
      "js": ["content.js"],
      "css": ["content.css"],
      "run_at": "document_idle"
    }
  ],

  "action": {
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon16.png",
      "32": "icons/icon32.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  }
}
```

### Permission Breakdown

| Permission | Why It's Needed |
|-----------|----------------|
| `activeTab` | Access the URL and content of the currently active tab |
| `storage` | Store API keys (Gemini, HuggingFace) in `chrome.storage.local` |
| `host_permissions: generativelanguage.googleapis.com` | Allow fetch requests to Gemini API |
| `host_permissions: https://*/*` | Allow fetching external links from job descriptions |
| `content_scripts matches: linkedin.com` | Inject content.js only on LinkedIn pages |

### MV3 vs MV2: Key Differences

| Feature | MV2 (Old) | MV3 (Current) |
|---------|-----------|---------------|
| Background | Persistent page | Service worker (can sleep) |
| Remote code | Allowed | Blocked (CSP enforcement) |
| `fetch` in background | Full access | Full access |
| `XMLHttpRequest` in background | Allowed | Blocked (use `fetch`) |

### Why `"run_at": "document_idle"`?

LinkedIn is a **Single Page Application (SPA)** — it loads content dynamically via JavaScript after the initial HTML. `document_idle` means the content script waits for the page to finish initial loading before injecting. However, LinkedIn still loads job details asynchronously, which is why `content.js` has additional retry logic.

---

## 5. Extension Components & Their Roles

### `manifest.json`
The configuration file. Declares permissions, entry points, and metadata.

### `content.js` — The DOM Agent
- **Where it runs**: Inside the LinkedIn page's JavaScript context
- **What it does**:
  - Detects when a LinkedIn job page is open
  - Scrapes job data from the DOM (title, company, description, location, salary, links)
  - Injects UI elements (the "Analyze Job" button, results overlay)
  - Sends scraped data to `background.js` via messages
  - Receives results from `background.js` and renders them

### `background.js` — The Orchestrator
- **Where it runs**: As a Chrome service worker (separate background context)
- **What it does**:
  - Receives job data from `content.js`
  - Manages the entire analysis pipeline
  - Calls RoBERTa API (HuggingFace)
  - Calls Gemini API
  - Orchestrates parallel execution of RoBERTa + link scraping
  - Sends results back to `content.js`
- **Why it's here and not in content.js**: API keys should not be exposed in the page context. Background service workers are isolated and more secure for handling credentials.

### `popup.html` + `popup.js` — The Settings Panel
- Shown when user clicks the extension icon in the toolbar
- Used to configure API keys (Gemini, HuggingFace)
- Saves keys to `chrome.storage.local`

### `content.css` — Injected Styles
- CSS for the "Analyze Job" button and results overlay
- Injected into the LinkedIn page alongside `content.js`

### `lib/langchain-core.js` — The Tool Framework
- Custom LangChain-inspired framework for composable tools and chains
- Provides `BaseTool`, `Chain`, `ChainStep`, `ToolRegistry`, `ToolResult`
- Enables modular, reusable analysis components

### `tools/` Directory
Individual analysis tools (RoBERTa, Gemini, link detector, link scraper, etc.)

### `lib/pipeline.js` — Pipeline Builder
- Fluent API for constructing analysis pipelines
- Configures which tools run in which order

---

## 6. Complete Data Flow — Step by Step

This is the most important section for a viva. Understand this flow thoroughly.

```
USER OPENS LINKEDIN JOB PAGE
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: CONTENT SCRIPT INJECTION                                │
│                                                                 │
│  Chrome auto-injects content.js into linkedin.com pages        │
│  content.js starts watching for a job listing to appear        │
│  Injects "Analyze Job" button into the page DOM                │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DOM SCRAPING (content.js)                               │
│                                                                 │
│  User clicks "Analyze Job" button                               │
│  content.js calls scrapeJobData():                              │
│    - Title: from document.title or <h1>                         │
│    - Company: from /company/ links in page                      │
│    - Description: from #job-details selector (with fallbacks)  │
│    - Location, salary, employment type, workplace type          │
│    - All links found in the description area                    │
│  Waits up to 5×800ms if content not yet loaded                  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: MESSAGE TO BACKGROUND (content.js → background.js)      │
│                                                                 │
│  chrome.runtime.sendMessage({                                   │
│    action: "ANALYZE_JOB",   // or QUICK / DEEP                  │
│    jobData: { title, company, description, links, ... }         │
│  })                                                             │
│                                                                 │
│  content.js shows a progress spinner while waiting              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: PARALLEL PIPELINE (background.js)                       │
│                                                                 │
│  background.js receives the message                             │
│  Launches TWO parallel operations via Promise.all():            │
│                                                                 │
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │ PIPELINE A          │    │ PIPELINE B                   │   │
│  │                     │    │                              │   │
│  │ RoBERTaTool         │    │ DetectLinksTool              │   │
│  │                     │    │      ↓                       │   │
│  │ 1. Build input text │    │ LinkScraperTool              │   │
│  │    (standardize     │    │      ↓                       │   │
│  │    job fields)      │    │ ContentAggregatorTool        │   │
│  │                     │    │                              │   │
│  │ 2. POST to HF API   │    │ Fetches & scrapes up to      │   │
│  │                     │    │ 5-10 external links          │   │
│  │ 3. Get fraud prob.  │    │                              │   │
│  └─────────────────────┘    └──────────────────────────────┘   │
│                ↓                          ↓                     │
│          fraud_prob: 0.92          scraped_content: [...]       │
│          verdict: FRAUDULENT       link_analysis: [...]         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: GEMINI FINAL ANALYSIS (background.js)                   │
│                                                                 │
│  JobAnalyzerTool (Gemini 2.5-Flash) receives:                   │
│    - Original job metadata (title, company, description...)     │
│    - RoBERTa result (fraud_probability: 0.92, HIGH confidence)  │
│    - Scraped link content + metadata                            │
│    - Red flags taxonomy (50+ known fraud patterns)              │
│                                                                 │
│  Sends structured JSON prompt to Gemini API                     │
│  Temperature: 0.2-0.4 (low = deterministic, factual)            │
│                                                                 │
│  Gemini returns JSON with:                                      │
│    { verdict, confidence, riskScore, reasons,                   │
│      positiveSignals, summary, tips,                            │
│      externalContentAnalysis }                                  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: SEND RESULT BACK (background.js → content.js)           │
│                                                                 │
│  chrome.tabs.sendMessage(tabId, {                               │
│    action: "ANALYSIS_COMPLETE",                                 │
│    result: { verdict, confidence, riskScore, reasons, ... }     │
│  })                                                             │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: RENDER RESULTS (content.js)                             │
│                                                                 │
│  content.js receives the result                                 │
│  Injects verdict overlay into the LinkedIn page:                │
│    - Verdict badge: ✅ SAFE / ⚠️ SUSPICIOUS / ❌ LIKELY_FAKE    │
│    - Risk score breakdown (5 categories × 1-10 scale)           │
│    - List of external links found + their analysis              │
│    - Red flags list                                             │
│    - Positive signals list                                      │
│    - Actionable tips for the job seeker                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. LinkedIn DOM Scraping

### Why DOM Scraping Is Hard on LinkedIn

LinkedIn is a **Single Page Application (SPA)** built with React. This means:
- Content is loaded asynchronously via JavaScript
- The DOM changes without full page reloads
- Selectors can break when LinkedIn updates their UI
- Some elements are hidden (aria-hidden, screen-reader only)

### Scraping Strategy: Multi-Selector Fallbacks

`content.js` uses a **defensive multi-fallback approach** for each field:

```
TITLE EXTRACTION PRIORITY:
  1. document.title parsing: "Software Engineer at Google | LinkedIn"
  2. <h1> elements on the page (most prominent heading)
  3. <title> tag directly
  4. Raw text nuclear fallback

COMPANY EXTRACTION PRIORITY:
  1. Anchors with href containing /company/ (most reliable)
  2. Meta tags for organization
  3. Raw text extraction

DESCRIPTION EXTRACTION PRIORITY:
  1. #job-details selector (LinkedIn's primary ID)
  2. TreeWalker looking for "About the job" heading
  3. Deep DOM search for largest text block with job keywords
  4. Full page raw text fallback
```

### Link Extraction

`extractLinksFromDOM()` finds all `<a href>` elements in:
- Job description section
- Company section
- Insights area
- Application section

Links are de-duplicated by URL. Each link is tagged with its source area. The list is passed to `background.js` for scraping.

### Waiting for Async Content

```javascript
async function waitForJobContent() {
  let attempts = 0;
  while (attempts < 5) {
    const data = scrapeJobData();
    if (data.title && data.description && data.description.length > 100) {
      return data;  // Valid content found
    }
    await sleep(800);  // Wait 800ms and retry
    attempts++;
  }
  return rawTextFallback();  // Last resort
}
```

This retry loop handles LinkedIn's async content loading.

### The Nuclear Fallback

If all DOM selectors fail, `rawTextFallback()` uses a `TreeWalker` to traverse the entire page DOM, collects all visible text nodes, and returns the concatenated result. This is less structured but ensures the pipeline always has *something* to analyze.

---

## 8. The ML Pipeline (RoBERTa)

### What Is RoBERTa?

RoBERTa (**R**obustly **O**ptimized **BERT** Pre-training **A**pproach) is a transformer model by Facebook AI. It's a variant of BERT, trained on a much larger corpus with improved training methodology.

**Architecture**:
- 125 million parameters
- 12 transformer layers
- 12 attention heads per layer
- 768 hidden dimensions
- Context window: 512 tokens

### Our Fine-Tuned Model

**Model ID**: `aditya963/fraud-job-classifier` (HuggingFace)

**Training Dataset**: EMSCAD (Employment Scam Aegean Corpus Dataset)
- 17,880 job postings
- Only ~4.84% are fraudulent (class imbalance problem)
- Features: title, description, requirements, company_profile, location, salary, etc.

**Loss Function**: **Focal Loss** (not standard cross-entropy)
- Focal Loss addresses class imbalance
- It penalizes easy correct predictions and focuses the model on hard examples
- Formula: `FL(p) = -α(1-p)^γ × log(p)` where γ (gamma) down-weights easy examples

**Classification Threshold**: **0.87**
- Standard binary classifiers use 0.5 (50%)
- Our threshold is 0.87 — the model must be 87% confident before calling a job fraudulent
- This **reduces false positives** (legitimate jobs marked as fake)
- The threshold was calibrated on the validation set to balance precision/recall

### Model Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ROC-AUC | 0.993 | 0.97 | ✓ Exceeded |
| Precision (fraud class) | 0.957 | 0.93 | ✓ Exceeded |
| Recall (fraud class) | 0.862 | 0.89 | Slight miss |
| F1 Score (fraud class) | 0.907 | 0.91 | Slight miss |
| MCC | 0.892 | - | Strong |

**ROC-AUC of 0.993** means the model is near-perfect at distinguishing legitimate vs. fraudulent jobs.

### Input Standardization

The model was trained on structured text. We must replicate that structure at inference time using `buildInputText()` in `roberta-tool.js`:

```
Location: New York [SEP] Salary: 80000-100000 [SEP]
Employment Type: Full-time [SEP] Workplace Type: Remote [SEP]
Industry: Technology [SEP] Job Title [SEP] Company Name [SEP]
This is a full-time position requiring... [SEP]
Company Profile: We are a Fortune 500...
```

**Key Design Decisions**:
- Structured metadata fields come **first** (location, salary, employment type)
- Free-text fields come **last** (title, description, company profile)
- `[SEP]` is RoBERTa's sentence separator token — signals field boundaries
- Max 3000 characters (tokenizes to ~512 tokens, RoBERTa's max)
- Fields with missing data are simply omitted (no placeholder)

### API Access Route (Extension)

The extension calls HuggingFace Inference API:

```
POST https://api-inference.huggingface.co/models/aditya963/fraud-job-classifier
Headers:
  Authorization: Bearer <HF_API_TOKEN>
  Content-Type: application/json
Body:
  { "inputs": "<standardized job text>" }

Response:
  [{ "label": "FRAUDULENT", "score": 0.9234 },
   { "label": "LEGITIMATE", "score": 0.0766 }]
```

The extension extracts the score for the `FRAUDULENT` label and compares it to the 0.87 threshold.

### Confidence Bands

```
fraud_probability >= 0.95  → HIGH confidence FRAUDULENT
fraud_probability 0.87-0.95 → MEDIUM confidence FRAUDULENT (above threshold)
fraud_probability 0.70-0.87 → SUSPICIOUS (near threshold)
fraud_probability < 0.70    → LOW risk, likely legitimate
```

---

## 9. The LLM Analysis Pipeline (Gemini)

### Why a Second AI Layer?

RoBERTa gives a statistical fraud probability based on patterns in training data. It has limitations:
- Trained on 2019-2023 data — may miss new fraud patterns
- Cannot access external information (linked websites, company reputation)
- No reasoning — just a probability score

Gemini 2.5-Flash adds **contextual reasoning**:
- Reads and interprets scraped content from external links
- Explains *why* a job looks fraudulent
- Cross-references job details with what's found on company websites
- Produces human-readable analysis

### The Prompt Engineering

`background.js` builds a comprehensive JSON prompt:

```javascript
{
  system: "You are a fraud detection expert...",
  jobData: {
    title: "Software Engineer",
    company: "TechCorp LLC",
    location: "Remote",
    salary: "$5,000/week guaranteed",
    description: "...",
    scrapedLinks: [
      { url: "...", title: "...", content: "...", wordCount: 340 }
    ]
  },
  mlResults: {
    fraudProbability: 0.92,
    confidence: "HIGH",
    verdict: "FRAUDULENT",
    threshold: 0.87
  },
  redFlagsTaxonomy: [
    "advance fee requests",
    "guaranteed high salary",
    "request for personal documents before interview",
    "free email domain (gmail) for company contact",
    "vague job requirements",
    ...50+ patterns
  ],
  analysisDepth: "standard"
}
```

### Gemini Response Schema

Gemini is prompted to return **structured JSON**:

```json
{
  "verdict": "LIKELY_FAKE",
  "confidence": 91,
  "riskScore": {
    "descriptionQuality": 8,
    "compensationFlags": 9,
    "companyLegitimacy": 7,
    "applicationProcess": 6,
    "externalContent": 7,
    "overall": 7.4
  },
  "reasons": [
    "Salary ($5,000/week) is 3-4x market rate for role",
    "Company website domain registered 2 weeks ago",
    "Description requests SSN and bank details before interview"
  ],
  "positiveSignals": [
    "Job title is specific and realistic",
    "Company has some web presence"
  ],
  "summary": "This posting shows multiple high-confidence fraud indicators...",
  "tips": "Do not provide any personal documents. Research the company...",
  "externalContentAnalysis": {
    "consistent": false,
    "discrepancies": ["Company website claims to be in NYC but job says London"],
    "additionalInfo": "..."
  }
}
```

### Fallback JSON Parsing

If Gemini returns malformed JSON, `job-analyzer-tool.js` uses **regex extraction** to parse the key fields:
- Extracts verdict with `/"verdict"\s*:\s*"([^"]+)"/`
- Extracts confidence with `/"confidence"\s*:\s*(\d+)/`
- Extracts reasons array with regex
- This ensures the pipeline never completely fails even if LLM output is slightly off-format

### Temperature Setting

- `temperature: 0.2` for quick analysis (more deterministic)
- `temperature: 0.3` for standard analysis
- `temperature: 0.4` for deep analysis (slightly more creative reasoning)

Lower temperature = more consistent, factual responses. This is appropriate for fraud detection where we want reproducible results.

---

## 10. Link Scraping Pipeline

### Why Scrape External Links?

Fraudulent job postings often include links to:
- Fake company websites (cloned from legitimate companies)
- Phishing pages disguised as application portals
- Unrelated websites (indicating copy-paste job descriptions)

By fetching and analyzing these links, Gemini can detect inconsistencies.

### The Three-Tool Pipeline

```
DetectLinksTool
      │
      │  Extracts all URLs from job description DOM
      │  Categories: company site, apply link, external ref
      │  De-duplicates
      ↓
LinkScraperTool
      │
      │  Fetches up to N links (5 standard, 10 deep)
      │  10 second timeout per link
      │  Handles CORS, redirects
      │  Extracts: title, meta description, word count
      │  Detects job-related keywords in content
      ↓
ContentAggregatorTool
      │
      │  Combines link metadata + scraped content
      │  Annotates content with source URL
      │  Builds enrichedDescription with all evidence
      ↓
     Final combined context → Gemini
```

### CORS Handling

Web pages cannot freely fetch content from other domains (Same-Origin Policy). Since `background.js` (service worker) runs outside the page context, it **can** make cross-origin requests with `fetch()` — no CORS restrictions apply to extension service workers.

---

## 11. Analysis Modes

The extension offers three analysis modes that trade speed for depth:

| Feature | Quick | Standard | Deep |
|---------|-------|----------|------|
| Steps | 3 | 5 | 6 |
| RoBERTa | ✓ | ✓ | ✓ |
| Link detection | ✓ | ✓ | ✓ |
| Link scraping | ✗ | Up to 5 | Up to 10 |
| Gemini analysis | Brief | Thorough | Extremely detailed |
| Gemini max tokens | 2048 | 4096 | 8192 |
| Estimated time | ~5s | ~15s | ~30s |
| Best for | Quick check | General use | Deep investigation |

### Mode Selection Flow

```
User clicks "Analyze Job" →
    Shows mode selection dialog →
    User picks Quick / Standard / Deep →
    content.js sends:
      { action: "ANALYZE_JOB_QUICK" }  or
      { action: "ANALYZE_JOB" }         or
      { action: "ANALYZE_JOB_DEEP" }
```

---

## 12. The LangChain-Inspired Framework

### Why Build a Custom Framework?

LangChain (the Python library) is not available in browser JavaScript environments. The team implemented a **LangChain-inspired framework** in `lib/langchain-core.js` to get:
- Composable, reusable tools
- Standardized inputs/outputs
- Chain execution with shared context
- Built-in caching and error handling
- Execution statistics

### Core Classes

#### `BaseTool` (Abstract)

Every tool in the extension extends `BaseTool`:

```javascript
class BaseTool {
  name: string           // Unique identifier
  description: string    // Human-readable purpose
  version: string        // Semantic version
  requiredInputFields: string[]   // Validated before execute()
  outputFields: string[]          // What the tool returns

  async execute(input):
    // 1. Validate required input fields
    // 2. Check cache
    // 3. Call _execute() (subclass implementation)
    // 4. Record timing stats
    // 5. Return ToolResult

  async _execute(input):
    // Override this in subclasses
}
```

#### `ToolResult`

Standardized return wrapper:

```javascript
{
  success: true/false,
  data: { ...actual results },
  error: "Error message if failed",
  metadata: {
    timestamp: "2025-...",
    toolName: "roberta_analyzer",
    executionTimeMs: 423,
    cached: false
  }
}
```

#### `ToolRegistry`

Central registry for all tools:

```javascript
const registry = new ToolRegistry();
registry.register(new RoBERTaTool(), "ml");
registry.register(new DetectLinksTool(), "scraping");
registry.register(new LinkScraperTool(), "scraping");
registry.register(new JobAnalyzerTool(), "analysis");

// Later:
const tool = registry.get("roberta_analyzer");
await tool.execute(jobData);
```

#### `Chain` + `ChainStep`

```javascript
const chain = new Chain({
  name: "fraud_analysis_pipeline",
  registry: registry,
  callbacks: {
    onStepStart: (step) => sendProgressUpdate(step.name),
    onStepComplete: (step, result) => ...,
    onError: (step, error) => ...
  }
});

chain.addStep(new ChainStep({
  toolName: "detect_links",
  optional: true  // Won't fail the chain if this step fails
}));

chain.addStep(new ChainStep({
  toolName: "link_scraper",
  condition: (context) => context.config.enableLinkScraping
}));

const result = await chain.run(initialInput);
```

Each step receives the **accumulated context** from all previous steps — output of step N becomes part of input to step N+1.

### Benefits of This Architecture

- **Modularity**: New tools are plug-and-play
- **Testability**: Each tool can be tested independently
- **Observability**: Built-in timing, stats, execution history
- **Fault tolerance**: `optional: true` steps don't break the chain
- **Conditional execution**: Steps can be skipped based on context

---

## 13. Tool Ecosystem (Extension Tools)

### `RoBERTaTool` (`tools/roberta-tool.js`)

- **Input**: Job data object (title, company, description, location, salary, etc.)
- **Process**: Builds standardized input text → calls HuggingFace Inference API
- **Output**: `{ fraudProbability, verdict, confidence, threshold, inputText }`
- **Failure behavior**: Returns graceful failure, pipeline continues without ML result

### `JobAnalyzerTool` (`tools/job-analyzer-tool.js`)

- **Input**: Job metadata + RoBERTa result + scraped link content
- **Process**: Builds comprehensive prompt → calls Gemini 2.5-Flash API
- **Output**: Full verdict object with risk scores, reasons, tips
- **Failure behavior**: Error propagated (this is the final output, cannot skip)

### `DetectLinksTool` (`tools/link-detector.js`)

- **Input**: `links` array from scraped DOM, job description text
- **Process**: URL normalization, deduplication, categorization
- **Output**: Categorized link list (company, apply, external)
- **Failure behavior**: Returns empty list, pipeline continues

### `LinkScraperTool` (`tools/link-scraper.js`)

- **Input**: Array of links + max count + timeout
- **Process**: `fetch()` each URL, parse response, extract text content
- **Output**: Array of `{ url, title, description, wordCount, content, jobRelated }`
- **Failure behavior**: Failed links are logged and skipped; others proceed

### `TextExtractorTool` (`tools/text-extractor.js`)

- **Input**: Raw HTML string
- **Process**: Removes tags, normalizes whitespace, extracts readable text
- **Output**: Clean text string
- **Used by**: `LinkScraperTool` to process fetched HTML

---

## 14. Message Passing Architecture

This is a critical concept for understanding Chrome extensions.

### The Three-Context Problem

```
content.js                background.js              popup.js
(runs in page)     ←→     (service worker)    ←→    (popup page)
Cannot share             Cannot share               Cannot share
variables directly       variables directly          variables directly
```

### Communication Methods

**Content Script → Background**:
```javascript
// In content.js
chrome.runtime.sendMessage(
  { action: "ANALYZE_JOB", jobData: {...} },
  (response) => {
    // Handle immediate acknowledgment
  }
);
```

**Background → Content Script**:
```javascript
// In background.js (needs tab ID)
chrome.tabs.sendMessage(
  tabId,
  { action: "ANALYSIS_COMPLETE", result: {...} }
);
```

**Background listening for messages**:
```javascript
// In background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "ANALYZE_JOB") {
    // Process asynchronously
    analyzeJob(message.jobData, sender.tab.id);
    sendResponse({ status: "processing" });
    return true;  // IMPORTANT: keeps message channel open for async response
  }
});
```

**Important**: `return true` in the message listener keeps the channel open for asynchronous responses. Without it, the connection closes immediately.

### Progress Updates

During analysis, `background.js` sends **progress messages** back to `content.js`:

```javascript
chrome.tabs.sendMessage(tabId, {
  action: "ANALYSIS_PROGRESS",
  step: "Running RoBERTa model...",
  stepNumber: 2,
  totalSteps: 5
});
```

`content.js` uses these to update the progress spinner/status text shown to the user.

---

## 15. Storage & API Key Management

### Why `chrome.storage.local`?

API keys for Gemini and HuggingFace must be:
1. **Persistent** across browser sessions
2. **Not hardcoded** in the extension (security risk)
3. **Accessible** from both popup and background contexts
4. **Isolated** from web pages (not accessible via `window.localStorage`)

`chrome.storage.local` satisfies all these requirements. It's Chrome's sandboxed storage, only accessible by the extension itself.

### Storing Keys (in popup.js)

```javascript
chrome.storage.local.set({
  geminiApiKey: "AIza...",
  huggingfaceToken: "hf_..."
});
```

### Reading Keys (in background.js)

```javascript
const { geminiApiKey, huggingfaceToken } = await chrome.storage.get(
  ["geminiApiKey", "huggingfaceToken"]
);
```

### Security Considerations

- Keys are stored locally on the user's machine (not synced to cloud)
- Keys are never logged or sent to any server other than their respective APIs
- The extension cannot access keys of other extensions
- If the user clears browser data, keys are lost and must be re-entered

---

## 16. UI Injection & Overlay System

### Injecting the "Analyze Job" Button

`content.js` creates and injects a floating button into the LinkedIn DOM:

```javascript
const button = document.createElement("div");
button.id = "fraudguard-analyze-btn";
button.innerHTML = `<button>🔍 Analyze Job</button>`;
document.body.appendChild(button);
```

The button is styled via `content.css` (also injected) to appear as a floating panel over the LinkedIn UI.

### The Results Overlay

After analysis, a results panel is injected:

```javascript
const overlay = document.createElement("div");
overlay.id = "fraudguard-results";
overlay.innerHTML = `
  <div class="verdict LIKELY_FAKE">❌ LIKELY FAKE</div>
  <div class="confidence">91% confidence</div>
  <div class="risk-scores">
    <div>Description Quality: 8/10</div>
    <div>Compensation Flags: 9/10</div>
    ...
  </div>
  <div class="reasons">
    <h3>Red Flags</h3>
    <ul>...</ul>
  </div>
  <div class="tips">...</div>
`;
document.body.appendChild(overlay);
```

### Dynamic Styling

The verdict badge color is set dynamically:
- `SAFE` → green (`#22c55e`)
- `SUSPICIOUS` → orange (`#f59e0b`)
- `LIKELY_FAKE` → red (`#ef4444`)

### Handling LinkedIn SPA Navigation

When users navigate between LinkedIn jobs without a full page reload (SPA behavior), `content.js` needs to re-run. This is handled by observing URL changes:

```javascript
let lastUrl = location.href;
const observer = new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    // Remove old overlay, re-inject button
    resetUI();
    setupJobAnalysis();
  }
});
observer.observe(document.body, { subtree: true, childList: true });
```

---

## 17. Error Handling & Fault Tolerance

### Design Philosophy: Fail Gracefully

The pipeline is designed to produce **some result even when components fail**:

```
RoBERTa API fails?
  → Continue with just Gemini analysis
  → Gemini prompt notes "ML model unavailable"
  → Still produces verdict

Link scraping fails?
  → Continue with just job text analysis
  → Gemini prompt notes "external links unavailable"
  → Still produces verdict

Gemini API fails?
  → Return error to user
  → This is the final output layer, cannot skip
  → Show friendly error message with API key troubleshooting tips
```

### Specific Error Scenarios

| Scenario | Handling |
|----------|----------|
| HF API rate limit (429) | Retry once, then skip ML layer |
| HF API model loading (503) | Wait 20s, retry, then skip |
| Link fetch timeout | Skip that link, continue with others |
| Gemini malformed JSON | Regex fallback parser extracts key fields |
| LinkedIn DOM changed | Fall back to raw text nuclear fallback |
| No API keys set | Show popup with configuration instructions |
| Network offline | Show connectivity error with instructions |

### Error Propagation

```javascript
async function analyzeWithRoberta(jobData) {
  try {
    const result = await roberta.execute(jobData);
    return result;
  } catch (error) {
    console.warn("RoBERTa failed:", error.message);
    return null;  // Return null, not throw — pipeline continues
  }
}

// In main pipeline:
const [robertaResult, scrapingResult] = await Promise.all([
  analyzeWithRoberta(jobData),    // Can return null
  scrapeLinks(jobData.links)      // Can return []
]);

// Gemini still runs even if robertaResult is null
await analyzeWithGemini(jobData, robertaResult, scrapingResult);
```

---

## 18. Security & Privacy Design

### Data Minimization

- Only job posting text is extracted (not cookies, browsing history, or other page content)
- The extension only activates on `linkedin.com` URLs
- Scraped external link content is only used for fraud analysis, not stored

### API Key Security

- Keys stored in `chrome.storage.local` (browser-isolated)
- Keys never transmitted except to their intended APIs
- Keys not logged in console (redacted in debug output)
- No hardcoded credentials in the extension code

### No Data Collection

- The extension does NOT send job data to any team servers
- All processing goes directly to Gemini (Google) and HuggingFace APIs
- No telemetry, analytics, or usage tracking

### External API Privacy Notes

- Job text IS sent to Google (Gemini API) — their data policies apply
- Job text IS sent to HuggingFace Inference API — their data policies apply
- Scraped link content IS sent to Gemini — same caveat
- Users should be aware that job details are processed by these third-party APIs

### Content Security Policy

Manifest V3 enforces strict CSP:
- No `eval()` or `new Function()` (dynamic code execution blocked)
- No remote script loading (all code is local to the extension)
- External API calls only to domains in `host_permissions`

---

## 19. Full System Context (Web-App + Backend)

The extension is one component. The full system includes:

### Flask Web App (`web-app/`)

A standalone web application for more thorough fraud analysis:

**Flow**:
1. User uploads job description (PDF, DOCX, TXT, HTML) or pastes text
2. **Job Parser Agent** (LangChain + GPT-4o-mini) extracts structured fields into a `JobPosting` Pydantic schema
3. **12 Investigation Tools** run sequentially:
   - Scam signal scanner (keyword scoring)
   - Email verification
   - Domain reputation (WHOIS + VirusTotal)
   - Website accessibility check
   - Website content analysis
   - Company Wikipedia search
   - Company web search (DuckDuckGo)
   - Company news search
   - Social media profile finder
   - Job board cross-reference
   - Phone number validation
   - Company WHOIS registry check
4. Each tool result analyzed by LLM (per-tool inference)
5. Final LLM report aggregates all findings → verdict + detailed markdown report

**When to use web app vs extension**:
- Extension: Quick analysis while browsing LinkedIn
- Web app: Deep investigation of any job posting from any source

### FastAPI Backend (`backend-api/`)

REST API exposing all web-app tools individually:
- `GET /api/v1/tools` — list all tools
- `POST /api/v1/run/{tool_name}` — run a single tool
- `POST /api/v1/run-batch` — run multiple tools in parallel
- `POST /api/v1/llm/*` — LLM parsing/analysis endpoints

### React Frontend (`frontend-app/`)

React + Vite UI for the backend API:
- Multi-tab input (text, file, URL)
- Tool status grid (12 cards showing ✓/✗/running)
- Extracted job fields display
- Final report rendering (markdown → HTML)

### Model API (`model-api/`)

Standalone FastAPI service for RoBERTa:
- Deployed to HuggingFace Spaces (free GPU hosting)
- Serves the fine-tuned model via REST API
- Used when HuggingFace Inference API is unavailable
- `POST /predict` — single prediction
- `POST /predict/batch` — up to 16 predictions at once

---

## 20. Key Algorithms Explained

### Scam Signal Scoring (Web App)

A rule-based weighted scoring system:

```python
RULES = {
  "asks_for_money": {
    "weight": 30,
    "keywords": ["advance payment", "training fee", "equipment deposit", ...]
  },
  "requests_bank_details": {
    "weight": 35,
    "keywords": ["bank account", "routing number", "wire transfer", ...]
  },
  "high_pressure": {
    "weight": 15,
    "keywords": ["act now", "limited spots", "urgent", ...]
  },
  "unrealistic_promises": {
    "weight": 20,
    "keywords": ["guaranteed income", "no experience needed", "earn $5000/day", ...]
  },
  ...7 total categories
}

# Scoring
total_score = sum(rule.weight for rule in RULES if any keyword in job_text)

# Risk levels
if total_score >= 60: "HIGH"
elif total_score >= 25: "MEDIUM"
else: "LOW"
```

### Email Scoring

```python
def score_email(email, company_name):
  score = 0
  domain = email.split("@")[1].lower()
  company_token = company_name.lower().replace(" ", "")

  if company_token in domain:
    score += 4   # Company-specific domain is good signal
  if domain in FREE_EMAIL_DOMAINS:  # gmail, yahoo, hotmail...
    score -= 2   # Free email for company contact is suspicious

  return score
```

The top 6 emails by score are passed to the email verification tool.

### Focal Loss (Why It Was Used for Training)

Standard cross-entropy treats all samples equally. With 4.84% fraud rate:
- 95.16% of training samples are legitimate
- The model would learn to always predict "legitimate" and achieve 95% accuracy
- But it would miss all fraud cases (recall = 0)

Focal Loss addresses this:
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

Where:
  γ (gamma) = focusing parameter, typically 2
  α_t = class weight balancing factor
  p_t = model's confidence in correct class

(1 - p_t)^γ reduces loss for easy examples (high confidence)
This forces model to focus on hard-to-classify examples
```

### RoBERTa Tokenization

Text is broken into **WordPiece tokens**:
- "fraudulent" → ["fraud", "##ulent"]
- "##" prefix means "continuation of previous word"
- Special tokens: `[CLS]` (start), `[SEP]` (separator), `[PAD]` (padding)
- Max 512 tokens — texts longer than this are truncated
- The `[CLS]` token's final hidden state is used for classification

---

## 21. Common Viva Questions & Answers

### Architecture & Design

**Q: Why use a Chrome extension instead of a standalone website?**

A: A Chrome extension offers **in-context analysis** — it works directly within LinkedIn where users are already browsing. There's no friction of copy-pasting job descriptions to another site. The extension also has access to the fully rendered DOM (including dynamically loaded content) and can inject UI elements directly into the job listing page for immediate visual feedback.

---

**Q: Why does the extension have a background script separate from the content script?**

A: The three extension contexts (content script, background, popup) are **isolated JavaScript environments** that cannot share variables. Content scripts run inside the web page's context, while background service workers run separately. This separation is needed because:
1. API keys should not be accessible to the web page's JavaScript context (security)
2. The background service worker can make unrestricted cross-origin `fetch()` requests
3. Complex pipeline logic and state management belong in the background, not the page

---

**Q: What is Manifest V3 and how does it differ from V2?**

A: Manifest V3 is the current Chrome extension API standard. Key differences from MV2:
- Background pages (persistent) replaced with **service workers** (can be terminated by Chrome when idle)
- Remote code execution (`eval`, `new Function`, external scripts) is blocked
- CORS restrictions enforced more strictly in some contexts
- Declarative net request API replaces webRequest for blocking (we don't use blocking)
- Service workers require explicit `return true` in `onMessage` for async responses

---

### ML & AI

**Q: Why RoBERTa specifically? Why not BERT or GPT?**

A: RoBERTa is optimized for **sequence classification** tasks like fraud detection:
- Trained longer with more data than original BERT
- Uses dynamic masking (re-masks each epoch, not static)
- Removes BERT's Next Sentence Prediction (NSP) task (proven unhelpful)
- Better at domain-specific fine-tuning
- GPT models are generative (left-to-right) and less suited for classification
- RoBERTa-base (125M params) is small enough to serve via free API

---

**Q: Why is the fraud threshold 0.87 instead of 0.5?**

A: The dataset has severe class imbalance (~5% fraud). At 0.5, the model would generate many false positives — legitimate jobs incorrectly flagged as fraud. This would destroy user trust. 0.87 was chosen by analyzing the precision-recall curve on the validation set to maximize precision (we want to be confident before flagging something as fraud) while maintaining acceptable recall.

---

**Q: What is Focal Loss and why was it used?**

A: Focal Loss is a modified cross-entropy loss function designed for **imbalanced datasets**. The key modification is the `(1-p_t)^γ` factor, which reduces the loss contribution from easy, correctly-classified examples. This forces the model to focus on the hard-to-classify examples (the rare fraud cases). Without it, the model would achieve 95%+ accuracy by always predicting "legitimate" while having 0% fraud recall.

---

**Q: What are the limitations of the RoBERTa model?**

A:
1. **Training data cutoff**: Trained on 2019-2023 data; new fraud patterns (post-2023) may not be detected
2. **Fixed context window**: Only processes 512 tokens; very long job descriptions get truncated
3. **No external knowledge**: Cannot check if a company actually exists or verify claimed salaries
4. **Language**: Trained on English text; non-English postings perform poorly
5. **Static threshold**: 0.87 may not be optimal for all job categories (tech vs retail jobs have different language patterns)

---

**Q: Why use Gemini as the second layer instead of calling the same RoBERTa again?**

A: They serve fundamentally different purposes:
- RoBERTa gives a **statistical fraud probability** based on text patterns alone
- Gemini provides **reasoning and context** — it can read the scraped external links, cross-reference the company website's content against the job posting, identify specific red flags by name, and generate human-readable explanations
- Gemini also stays current (it's a cloud API that gets regular updates), compensating for RoBERTa's fixed training data
- Together they provide defense-in-depth: pattern matching + contextual reasoning

---

**Q: How does the model handle class imbalance in the dataset?**

A:
1. **Focal Loss** during training — down-weights easy examples, focuses on hard ones
2. **Calibrated threshold** (0.87) — requires high confidence before predicting fraud
3. **ROC-AUC as primary metric** — measures discriminative ability independent of threshold
4. **Weighted evaluation** — metrics computed separately for fraud class to avoid being fooled by accuracy on imbalanced data

---

### Data Flow & Scraping

**Q: How does the extension handle LinkedIn's dynamic content loading?**

A: LinkedIn is a React SPA that loads content asynchronously. The extension handles this with:
1. `run_at: "document_idle"` — waits for initial DOM to load
2. `waitForJobContent()` — retries up to 5 times with 800ms delays until valid content appears
3. `MutationObserver` — detects URL changes when navigating between jobs without page reload
4. Multi-selector fallback system — if primary selectors fail, tries progressively more aggressive extraction methods

---

**Q: What happens if LinkedIn changes their DOM structure and selectors break?**

A: The extension has multiple fallback layers:
1. Primary selectors (most specific, most reliable)
2. Generic heading/content selectors
3. `getDescriptionFromAboutSection()` — TreeWalker searching for "About the job" heading
4. `getDescriptionNuclearFallback()` — finds the largest text block containing job-related keywords
5. `rawTextFallback()` — extracts all visible text from the entire page

Even in the worst case, the pipeline gets *some* text to analyze.

---

**Q: Why does the extension scrape external links from job descriptions?**

A: Fraudulent job postings often include links to:
- Cloned company websites (look like real companies but are fake)
- Phishing application portals
- Irrelevant websites (indicating copy-paste descriptions)

By fetching and analyzing these links, Gemini can detect inconsistencies between what the job says and what external websites actually show. For example: "Job claims company is in NYC, but company website says London with no NYC office."

---

**Q: Why is link scraping done in the background script, not content script?**

A: Content scripts are subject to **CORS (Cross-Origin Resource Sharing)** restrictions — they cannot freely fetch content from domains other than the current page's origin. Background service workers make requests from the extension's origin, not from `linkedin.com`, so they bypass CORS restrictions entirely and can fetch any URL.

---

### Security & Privacy

**Q: Is it safe to store API keys in the extension? Could a malicious website steal them?**

A: `chrome.storage.local` is **extension-isolated storage**. Web pages cannot access it via JavaScript — there is no `window.chromeStorage` or similar API available to web pages. Only the extension's own scripts (background, popup, content) can access it via `chrome.storage` API. A malicious website cannot steal keys stored this way.

---

**Q: Does the extension send user browsing data to any servers?**

A: No. The extension only processes job posting content visible on the current LinkedIn page. The data flow is:
- Job text → HuggingFace API (only for ML prediction)
- Job text + link content → Google Gemini API (only for fraud analysis)
- No data is sent to the project team's servers
- No telemetry or analytics collected

---

**Q: What are the privacy implications of using third-party APIs?**

A: When the extension sends job data to Gemini and HuggingFace:
- These companies process the text according to their privacy policies
- HuggingFace Inference API does not retain request data for training by default
- Gemini processes data per Google's cloud AI terms of service
- Users should be informed that job descriptions are processed by these external services

---

### General Concepts

**Q: What is the difference between the Chrome extension and the Flask web app?**

A:
| | Chrome Extension | Flask Web App |
|---|---|---|
| Input source | LinkedIn DOM (real-time scraping) | Any uploaded file or pasted text |
| Analysis pipeline | RoBERTa + Gemini (2-layer) | 12 tools + LLM reports |
| Tools used | 5 extension tools | 12 web tools |
| Best for | Quick in-context checks | Deep investigation of any job posting |
| User experience | Overlay on LinkedIn page | Separate web interface |
| API access | Browser extension APIs | Server-side Python |

---

**Q: What is the LangChain-inspired framework and why was it built?**

A: LangChain is a popular Python framework for building AI pipelines with composable tools. Since LangChain is not available in browser JavaScript, we built a lightweight equivalent in `lib/langchain-core.js`. It provides:
- `BaseTool`: Standard interface for all tools (input validation, error handling, timing, caching)
- `ToolResult`: Standardized output wrapper (`{ success, data, error, metadata }`)
- `Chain`: Sequential tool orchestration with shared context and callbacks
- `ToolRegistry`: Central tool management

This makes the codebase modular — new tools can be added by extending `BaseTool` and registering with `ToolRegistry` without changing the pipeline logic.

---

**Q: What is a service worker and why does Manifest V3 use them?**

A: A service worker is a JavaScript file that runs in the background, separate from any web page. Chrome MV3 uses service workers instead of persistent background pages (MV2) because:
1. **Resource efficiency**: Service workers can be terminated when idle (unlike persistent pages that consume memory/CPU constantly)
2. **Security**: No persistent document context reduces attack surface
3. **Lifecycle alignment**: Aligns with Progressive Web App (PWA) service worker model
4. **Consistency**: Same APIs as other modern web platform features

Tradeoff: Service workers cannot maintain state in memory between events — any state must be persisted to `chrome.storage` or handled within a single event.

---

**Q: Explain how `Promise.all()` is used in the extension pipeline.**

A: `Promise.all()` executes multiple async operations **in parallel** and waits for all to complete:

```javascript
const [robertaResult, scrapingResult] = await Promise.all([
  runRoberta(jobData),    // Runs simultaneously
  scrapeLinks(jobData)    // Runs simultaneously
]);
// Code here only runs after BOTH complete
```

This is crucial for performance: RoBERTa (HuggingFace API call, ~1-3s) and link scraping (fetching multiple URLs, ~5-10s) run at the same time. Without parallelism, these would be sequential and take 6-13+ seconds. With `Promise.all`, total time is approximately `max(roberta_time, scraping_time)`.

---

**Q: What is the EMSCAD dataset?**

A: EMSCAD (Employment Scam Aegean Corpus Dataset) is a public dataset of 17,880 real job postings collected from online job boards, with ~866 labeled as fraudulent (4.84%). It was created by researchers at the University of the Aegean for studying online employment fraud. Each record includes: title, location, department, salary range, company profile, job description, requirements, benefits, employment type, required experience/education, telecommuting flag, company logo flag, and screening questions flag.

---

**Q: If someone asks "what is the most critical component of the extension?"**

A: The most critical component is `background.js` — it is the orchestrator that:
1. Receives job data from `content.js`
2. Manages the entire dual-layer AI pipeline
3. Coordinates parallel execution (RoBERTa + link scraping)
4. Calls the Gemini API with comprehensive context
5. Returns the final verdict to `content.js`

Without `background.js`, the extension has no analysis capability. The ML model, the LLM, and the tool framework are all coordinated here. If `content.js` is the "eyes" (scraping the page) and the results overlay is the "voice" (showing results), then `background.js` is the "brain."

---

## Summary: Extension at a Glance

```
User opens LinkedIn job page
  ↓
content.js detects the page, injects "Analyze Job" button
  ↓
User clicks button, chooses analysis mode (Quick/Standard/Deep)
  ↓
content.js scrapes job data from DOM (title, company, description, links)
  ↓
Sends to background.js via chrome.runtime.sendMessage()
  ↓
background.js runs PARALLEL pipelines:
  ├─ RoBERTaTool → HuggingFace API → fraud probability (0.0-1.0)
  └─ DetectLinks → LinkScraper → ContentAggregator → scraped link content
  ↓
background.js runs JobAnalyzerTool (Gemini 2.5-Flash):
  Input: job metadata + RoBERTa result (0.92 probability) + scraped links
  Output: { verdict: "LIKELY_FAKE", confidence: 91, riskScores, reasons, tips }
  ↓
background.js sends result to content.js via chrome.tabs.sendMessage()
  ↓
content.js injects results overlay on LinkedIn page:
  ❌ LIKELY FAKE (91% confidence)
  Risk Scores: Description Quality 8/10, Compensation 9/10...
  Red Flags: [list of specific red flags]
  Tips: [actionable advice]
```

**The Two-Layer Philosophy**:
- **RoBERTa** answers: "Does this text statistically resemble fraud based on 17,880 training examples?"
- **Gemini** answers: "Given what we know about this job and what's on linked websites, what specifically makes this suspicious and what should the user do?"

Together they provide **high confidence fraud detection** that neither could achieve alone.

---

*Document prepared for Group 9 DS & AI Lab Project viva — FraudGuard Chrome Extension*
