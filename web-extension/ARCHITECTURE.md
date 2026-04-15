# 🏗️ Extension Architecture (v2.0 — Chain Pipeline)

## Data Flow (v2.0 with Link Scraping)

```
LinkedIn Job Page (DOM)
        │
        ├── content.js scrapes job data
        │     {title, company, description, ...}
        │
        ├── content.js extracts DOM links
        │     [{url, text, source}, ...]
        │
        └── chrome.runtime.sendMessage({
              type: "ANALYZE_JOB",
              data: { jobData, domLinks }
            })
                │
                ▼
┌─── background.js (Pipeline Orchestrator) ────────────┐
│                                                       │
│  ToolRegistry (LangChain-inspired framework)          │
│  ┌────────────────────────────────────────────────┐   │
│  │ 1. DetectLinksTool                             │   │
│  │    • Regex URL extraction from text            │   │
│  │    • Categorize: job board, career, doc, form  │   │
│  │    • Filter LinkedIn-internal links            │   │
│  │    • Priority sort & limit to maxLinksToScrape │   │
│  ├────────────────────────────────────────────────┤   │
│  │ 2. LinkScraperTool [conditional]               │   │
│  │    • Parallel fetch (concurrency = 3)          │   │
│  │    • Retry with exponential backoff            │   │
│  │    • TextExtractor: HTML → clean text          │   │
│  │      ├── DOMParser for HTML parsing            │   │
│  │      ├── Noise removal (scripts/nav/ads)       │   │
│  │      ├── Main content area detection           │   │
│  │      └── JSON-LD schema extraction             │   │
│  ├────────────────────────────────────────────────┤   │
│  │ 3. ContentAggregatorTool                       │   │
│  │    • Merge job data + scraped content          │   │
│  │    • Source attribution                        │   │
│  │    • Smart truncation at maxContentLength      │   │
│  ├────────────────────────────────────────────────┤   │
│  │ 4. JobAnalyzerTool                             │   │
│  │    • Build enhanced prompt with red flags      │   │
│  │    • Cross-reference LinkedIn vs external      │   │
│  │    • Gemini API call (gemini-2.0-flash)        │   │
│  │    • Parse structured JSON response            │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  Progress updates sent to content script via          │
│  chrome.tabs.sendMessage(tabId, {...})                │
│                                                       │
└───────────────────────────────────────────────────────┘
                │
                ▼
        Analysis Result
        {verdict, confidence, riskScore,
         reasons, positiveSignals, tips,
         externalContentAnalysis, ...}
                │
                ▼
        content.js renders results overlay
        (color-coded verdict card with risk breakdown)
```

## File Responsibilities

| File | Role | Runs In |
|------|------|---------|
| `manifest.json` | Extension config, permissions, script registration | Chrome |
| `content.js` | Button injection, DOM scraping, link extraction, progress UI, overlay | LinkedIn page |
| `content.css` | All injected styles (button, overlay, progress bar) | LinkedIn page |
| `background.js` | Pipeline orchestrator — imports and chains all tools | Service worker |
| `lib/langchain-core.js` | Framework: BaseTool, ToolResult, ToolRegistry, Chain | Service worker |
| `lib/pipeline.js` | PipelineConfig, PipelineBuilder, ContentAggregatorTool | Service worker |
| `tools/link-detector.js` | DetectLinksTool — URL discovery and categorization | Service worker |
| `tools/text-extractor.js` | TextExtractor — HTML to clean text | Service worker |
| `tools/link-scraper.js` | LinkScraperTool — fetch + extract external content | Service worker |
| `tools/job-analyzer-tool.js` | JobAnalyzerTool — Gemini API analysis | Service worker |
| `popup.html/css/js` | API key settings UI | Extension popup |

## Analysis Modes

| Mode | Invoked With | Link Scraping | Max Links | AI Depth |
|------|---|:---:|---:|---|
| Standard | `ANALYZE_JOB` | ✅ | 5 | Thorough |
| Quick | `ANALYZE_JOB_QUICK` | ❌ | 0 | Brief |
| Deep | `ANALYZE_JOB_DEEP` | ✅ | 10 | Exhaustive |

## Swapping to Your Agentic Flow

Modify only `tools/job-analyzer-tool.js`:

```javascript
// Replace the _callGemini method:
async _callGemini(apiKey, prompt, depthConfig) { ... }

// With your multi-agent call:
async _callAgenticPipeline(apiKey, prompt, depthConfig) {
    // Your multi-agent orchestration (CrewAI, AutoGen, etc.)
    // Must return: { verdict, confidence, riskScore, reasons,
    //               positiveSignals, summary, tips,
    //               externalContentAnalysis }
}
```

The rest of the pipeline stays unchanged — `background.js`, `content.js`, and all other tools work with any AI backend.

## See Also
- [CHAIN_DOCS.md](CHAIN_DOCS.md) — Detailed framework & tool reference
- [SETUP.md](SETUP.md) — Quick installation guide
- [README.md](README.md) — Full project overview
