# 🔗 LangChain-Inspired Pipeline Architecture

## Overview

This extension uses a **LangChain-inspired chain/tool pattern** to orchestrate multi-step job analysis. Instead of a simple "scrape → API call" flow, we have a configurable pipeline that:

1. **Detects links** in job descriptions
2. **Scrapes external content** from those links
3. **Aggregates everything** into an enriched document
4. **Analyzes with AI** using Gemini for a comprehensive verdict

## Pipeline Data Flow

```
LinkedIn Job Page (DOM)
        │
        ├── content.js scrapes job data
        │   {title, company, description, ...}
        │
        ├── content.js extracts DOM links
        │   [{url, text, source}, ...]
        │
        └── chrome.runtime.sendMessage()
                │
                ▼
    ┌─── background.js (Pipeline Orchestrator) ───┐
    │                                               │
    │   Step 1: DetectLinksTool                    │
    │   ├── Regex URL extraction from text         │
    │   ├── URL categorization (job board, doc,    │
    │   │   career page, form, social)             │
    │   ├── Skip LinkedIn-internal links           │
    │   └── Priority sort + limit                  │
    │           │                                   │
    │   Step 2: LinkScraperTool [conditional]       │
    │   ├── Parallel fetch with concurrency=3      │
    │   ├── Retry + exponential backoff            │
    │   ├── Content type validation                │
    │   ├── TextExtractor: HTML → clean text       │
    │   │   ├── DOMParser parsing                  │
    │   │   ├── Noise removal (scripts, nav, ads)  │
    │   │   ├── Main content detection             │
    │   │   └── JSON-LD schema extraction          │
    │   └── Per-link metadata tracking             │
    │           │                                   │
    │   Step 3: ContentAggregatorTool              │
    │   ├── Combine LinkedIn data + link content   │
    │   ├── Source attribution per section          │
    │   └── Smart truncation at maxContentLength   │
    │           │                                   │
    │   Step 4: JobAnalyzerTool                    │
    │   ├── Enhanced prompt with red flags taxonomy │
    │   ├── Cross-reference LinkedIn vs external   │
    │   ├── Gemini API call (0.3 temperature)      │
    │   └── Risk score breakdown (6 categories)    │
    │           │                                   │
    └───────────┘                                   │
                │                                   │
                ▼                                   │
        Analysis Result                             │
        {verdict, confidence, riskScore,            │
         reasons, positiveSignals, tips,            │
         externalContentAnalysis, ...}              │
                │
                ▼
        content.js renders overlay
        (color-coded verdict card)
```

## Framework Classes

### Core (`lib/langchain-core.js`)

| Class | Purpose |
|-------|---------|
| `BaseTool` | Abstract base — validation, caching, metrics, error handling |
| `ToolResult` | Standardized success/failure wrapper for tool outputs |
| `ToolRegistry` | Central registry for tool discovery and management |
| `Chain` | Sequential pipeline — feeds each tool's output to the next |
| `ChainStep` | Wraps a tool with input/output transforms and conditions |
| `ChainResult` | Final output with step-by-step execution tracking |

### Pipeline (`lib/pipeline.js`)

| Class | Purpose |
|-------|---------|
| `PipelineConfig` | Configuration: analysis depth, link limits, timeouts |
| `PipelineBuilder` | Fluent builder for constructing chains |
| `ContentAggregatorTool` | Merges job data + scraped content |

### Tools (`tools/`)

| File | Tool | Purpose |
|------|------|---------|
| `link-detector.js` | `DetectLinksTool` | Find and categorize URLs in text |
| `text-extractor.js` | `TextExtractor` | Parse HTML → clean text |
| `link-scraper.js` | `LinkScraperTool` | Fetch URLs + extract content |
| `job-analyzer-tool.js` | `JobAnalyzerTool` | Gemini AI analysis |

## Analysis Modes

| Mode | Link Scraping | Depth | Max Content |
|------|:---:|---|---:|
| **Quick** | ❌ | Brief analysis | 5,000 chars |
| **Standard** | ✅ (5 links) | Thorough | 15,000 chars |
| **Deep** | ✅ (10 links) | Exhaustive | 25,000 chars |

## Red Flags Taxonomy

The analyzer checks **6 categories** with **30 specific indicators**:

1. **Description Quality** — Vague content, buzzwords, copy-paste
2. **Compensation** — Unrealistic salary, upfront fees
3. **Company Info** — Missing/inconsistent company data
4. **Application Process** — Personal info requests, no interviews
5. **Job Logistics** — Wrong location, mismatched requirements
6. **External Content** — Broken links, inconsistent data across sources

## Adding New Tools

```javascript
import { BaseTool, ToolResult } from "../lib/langchain-core.js";

class MyNewTool extends BaseTool {
    constructor() {
        super({
            name: "my_tool",
            description: "Does something new",
            version: "1.0.0",
        });
    }

    async _execute(input, context) {
        // Your logic here
        const result = doSomething(input);
        return ToolResult.ok(result);
    }
}

// Register in background.js:
registry.register(new MyNewTool(), "category");
```

## Swapping to Your Agentic Flow

To replace Gemini with your multi-agent pipeline, modify **only** `tools/job-analyzer-tool.js`:

```javascript
// Replace _callGemini() with your agentic call:
async _callAgenticPipeline(apiKey, prompt, depthConfig) {
    // Your multi-agent orchestration here
    // Must return: { verdict, confidence, riskScore, reasons, ... }
}
```

The pipeline architecture makes this swap trivial — the rest of the chain doesn't change.
