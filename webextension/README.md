# 🛡️ LinkedIn Job Predictor — Chrome Extension

> A Chrome extension that analyzes LinkedIn job listings using Google Gemini AI to predict whether a job is **legit**, **suspicious**, or **potentially fake** — so you don't waste time applying to scam postings.

---

## 📁 Project Structure

```
Webetention/
├── manifest.json      ← The brain — tells Chrome what this extension is
├── background.js      ← Talks to Gemini API (runs behind the scenes)
├── content.js         ← Injected INTO LinkedIn pages (button + scraper + overlay)
├── content.css        ← Styles for everything injected on LinkedIn
├── popup.html         ← The small window when you click the extension icon
├── popup.css          ← Styles for the popup
├── popup.js           ← Saves/loads your API key
└── icons/
    ├── icon16.png     ← Tiny icon (tabs, favicon)
    ├── icon48.png     ← Medium icon (extensions page)
    └── icon128.png    ← Large icon (Chrome Web Store)
```

---

## 🧠 Complete Code Flow (How It Actually Works)

Here's exactly what happens under the hood, step by step:

### Step 1: Extension Loads → `manifest.json`

When you install the extension, Chrome reads `manifest.json` first. This file is like a config/resume — it tells Chrome:

```json
"content_scripts": [{
  "matches": ["*://*.linkedin.com/*"],   // ← ONLY activate on LinkedIn
  "js": ["content.js"],                   // ← Inject this JavaScript
  "css": ["content.css"],                 // ← Inject these styles
  "run_at": "document_idle"               // ← Wait till page is fully loaded
}]
```

**What this means:** The moment you visit *any* LinkedIn page, Chrome automatically injects `content.js` and `content.css` into that page. You don't click anything — it just happens.

It also registers:
- **`background.js`** as a service worker (runs in the background, not on any page)
- **`popup.html`** as the popup UI when you click the extension icon in the toolbar

### Step 2: Button Appears on LinkedIn → `content.js` (Part 1)

As soon as `content.js` loads on a LinkedIn page, this runs:

```javascript
// Prevent double injection (LinkedIn is a SPA, pages don't fully reload)
if (window.__linkedinJobPredictor) return;
window.__linkedinJobPredictor = true;

// Create and inject the floating button
function createAnalyzeButton() {
  const btn = document.createElement("button");
  btn.id = "ljp-analyze-btn";
  btn.innerHTML = `<span>🔍</span><span>Analyze Job</span>`;
  btn.addEventListener("click", handleAnalyzeClick);
  document.body.appendChild(btn);  // ← Added directly to LinkedIn's page!
}
```

The button is positioned at the **bottom-right** of the screen with `position: fixed` in CSS, with a glassmorphism (frosted glass) effect and a subtle pulse animation.

**Why the MutationObserver?** LinkedIn is a Single Page Application (SPA) — when you navigate between pages, it doesn't do a full page reload. The DOM changes dynamically. So we watch for DOM changes and re-inject the button if it disappears:

```javascript
const observer = new MutationObserver(() => {
  if (!document.getElementById("ljp-analyze-btn")) {
    createAnalyzeButton();  // ← Button got removed? Put it back!
  }
});
observer.observe(document.body, { childList: true, subtree: true });
```

### Step 3: User Clicks "Analyze Job" → `content.js` (Part 2 — Scraping)

When you click the button, `handleAnalyzeClick()` fires. First, it **scrapes data from the page**:

```javascript
function scrapeJobData() {
  const data = {};

  // Each field tries multiple CSS selectors (LinkedIn changes their DOM often)
  data.title =
    getTextContent(".job-details-jobs-unified-top-card__job-title h1") ||
    getTextContent(".jobs-unified-top-card__job-title") ||
    getTextContent("h1") || "";

  data.company =
    getTextContent(".job-details-jobs-unified-top-card__company-name") ||
    getTextContent(".jobs-unified-top-card__company-name") || "";

  data.description =
    getTextContent(".jobs-description-content__text") ||
    getTextContent("#job-details") || "";

  // ... also scrapes: location, salary, seniority, employment type, etc.
  return data;
}
```

**Why multiple selectors?** LinkedIn A/B tests their UI constantly, so class names differ. We try the most common ones in order and grab the first match.

The `getTextContent()` helper is simple — it does `document.querySelector(selector)?.textContent.trim()`.

### Step 4: Data Sent to Background → Chrome Message Passing

The content script **can't call external APIs directly** (Content Security Policy). So it sends the scraped data to the background worker:

```javascript
// content.js sends:
const response = await chrome.runtime.sendMessage({
  type: "ANALYZE_JOB",
  data: jobData,    // ← The scraped job data object
});

// background.js listens:
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "ANALYZE_JOB") {
    handleAnalyzeJob(request.data)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;  // ← IMPORTANT: tells Chrome "I'll respond asynchronously"
  }
});
```

This is Chrome's built-in message passing system — `content.js` (running on LinkedIn's page) talks to `background.js` (running in Chrome's background).

### Step 5: Gemini API Call → `background.js`

The background worker does three things:

**5a. Gets the API key from storage:**
```javascript
const apiKey = await new Promise(resolve => {
  chrome.storage.local.get(["geminiApiKey"], result => {
    resolve(result.geminiApiKey || null);
  });
});
```

**5b. Builds a structured prompt:**
```javascript
const systemInstruction = `You are a Job Legitimacy Analyzer. Analyze this job listing
and look for red flags like: vague descriptions, unrealistic salary, requests for
personal info, poor grammar, too-good-to-be-true benefits...

Respond ONLY with JSON: { verdict, confidence, reasons, summary, tips }`;

const userMessage = `Job Title: ${jobData.title}
Company: ${jobData.company}
Description: ${jobData.description}
...`;
```

**5c. Calls the Gemini API:**
```javascript
const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

const response = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    system_instruction: { parts: [{ text: systemInstruction }] },
    contents: [{ role: "user", parts: [{ text: userMessage }] }],
    generationConfig: { temperature: 0.3, maxOutputTokens: 1024 }
  })
});
```

The response comes back as JSON like:
```json
{
  "verdict": "SAFE",
  "confidence": 85,
  "reasons": ["Well-known company", "Clear job requirements", "Realistic salary range"],
  "summary": "This appears to be a legitimate posting from a reputable company...",
  "tips": "Research the company's Glassdoor reviews before applying."
}
```

### Step 6: Results Overlay → `content.js` (Part 3)

The result is sent back to the content script, which creates a **slide-in overlay panel** on the right side of the LinkedIn page:

```javascript
function showResultsOverlay(result, jobData) {
  const verdictConfig = {
    SAFE:        { emoji: "✅", label: "Safe to Apply",  color: "#00c853" },
    SUSPICIOUS:  { emoji: "⚠️", label: "Suspicious",     color: "#ff9100" },
    LIKELY_FAKE: { emoji: "❌", label: "Likely Fake",    color: "#ff1744" },
  };
  // Creates a full overlay with backdrop, verdict card, reasons list, tips...
}
```

The overlay shows:
- 🏷️ **Job title & company** you analyzed
- ✅/⚠️/❌ **Color-coded verdict** with confidence bar
- 📋 **Summary** of the analysis
- 🔎 **Key findings** (bullet points)
- 💡 **Actionable tip**
- Click the **✕ button** or the backdrop to dismiss

### Step 7: API Key Setup → `popup.html` + `popup.js`

When you click the extension icon in Chrome's toolbar, `popup.html` opens. This is a tiny settings panel where you:

1. Paste your Gemini API key
2. Click **"Save Key"**
3. The key is stored in `chrome.storage.local` (persists across browser restarts)

```javascript
// popup.js saves it:
chrome.storage.local.set({ geminiApiKey: key });

// background.js reads it later:
chrome.storage.local.get(["geminiApiKey"], callback);
```

The key is **never hardcoded** and **never sent anywhere except Google's Gemini API**.

---

## 🔑 Getting Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (it looks like `AIzaSy...`)
5. Paste it in the extension popup and click Save

> ⚡ **Free tier** gives you 15 requests/minute and 1 million tokens/day — more than enough for casual use.

---

## 🚀 How to Install in Chrome

### Step 1: Open Extensions Page
- Open Chrome
- Type `chrome://extensions` in the address bar and hit Enter

### Step 2: Enable Developer Mode
- Toggle the **"Developer mode"** switch in the **top-right corner** → it should turn blue

### Step 3: Load the Extension
- Click **"Load unpacked"** (top-left)
- Navigate to the `Webetention` folder on your Desktop
- Select the folder and click **"Select"**

### Step 4: Pin the Extension
- Click the 🧩 **puzzle piece icon** in Chrome's toolbar
- Find **"LinkedIn Job Predictor"** in the list
- Click the 📌 **pin icon** to keep it visible

### Step 5: Set Your API Key
- Click the **🛡️ extension icon** in the toolbar
- Paste your Gemini API key
- Click **"Save Key"**
- You should see the status turn green: "API key configured — Ready to analyze"

### Step 6: Use It!
- Go to [linkedin.com](https://www.linkedin.com)
- Open any job listing
- Click the **"🔍 Analyze Job"** floating button (bottom-right corner)
- Wait 2-3 seconds for the AI analysis
- Read the verdict overlay!

---

## 🔄 The Complete Flow (Visual Summary)

```
You open LinkedIn job page
        ↓
Chrome auto-injects content.js + content.css
        ↓
Floating "🔍 Analyze Job" button appears (bottom-right)
        ↓
You click it
        ↓
content.js scrapes: title, company, description, salary, location, etc.
        ↓
Data sent to background.js via chrome.runtime.sendMessage()
        ↓
background.js reads your API key from chrome.storage.local
        ↓
background.js calls Gemini API with structured prompt
        ↓
Gemini returns: { verdict, confidence, reasons, summary, tips }
        ↓
background.js sends result back to content.js
        ↓
content.js creates a slide-in overlay showing the prediction
        ↓
You see: ✅ Safe / ⚠️ Suspicious / ❌ Likely Fake
```

---

## 🔮 Future: Swapping in Your Agentic Flow

When you're ready to replace the Gemini call with your agentic model, you only need to touch **one file**: `background.js`.

Replace the `callGeminiAPI()` function with your agentic pipeline. Just make sure it returns the same shape:

```javascript
{
  verdict: "SAFE" | "SUSPICIOUS" | "LIKELY_FAKE",
  confidence: 1-100,
  reasons: ["reason1", "reason2"],
  summary: "Brief analysis...",
  tips: "Actionable advice..."
}
```

Everything else (scraping, UI, overlay) stays the same.

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| Button doesn't appear | Make sure you're on `linkedin.com`. Check `chrome://extensions` that the extension is enabled |
| "No API key" error | Click the extension icon → paste your Gemini API key → Save |
| "Gemini API error (400)" | Your API key might be invalid. Generate a new one at [AI Studio](https://aistudio.google.com/app/apikey) |
| "No job data found" | Make sure you have a specific job listing open (not just the feed) |
| Extension disappeared after Chrome update | Go to `chrome://extensions` → "Load unpacked" again |

---

**Built with** ❤️ using vanilla JavaScript + Gemini AI — no frameworks, no build tools, just clean code.
