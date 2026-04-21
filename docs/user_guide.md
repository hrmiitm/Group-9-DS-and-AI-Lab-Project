# User Guide

**FraudGuard — Fake Job Listing Detection**
*A guide for non-technical users*

---

## What Does This App Do and Why Is It Useful?

FraudGuard is an AI-powered tool that helps you determine whether a job listing is real or fraudulent **before** you apply, share personal information, or pay any fees.

Every year, thousands of job seekers lose money and personal data to fake job postings that look completely legitimate. FraudGuard analyzes a job posting using:

- A trained AI model (RoBERTa) that has read thousands of real and fake job ads
- 12 real-time verification checks (company domain, email validity, website health, news, social media, job boards, and more)
- A final AI-written investigation report with a clear verdict: **SAFE**, **SUSPICIOUS**, or **LIKELY FAKE**

**When should you use it?**
- Before applying to any job you found online
- When a job offer looks too good to be true
- When a recruiter contacts you out of nowhere
- Before paying any "registration fee", "training fee", or "equipment deposit"

---

## How to Launch the Web-App (Step-by-Step)

### Prerequisites

- Python 3.11 or newer installed on your computer
- A terminal (Command Prompt on Windows, Terminal on Mac/Linux)
- An API key from AIPipe or OpenRouter (your instructor should provide this)

### Step 1: Download the Project

```bash
git clone https://github.com/hrmiitm/Group-9-DS-and-AI-Lab-Project.git
cd Group-9-DS-and-AI-Lab-Project
```

Or download the ZIP from GitHub and extract it.

### Step 2: Install Dependencies

```bash
pip install flask langchain-openai langchain-community pydantic requests \
            beautifulsoup4 ddgs trafilatura python-whois email-validator \
            phonenumbers markupsafe
```

### Step 3: Set Your API Key

On Mac/Linux:
```bash
export OPENAI_API_KEY="paste-your-api-key-here"
export OPENAI_BASE_URL="https://aipipe.org/openrouter/v1"
export LLM_MODEL="openai/gpt-4o-mini"
```

On Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=paste-your-api-key-here
set OPENAI_BASE_URL=https://aipipe.org/openrouter/v1
set LLM_MODEL=openai/gpt-4o-mini
```

### Step 4: Start the App

```bash
python web-app/app.py
```

You should see:
```
==================================================
  Webetention — Fraud Job Detector
==================================================
  URL     : http://localhost:5000
==================================================
```

### Step 5: Open Your Browser

Go to: **http://localhost:5000**

The app's home page will appear with three tabs:
- **Paste Text** — copy-paste a job description
- **Upload File** — upload a PDF, Word doc, or text file
- **LinkedIn URL** — paste a direct LinkedIn job URL

---

## How to Install and Use the Chrome Extension (Step-by-Step)

The Chrome extension lets you analyze any LinkedIn job posting **without leaving LinkedIn**.

### Step 1: Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (it starts with `AIzaSy...`)

### Step 2: Install the Extension

1. Open Google Chrome
2. Type `chrome://extensions` in the address bar and press Enter
3. Toggle **"Developer mode"** ON (top-right corner)
4. Click **"Load unpacked"** (top-left)
5. Navigate to the project folder and select the **`web-extension`** subfolder
6. Click **"Select"**

### Step 3: Pin the Extension

1. Click the 🧩 puzzle piece icon in Chrome's toolbar
2. Find **"LinkedIn Job Predictor"** in the list
3. Click the 📌 pin icon to keep it always visible

### Step 4: Add Your API Key

1. Click the extension icon (🛡️) in the Chrome toolbar
2. Paste your Gemini API key into the text box
3. Click **"Save Key"**
4. The status should turn green: **"API key configured — Ready to analyze"**

### Step 5: Analyze a Job on LinkedIn

1. Go to [linkedin.com](https://www.linkedin.com)
2. Open any job listing
3. Look for the **"🔍 Analyze Job"** floating button in the bottom-right corner of your screen
4. Click it
5. Wait 3–5 seconds
6. An overlay panel will slide in from the right with your verdict

---

## What Inputs to Provide, What Outputs to Expect

### Inputs

| Input Type | What to Provide | Example |
|---|---|---|
| **Text paste** | Copy the full job description from any website | Copy everything from "Job Title" to "How to Apply" |
| **File upload** | A PDF or Word document containing the job posting | A job offer email saved as PDF |
| **LinkedIn URL** | The full URL of a LinkedIn job listing page | `https://www.linkedin.com/jobs/view/1234567890/` |

**Tip:** More information = better analysis. Include the company name, job description, requirements, salary range, and contact details if available.

### Outputs (Web-App)

After analysis, you will see a **Results page** with:

1. **Verdict Banner** — Large colored card at the top:
   - 🟢 **SAFE** — No significant red flags detected
   - 🟡 **SUSPICIOUS** — Some concerning signals, verify before applying
   - 🔴 **LIKELY FAKE** — Multiple strong fraud indicators detected

2. **Job Information Card** — The 16 structured fields the AI extracted (title, company, location, salary, etc.)

3. **Deep Research Section** — Any additional information the system found online to fill in missing details

4. **Tool Evidence Grid** — 12 verification cards, each showing:
   - Status (OK / Warning / Failed)
   - What the tool found
   - What it means for fraud risk

5. **Final Report** — A detailed AI-written investigation report explaining the verdict, red flags, supporting evidence, and what you should do next

### Outputs (Chrome Extension)

A slide-in overlay showing:
- ✅ / ⚠️ / ❌ **Color-coded verdict** with confidence score
- **Summary** of the analysis
- **Key findings** (bullet points)
- **Actionable tip**

---

## Example Use Cases

### Use Case 1: Work-From-Home Data Entry Job

**Input:**
```
Title: Data Entry Specialist (Work From Home)
Company: (Not listed)
Description: Earn $500 per day working from home. No experience needed.
Simply type names and addresses. Send your bank details to get started.
Requirements: None
Salary: $500/day
Contact: dataentry.jobs2026@gmail.com
```

**Expected Output:** 🔴 LIKELY FAKE
- **Why:** No company name, unrealistic salary, Gmail contact, "send bank details" request, no experience required — multiple high-confidence fraud signals.

---

### Use Case 2: Legitimate Software Engineering Role

**Input:**
```
Title: Senior Software Engineer
Company: Infosys Technologies
Location: Bengaluru, Karnataka, India
Description: We are seeking an experienced software engineer...
Requirements: 5+ years Python, B.Tech/M.Tech Computer Science
Salary: 18-24 LPA
Employment: Full-time
Website: www.infosys.com
```

**Expected Output:** 🟢 SAFE
- **Why:** Well-known company, verifiable website, realistic salary range, complete job details, professional language.

---

### Use Case 3: Suspicious Overseas Opportunity

**Input:**
```
Title: Customer Service Representative
Company: Global Trading Corp
Location: Dubai (accommodation provided)
Description: Exciting opportunity! Earn $5000/month. Immediate joining.
Requirements: Basic English, no experience needed
Contact: globaltrading.hr@outlook.com
Note: Pay $200 registration fee before interview
```

**Expected Output:** 🔴 LIKELY FAKE
- **Why:** Request for upfront payment, unverifiable company, free email contact, "immediate joining" urgency, location mismatch with company claims.

---

### Use Case 4: Early-Career Position with Mixed Signals

**Input:**
```
Title: Marketing Intern
Company: StartupX (stealth mode)
Description: Join our growing team! Work on exciting campaigns.
Stipend: 5000-8000/month
Duration: 6 months
Contact: hr@startupx.in
```

**Expected Output:** 🟡 SUSPICIOUS
- **Why:** Company in "stealth mode" (unverifiable), domain is new/unregistered, no company website, but salary is realistic and no upfront payment requested.

---

### Use Case 5: Government Job Scam

**Input:**
```
Title: Clerk Grade B – State Public Service Commission
Description: Direct recruitment. No exam required. Join immediately.
Salary: ₹45,000/month
Requirements: 12th Pass
Contact: psc.recruitment2026@gmail.com
Fee: Pay ₹2,500 processing fee via UPI
```

**Expected Output:** 🔴 LIKELY FAKE
- **Why:** Government jobs never use Gmail contacts, never require upfront fees, and never offer "no exam" recruitment for PSC positions.

---

## Troubleshooting

### Web-App Issues

| Problem | Solution |
|---|---|
| App won't start | Check Python version (`python --version` → must be 3.11+). Check `OPENAI_API_KEY` is set. |
| `ModuleNotFoundError` | Re-run: `pip install flask langchain-openai langchain-community pydantic requests beautifulsoup4 ddgs trafilatura python-whois email-validator phonenumbers markupsafe` |
| Analysis shows "error" | Usually means API key is wrong or expired. Check the key and try again. |
| Analysis takes very long | Normal for large job descriptions (can take 60–90 seconds). Don't refresh the page. |
| "LinkedIn URL" tab returns empty | LinkedIn blocks automated scraping. Paste the text manually instead. |
| Results page is blank | Refresh once. If persists, re-submit the job description. |
| Error 402 from LLM | Your API key has insufficient credits. Check your account balance or use a different key. |

### Chrome Extension Issues

| Problem | Solution |
|---|---|
| "🔍 Analyze Job" button doesn't appear | Make sure you're on a specific LinkedIn **job listing page** (not the job feed). Try refreshing. |
| "No API key" error | Click the extension icon → paste your Gemini API key → click "Save Key". |
| "Gemini API error (400)" | Your API key is invalid. Generate a new one at [AI Studio](https://aistudio.google.com/app/apikey). |
| "No job data found" | Open a specific job listing (click on a job title, not just scroll past it). |
| Extension disappeared after Chrome update | Go to `chrome://extensions` → "Load unpacked" → select `web-extension/` folder again. |
| Analysis returns SUSPICIOUS for a job you trust | This can happen for legitimate startups with low online presence. Use your judgment — the extension flags signals, not definitive fraud. |

---

## Screenshots

<!-- INSERT SCREENSHOT: Web-app home page showing the 3-tab input form (Paste Text, Upload File, LinkedIn URL) -->

<!-- INSERT SCREENSHOT: Web-app results page showing the green SAFE verdict banner at the top -->

<!-- INSERT SCREENSHOT: Web-app results page showing the LIKELY FAKE red verdict with the 12 tool evidence grid -->

<!-- INSERT SCREENSHOT: Web-app deep research section showing enriched company data discovered online -->

<!-- INSERT SCREENSHOT: Chrome extension overlay on LinkedIn showing a SUSPICIOUS verdict with reasons list -->

<!-- INSERT SCREENSHOT: Chrome extension settings popup with API key field -->
