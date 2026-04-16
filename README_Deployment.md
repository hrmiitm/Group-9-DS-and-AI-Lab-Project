# FraudGuard — Deployment & Developer Guide

FraudGuard is an AI-powered fake job posting detector. It uses 13 investigative tools (WHOIS, email/phone validation, DuckDuckGo search, Wikipedia, social profiles, job boards, website content extraction) plus a fine-tuned RoBERTa classifier and a 5-step LLM pipeline to produce a structured fraud-risk report for any job description.

---

## Architecture Overview

```mermaid
graph TB
    subgraph FE["🖥️  Frontend — React 19 + Vite  (port 5173)"]
        direction TB
        UI_IN["JDInput\nText · Drag-and-drop · File upload"]
        UI_PP["PipelineProgress\n5-step tracker"]
        UI_EI["ExtractedInfo\nStructured JD fields + Deep Research"]
        UI_TG["ToolGrid / ToolCard\n13 investigative tools"]
        UI_FR["FinalReport\nVerdict banner + Markdown report"]
        UI_SM["SettingsModal\nLLM API config · localStorage"]
        CTX["SettingsContext\nGlobal state · localStorage persistence"]
        APP["App.jsx\nPipeline orchestrator"]
        SVC["api.js\nFetch service layer"]
    end

    subgraph BE["⚙️  Backend — FastAPI  (port 7860)"]
        direction TB
        subgraph ROUTERS["Routers"]
            R_META["tools_meta\nGET /api/v1/tools"]
            R_EXEC["tools_exec\nPOST /api/v1/run/{tool}"]
            R_LLM["llm\nPOST /api/v1/llm/*"]
        end
        subgraph SERVICES["Services"]
            LC["langchain_service\nextract · research · inference · summary"]
        end
        subgraph TOOLS["Tools (13)"]
            T1["Text Analysis\nscam_signals"]
            T2["Contact\nemail_verify · domain_reputation · phone_check"]
            T3["Website\nwebsite_verify · website_content"]
            T4["Company\nwikipedia · web_search · news\nsocial_profiles · job_boards · registry"]
            T5["ML Model\nroberta_classifier"]
        end
        CFG["llm_config.py\nenv → override → default"]
        REG["tool_registry.py\nCentral tool registry"]
    end

    subgraph EXT["☁️  External Services"]
        AIPIPE["AIPipe / OpenRouter\ngpt-4.1-mini"]
        DDGS["DuckDuckGo Search\nDDGS library"]
        WHOIS["WHOIS / DNS\npython-whois · email-validator"]
        HF["HuggingFace Hub\naditya963/fraud-job-classifier"]
        WEB["Public Web\nWikipedia · Job boards · Social media"]
    end

    SVC -- "HTTP/JSON (CORS: *)" --> ROUTERS
    APP --> SVC
    APP --> CTX
    UI_IN & UI_PP & UI_EI & UI_TG & UI_FR & UI_SM --> APP

    R_LLM --> LC
    LC --> CFG
    CFG --> AIPIPE
    LC --> DDGS

    R_EXEC --> REG
    REG --> T1 & T2 & T3 & T4 & T5
    T2 --> WHOIS
    T3 & T4 --> WEB
    T4 --> DDGS
    T5 --> HF

    style FE fill:#1a1e2e,stroke:#6366f1,color:#e2e8f0
    style BE fill:#1a2035,stroke:#3b82f6,color:#e2e8f0
    style EXT fill:#1a2820,stroke:#10b981,color:#e2e8f0
    style ROUTERS fill:#0f172a,stroke:#475569,color:#94a3b8
    style SERVICES fill:#0f172a,stroke:#475569,color:#94a3b8
    style TOOLS fill:#0f172a,stroke:#475569,color:#94a3b8
```

### Analysis Pipeline (5 Steps)

```mermaid
flowchart TD
    START(["👤 User submits\nJob Description text"])

    subgraph S1["Step 1 — Extract JD"]
        E1["LLM parses raw text\ngpt-4.1-mini"]
        E2[("19 structured fields\ntitle · company · email\nphone · website · salary …")]
    end

    subgraph S2["Step 2 — Deep Research"]
        D1["DuckDuckGo search\n3 targeted queries"]
        D2["LLM enriches\nmissing fields"]
        D3[("Applied overrides\nwebsite · email · phone")]
    end

    subgraph S3["Step 3 — Run Tools"]
        direction LR
        T_A["scam_signals\nemail_verify\ndomain_reputation\nphone_check"]
        T_B["website_verify\nwebsite_content\ncompany_wikipedia\ncompany_web_search"]
        T_C["company_news\nsocial_profiles\njob_boards\nroberta_classifier"]
        BATCH["Concurrent execution\nbatches of 4"]
        T_A & T_B & T_C --> BATCH
    end

    subgraph S4["Step 4 — LLM Inference"]
        I1["Per-tool analysis\ngpt-4.1-mini × N tools"]
        I2[("2–4 bullet points\nper tool result")]
    end

    subgraph S5["Step 5 — Final Report"]
        R1["Senior fraud analyst LLM\ncompiles all inferences"]
        VERDICT{"Verdict"}
        VS["✅ SAFE"]
        VP["⚠️ SUSPICIOUS"]
        VF["❌ LIKELY_FAKE"]
    end

    START --> S1
    E1 --> E2
    S1 --> S2
    D1 --> D2 --> D3
    S2 --> S3
    S3 --> S4
    I1 --> I2
    S4 --> S5
    R1 --> VERDICT
    VERDICT --> VS & VP & VF

    style START fill:#6366f1,stroke:#818cf8,color:#fff
    style S1 fill:#1e293b,stroke:#6366f1,color:#e2e8f0
    style S2 fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style S3 fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style S4 fill:#1e293b,stroke:#06b6d4,color:#e2e8f0
    style S5 fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style VS fill:#064e3b,stroke:#10b981,color:#6ee7b7
    style VP fill:#451a03,stroke:#f59e0b,color:#fcd34d
    style VF fill:#450a0a,stroke:#ef4444,color:#fca5a5
    style VERDICT fill:#0f172a,stroke:#64748b,color:#e2e8f0
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Backend |
| Node.js | 18+ | Frontend |
| AIPipe account | — | Free at [aipipe.org](https://aipipe.org) — provides OpenRouter access |
| Git | any | — |

> **No GPU required.** The RoBERTa classifier runs on CPU. HuggingFace Spaces free tier is sufficient.

---

## Backend — Local Development

### 1. Install dependencies

```bash
cd backend-api
pip install -r requirements.txt
```

> First run downloads the RoBERTa model (~500 MB) from HuggingFace Hub. Subsequent runs use the local cache.

### 2. Set environment variables

```bash
export OPENAI_API_KEY=<your-aipipe-token>
export OPENAI_BASE_URL=https://aipipe.org/openrouter/v1
export LLM_MODEL=openai/gpt-4.1-mini
```

> If you have a regular OpenAI key, set `OPENAI_BASE_URL=https://api.openai.com/v1` instead.

### 3. Start the server

```bash
python app.py
# or explicitly:
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Server starts at **http://localhost:7860**

### 4. Verify

```bash
curl http://localhost:7860/
# → {"status":"ok","version":"1.1.0","llm_settings":{...}}

curl http://localhost:7860/api/v1/llm/status
# → {"ok":true,"api_key_from_env":true,"effective_model":"openai/gpt-4.1-mini",...}

curl http://localhost:7860/docs
# → Opens Swagger UI in browser
```

---

## Frontend — Local Development

### 1. Install dependencies

```bash
cd frontend-app
npm install
```

### 2. Start dev server

```bash
npm run dev
# → http://localhost:5173
```

### 3. Connect to backend

Open the app in your browser → click **⚙️ Settings** → confirm:
- **Backend API URL**: `http://localhost:7860`
- **API Key**: your AIPipe token (only needed if backend env vars are not set)
- **Models**: all four default to `openai/gpt-4.1-mini`

> Settings are stored in `localStorage`. Click **Reset to defaults** if you need to clear them.

---

## Backend — HuggingFace Spaces Deployment

HuggingFace Spaces runs the backend as a Docker container. The free CPU tier is sufficient — no GPU needed.

### Deployment Flow

```mermaid
flowchart TD
    A(["🧑‍💻 Developer\nlocal machine"]) --> B

    subgraph PREP["① Prepare Files"]
        B["backend-api/ directory\napp.py · requirements.txt\nDockerfile · .dockerignore\ncore/ · routers/ · services/ · tools/"]
    end

    subgraph HFS["② HuggingFace Spaces Setup"]
        C["Create new Space\nhuggingface.co/new-space\nSDK: Docker  |  Template: Blank"]
        D["Add Repository Secrets\nOPENAI_API_KEY\nOPENAI_BASE_URL\nLLM_MODEL"]
        C --> D
    end

    subgraph PUSH["③ Push Code"]
        E["git clone Space repo\ngit add backend-api/ files\ngit push origin main"]
        F["HF Spaces auto-builds\nDockerfile → container image\n⏱ ~5–10 min first build"]
        E --> F
    end

    subgraph LIVE["④ Space Running"]
        G["Health check passes\n/health → 200 OK"]
        H["Backend URL live\nhttps://username-spacename.hf.space"]
        G --> H
    end

    subgraph FE_CONNECT["⑤ Connect Frontend"]
        I["Open ⚙️ Settings in browser\nSet Backend API URL\nhttps://username-spacename.hf.space"]
        J["End-to-end pipeline\nworks on production"]
        I --> J
    end

    PREP --> HFS --> PUSH --> LIVE --> FE_CONNECT

    style PREP fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff
    style HFS fill:#172554,stroke:#3b82f6,color:#bfdbfe
    style PUSH fill:#1c1917,stroke:#d97706,color:#fde68a
    style LIVE fill:#14532d,stroke:#16a34a,color:#dcfce7
    style FE_CONNECT fill:#4a044e,stroke:#a855f7,color:#f3e8ff
    style A fill:#6366f1,stroke:#818cf8,color:#fff
```

---

### Step-by-Step Instructions

#### Step 1 — Create a HuggingFace Space

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Fill in:
   - **Owner**: your username or org
   - **Space name**: e.g. `fraudguard-api`
   - **License**: MIT (or your choice)
   - **SDK**: **Docker**
   - **Docker template**: **Blank**
   - **Visibility**: Public (free) or Private (requires Pro)
3. Click **Create Space** — you'll be taken to the Space page.

#### Step 2 — Clone the Space repository

```bash
# Install git-lfs first (required by HuggingFace)
git lfs install

# Clone your new Space (replace USERNAME and SPACENAME)
git clone https://huggingface.co/spaces/USERNAME/SPACENAME
cd SPACENAME
```

#### Step 3 — Copy backend files into the Space repo

Copy **only the contents of `backend-api/`** (not the folder itself) into the cloned Space directory:

```bash
# From the project root
cp -r backend-api/. SPACENAME/

# Verify the structure at the root of SPACENAME/:
# app.py
# requirements.txt
# Dockerfile
# .dockerignore
# core/
# routers/
# services/
# tools/
```

> The `Dockerfile` must be at the **root** of the Space repository, not inside a subdirectory.

#### Step 4 — Configure Repository Secrets

Secrets are environment variables injected at container startup — **never put API keys in code or `requirements.txt`**.

1. Go to your Space page on HuggingFace
2. Click **Settings** (top-right gear icon)
3. Scroll to **Repository Secrets**
4. Add each secret:

| Secret Name | Value | Required |
|-------------|-------|----------|
| `OPENAI_API_KEY` | Your AIPipe or OpenAI token | **Yes** |
| `OPENAI_BASE_URL` | `https://aipipe.org/openrouter/v1` | No (has default) |
| `LLM_MODEL` | `openai/gpt-4.1-mini` | No (has default) |
| `LLM_TEMPERATURE` | `0.3` | No (has default) |

#### Step 5 — Push code and trigger the build

```bash
cd SPACENAME

git add .
git commit -m "Deploy FraudGuard backend API"
git push origin main
```

HuggingFace Spaces detects the push and automatically builds the Docker image.

- **First build**: ~5–10 minutes (downloads RoBERTa model ~500 MB + PyTorch CPU wheel)
- **Subsequent builds**: ~2–3 minutes (Docker layer cache reused)

Monitor progress in the **Logs** tab on your Space page.

#### Step 6 — Verify the deployment

Once the Space shows **Running** (green dot):

```bash
# Replace with your actual Space URL
export SPACE_URL=https://USERNAME-SPACENAME.hf.space

# Health check
curl $SPACE_URL/health
# → {"status":"ok"}

# Full status with LLM config
curl $SPACE_URL/
# → {"status":"ok","version":"1.1.0","llm_settings":{"api_key_from_env":true,...}}

# Confirm model is gpt-4.1-mini
curl $SPACE_URL/api/v1/llm/status
# → {"ok":true,"effective_model":"openai/gpt-4.1-mini","api_key_from_env":true,...}

# Browse all tool metadata
curl $SPACE_URL/api/v1/tools | python3 -m json.tool
```

Swagger UI is also available at: `https://USERNAME-SPACENAME.hf.space/docs`

#### Step 7 — Connect the frontend

Open the FraudGuard frontend → click **⚙️ Settings** → set:

```
Backend API URL:  https://USERNAME-SPACENAME.hf.space
API Key:          (leave blank — the Space has it from env)
```

---

### Dockerfile Explained

The production `Dockerfile` uses a **two-stage build** for a smaller, more secure image:

```mermaid
flowchart LR
    subgraph S1["Stage 1 — builder"]
        B1["python:3.11-slim\nbase image"]
        B2["Install build tools\nbuild-essential · gcc · libffi"]
        B3["Create /opt/venv\nisolated virtualenv"]
        B4["pip install torch CPU-only\n~500 MB but no CUDA"]
        B5["pip install -r requirements.txt\nall other deps"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph S2["Stage 2 — runtime"]
        R1["python:3.11-slim\nfresh base (no build tools)"]
        R2["Install runtime libs only\ncurl · whois"]
        R3["Copy /opt/venv\nfrom builder stage"]
        R4["Create non-root user\nappuser (UID 1000)"]
        R5["Copy app source\nCHOWN to appuser"]
        R6["EXPOSE 7860\nHEALTHCHECK /health\nCMD uvicorn"]
        R1 --> R2 --> R3 --> R4 --> R5 --> R6
    end

    S1 -->|"COPY --from=builder\n/opt/venv"| S2

    style S1 fill:#1c1917,stroke:#78716c,color:#e7e5e4
    style S2 fill:#14532d,stroke:#16a34a,color:#dcfce7
```

**Why two stages?**
- The `builder` stage installs `gcc`, `build-essential`, etc. — needed to compile Python wheels
- The `runtime` stage discards all those tools, keeping the final image lean (~800 MB vs ~1.6 GB)
- Build tools are a common attack surface; removing them reduces the security footprint

**Key Dockerfile features:**

| Feature | Purpose |
|---------|---------|
| `torch CPU-only wheel` | Avoids pulling CUDA (~3 GB) — free Spaces have no GPU |
| Requirements copied before source | Docker caches the expensive pip install layer; only invalidated when `requirements.txt` changes |
| Non-root user `appuser` (UID 1000) | Security best practice; HF Spaces also runs as UID 1000 by default |
| `HEALTHCHECK` | Docker/HF Spaces polls `/health` every 30s; 120s grace for cold-start model download |
| `PYTHONUNBUFFERED=1` | Log output appears immediately in HF Spaces Logs tab |
| `HF_HOME` env var | RoBERTa model cached in `/app/.cache` — persists if a storage volume is mounted |

---

## Frontend — Production Deployment

The frontend is a React SPA built with Vite. The production output is a folder of static files (`dist/`) that can be served by any static host or a containerised Nginx server.

### Build locally first (all options)

```bash
cd frontend-app
npm install
npm run build
# → dist/  (index.html + hashed JS/CSS bundles)
```

Verify the build works before deploying:

```bash
npm run preview
# → http://localhost:4173  (serves dist/ exactly as production would)
```

---

## Frontend — HuggingFace Spaces Deployment (Docker + Nginx)

> **Recommended** when you want both backend and frontend on HuggingFace Spaces for a fully self-contained demo.

HuggingFace's **Static SDK** cannot serve a React SPA — it has no fallback route for client-side navigation. The Docker + Nginx approach solves this and gives you gzip compression, cache headers, and a health endpoint too.

### Deployment Flow

```mermaid
flowchart TD
    A(["🧑‍💻 Developer\nlocal machine"]) --> B

    subgraph PREP["① Prepare Files"]
        B["frontend-app/ directory\nDockerfile · .dockerignore · nginx.conf\npackage.json · vite.config.js\nsrc/ · public/ · index.html"]
    end

    subgraph HFS["② HuggingFace Spaces Setup"]
        C["Create new Space\nhuggingface.co/new-space\nSDK: Docker  |  Template: Blank"]
        D["No Secrets needed\n(frontend has no server-side keys)\nBackend URL set at runtime in browser"]
        C --> D
    end

    subgraph PUSH["③ Push Code"]
        E["git clone Space repo\ngit add frontend files\ngit push origin main"]
        F["HF Spaces auto-builds\nStage 1: npm install + vite build\nStage 2: Nginx serves dist/\n⏱ ~2–3 min"]
        E --> F
    end

    subgraph LIVE["④ Space Running"]
        G["HEALTHCHECK passes\n/health → 200 OK"]
        H["Frontend URL live\nhttps://username-spacename.hf.space"]
        G --> H
    end

    subgraph CONFIG["⑤ Configure Backend URL"]
        I["Open app in browser\nClick ⚙️ Settings\nSet Backend API URL to\nhttps://username-BACKEND-space.hf.space"]
        J["Full pipeline works\nJD → Verdict in browser"]
        I --> J
    end

    PREP --> HFS --> PUSH --> LIVE --> CONFIG

    style PREP fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff
    style HFS fill:#172554,stroke:#3b82f6,color:#bfdbfe
    style PUSH fill:#1c1917,stroke:#d97706,color:#fde68a
    style LIVE fill:#14532d,stroke:#16a34a,color:#dcfce7
    style CONFIG fill:#4a044e,stroke:#a855f7,color:#f3e8ff
    style A fill:#6366f1,stroke:#818cf8,color:#fff
```

---

### Step-by-Step Instructions

#### Step 1 — Create a HuggingFace Space

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**
2. Fill in:
   - **Owner**: your username or org
   - **Space name**: e.g. `fraudguard-ui`
   - **License**: MIT (or your choice)
   - **SDK**: **Docker**
   - **Docker template**: **Blank**
   - **Visibility**: Public (free) or Private (Pro)
3. Click **Create Space**

> No Secrets are needed for the frontend — it has no server-side API keys. The backend URL is configured at runtime by the user in the ⚙️ Settings modal.

#### Step 2 — Clone the Space repository

```bash
git lfs install

# Replace USERNAME and SPACENAME
git clone https://huggingface.co/spaces/USERNAME/SPACENAME
cd SPACENAME
```

#### Step 3 — Copy frontend files into the Space repo

Copy **the contents of `frontend-app/`** (not the folder itself) into the cloned Space directory:

```bash
# From the project root
cp -r frontend-app/. SPACENAME/

# The root of SPACENAME/ must contain:
# Dockerfile
# .dockerignore
# nginx.conf
# package.json
# package-lock.json
# vite.config.js
# index.html
# src/
# public/
```

> `node_modules/` and `dist/` are excluded by `.dockerignore` — Docker rebuilds them inside the container.

#### Step 4 — Push code and trigger the build

```bash
cd SPACENAME

git add .
git commit -m "Deploy FraudGuard frontend"
git push origin main
```

HuggingFace Spaces detects the push and builds automatically. Monitor progress in the **Logs** tab:

- **Stage 1 (builder)**: `npm install` + `vite build` — ~1–2 min
- **Stage 2 (runtime)**: Nginx starts — ~10 sec
- **Total**: ~2–3 min (much faster than the backend — no model download)

#### Step 5 — Verify the deployment

Once the Space shows **Running** (green dot):

```bash
export FRONTEND_URL=https://USERNAME-SPACENAME.hf.space

# Health check (served by Nginx directly)
curl $FRONTEND_URL/health
# → ok

# Main page loads (returns index.html)
curl -sI $FRONTEND_URL/ | grep "HTTP/"
# → HTTP/1.1 200 OK

# Any SPA sub-route also returns index.html (React Router handles it)
curl -sI $FRONTEND_URL/some-route | grep "HTTP/"
# → HTTP/1.1 200 OK  (not 404)
```

#### Step 6 — Connect to the backend Space

Open `https://USERNAME-SPACENAME.hf.space` in your browser:

1. Click **⚙️ Settings** (top-right)
2. Set **Backend API URL** to your backend Space URL:
   ```
   https://USERNAME-BACKEND-SPACENAME.hf.space
   ```
3. Optionally set an **API Key** if needed
4. Click **Save Settings**
5. Paste a job description and click **Analyse Job Posting** — the full pipeline runs end-to-end

---

### Dockerfile Explained (Frontend)

```mermaid
flowchart LR
    subgraph S1["Stage 1 — builder  (node:22-alpine)"]
        B1["Copy package.json\npackage-lock.json"]
        B2["npm install\ninstalls deps (resilient to\nnative module lockfile drift)"]
        B3["Copy src/ public/\nindex.html vite.config.js"]
        B4["npm run build\nvite compiles + tree-shakes\n→ /app/dist/"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph S2["Stage 2 — runtime  (nginx:1.27-alpine)"]
        R1["Copy nginx.conf\n→ /etc/nginx/conf.d/app.conf"]
        R2["Copy /app/dist\n→ /usr/share/nginx/html"]
        R3["chown -R 1000:1000\nnginx runtime dirs"]
        R4["USER 1000\nnon-root security"]
        R5["EXPOSE 7860\nHEALTHCHECK /health\nCMD nginx -g daemon off"]
        R1 --> R2 --> R3 --> R4 --> R5
    end

    S1 -->|"COPY --from=builder\n/app/dist"| S2

    style S1 fill:#1c1917,stroke:#d97706,color:#fde68a
    style S2 fill:#172554,stroke:#3b82f6,color:#bfdbfe
```

**Key decisions:**

| Feature | Why |
|---------|-----|
| `node:22-alpine` builder | Required by vite@8, @vitejs/plugin-react@6, and rolldown which all need `^20.19.0 \|\| >=22.12.0`; Node 18 is incompatible |
| `package.json` copied before `src/` | Docker caches the `npm install` layer — only re-runs when dependencies change, not on every code edit |
| `npm install` not `npm ci` | Resilient to native module lockfile drift (e.g. rolldown's `@emnapi/core`); `npm ci` fails when these platform entries are missing from the lockfile |
| `nginx:1.27-alpine` runtime | ~23 MB image; no Node runtime in production |
| `try_files $uri /index.html` | React SPA fallback — all unknown URLs return `index.html` so client-side routing works |
| Port 7860 in `nginx.conf` | HuggingFace Spaces requires the container to listen on 7860 |
| Hashed asset cache headers (`1y`) | Vite adds a content hash to every JS/CSS filename; safe to cache forever |
| `index.html` no-cache header | Forces browser to fetch fresh `index.html` after every redeployment |
| Non-root user UID 1000 | Security best practice; matches HF Spaces default UID |
| `HEALTHCHECK` via `wget` | Alpine's Nginx image includes `wget` but not `curl` |

---

## Other Frontend Deployment Options

### Option B: Vercel (simplest for frontend)

```bash
npm install -g vercel
cd frontend-app
vercel deploy --prod
```

Or connect the GitHub repo to Vercel via the dashboard — it auto-detects Vite and configures:
- **Build command**: `npm run build`
- **Output directory**: `dist`
- **Framework preset**: Vite

No additional configuration needed. Vercel handles SPA routing automatically.

### Option C: Netlify

```bash
npm install -g netlify-cli
cd frontend-app
npm run build
netlify deploy --prod --dir=dist
```

Create a `frontend-app/public/_redirects` file to enable SPA routing:

```
/*  /index.html  200
```

Or via `netlify.toml` at the project root:

```toml
[[redirects]]
  from = "/*"
  to   = "/index.html"
  status = 200
```

### Option D: GitHub Pages

1. In `frontend-app/vite.config.js`, change `base`:
   ```javascript
   base: '/Group-9-DS-and-AI-Lab-Project/'
   ```
2. Build and deploy:
   ```bash
   cd frontend-app
   npm run build
   npm install -g gh-pages
   gh-pages -d dist
   ```
3. GitHub Pages does not support SPA routing out of the box. Add a `404.html` that redirects to `index.html`, or use a [hash router](https://reactrouter.com/en/main/routers/create-hash-router).

### Connecting Any Frontend Deployment to the HuggingFace Backend

After deploying the frontend anywhere, configure the backend URL once in the Settings modal — no rebuild required:

```
⚙️ Settings → Backend API URL → https://USERNAME-BACKEND-SPACENAME.hf.space
```

Settings persist in `localStorage`. Users only need to set this once per browser.

---

## Environment Variables Reference

### Backend (`backend-api/`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | AIPipe or OpenAI API token |
| `OPENAI_BASE_URL` | `https://aipipe.org/openrouter/v1` | LLM proxy endpoint |
| `LLM_MODEL` | `openai/gpt-4.1-mini` | Model identifier passed to LangChain |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (0.0–1.0) |
| `PORT` | `7860` | Server listen port |
| `ROBERTA_MODEL_ID` | `aditya963/fraud-job-classifier` | HuggingFace model ID for RoBERTa |
| `ROBERTA_THRESHOLD` | `0.87` | Fraud probability threshold (0.0–1.0) |

### Frontend (`frontend-app/`)

All settings are configurable at runtime via the **⚙️ Settings** modal. No build-time env vars needed. Defaults (in `src/contexts/SettingsContext.jsx`):

| Setting | Default |
|---------|---------|
| Backend URL | `http://localhost:7860` |
| LLM Base URL | `https://aipipe.org/openrouter/v1` |
| All 4 models | `openai/gpt-4.1-mini` |

---

## API Endpoints Reference

All endpoints served by the FastAPI backend. Swagger UI available at `/docs`, ReDoc at `/redoc`.

### Health & Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check + version + LLM config status |
| `GET` | `/health` | Simple `{"status":"ok"}` probe |
| `GET` | `/api/v1/llm/status` | Current LLM configuration (which settings come from env) |

### Tool Registry

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/tools` | List all 13 tools with metadata (label, icon, description, input_schema) |
| `GET` | `/api/v1/tools/{tool_name}` | Single tool metadata |

### Tool Execution

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/run/{tool_name}` | Execute any tool. Body: free-form JSON matching the tool's `input_schema` |
| `POST` | `/api/v1/run-batch` | Execute multiple tools concurrently. Body: `[{"tool_name": str, ...kwargs}]` |

**Tool execution response format:**
```json
{
  "ok": true,
  "tool": "email_verify",
  "label": "Email Verification",
  "result": {
    "ok": true,
    "data": { "is_syntax_valid": true, "has_mx_record": true, ... }
  }
}
```

### LLM Services

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/llm/extract` | Parse raw JD text → 19 structured fields |
| `POST` | `/api/v1/llm/deep-research` | DuckDuckGo + LLM to recover missing email/phone/website |
| `POST` | `/api/v1/llm/tool-inference` | Generate 2–4 bullet analysis of one tool's output |
| `POST` | `/api/v1/llm/final-summary` | Compile all inferences → fraud verdict + markdown report |

All LLM endpoints accept an optional `llm_config` block to override the server's default model:
```json
{
  "llm_config": {
    "api_key": "sk-...",
    "base_url": "https://aipipe.org/openrouter/v1",
    "model": "openai/gpt-4.1-mini"
  }
}
```

### Final Summary Response

```json
{
  "ok": true,
  "verdict": "SAFE | SUSPICIOUS | LIKELY_FAKE",
  "report": "## Executive Summary\n..."
}
```

---

## Tools Reference

| Tool Name | Category | Description |
|-----------|----------|-------------|
| `scam_signals` | text_analysis | Keyword-based scam pattern detection |
| `email_verify` | contact | Syntax + DNS MX record validation |
| `domain_reputation` | contact | WHOIS domain age and risk scoring |
| `website_verify` | website | HTTP/HTTPS liveness + SSL check |
| `website_content` | website | Text/metadata extraction via trafilatura |
| `company_wikipedia` | company | Wikipedia REST API company lookup |
| `company_web_search` | company | Multi-angle DuckDuckGo company search |
| `company_news` | company | DuckDuckGo news articles about company |
| `social_profiles` | company | 7-platform social media presence check |
| `job_boards` | company | Presence on LinkedIn, Indeed, Glassdoor, etc. |
| `phone_check` | contact | Phone number parsing and validation |
| `roberta_classifier` | ml_model | Fine-tuned RoBERTa fraud classifier (threshold: 0.87) |
| `company_registry` | company | Company registry lookup *(stub — not yet implemented)* |

---

## Troubleshooting

### "Cannot reach backend" in the UI

1. Check that the backend is running: `curl http://localhost:7860/health`
2. Open ⚙️ Settings and verify the Backend API URL matches the actual port
3. Check for CORS issues — the backend allows all origins by default
4. If deployed on HF Spaces, wait for the Space to finish building (check Logs tab)

### "LLM inference unavailable" on tool cards

1. Verify `OPENAI_API_KEY` is set (check `GET /api/v1/llm/status`)
2. Test the key: try a simple curl to the AIPipe endpoint
3. Check AIPipe account credits at [aipipe.org](https://aipipe.org)

### 401 Unauthorized from LLM

- AIPipe token expired or revoked — generate a new one at aipipe.org
- Wrong `OPENAI_BASE_URL` — confirm it is `https://aipipe.org/openrouter/v1`

### Tool returns empty or error

- **email_verify / domain_reputation**: requires internet access; fails in sandboxed environments
- **WHOIS timeouts**: some registrars block WHOIS queries — the tool degrades gracefully
- **RoBERTa slow on first run**: model downloads ~500 MB; subsequent calls use cache

### Frontend Settings not updating

- Previous settings are stored in `localStorage` — click **Reset to defaults** in ⚙️ Settings
- Clear `localStorage` in browser DevTools → Application → Storage → `fraudguard_settings`

### HuggingFace Space stuck building

- Check the Build logs in the Space dashboard
- Ensure `Dockerfile` and `requirements.txt` are at the repo root of the Space
- The `torch` package makes the first build slow (~10 min) — this is normal

---

## Project Structure

```mermaid
graph TD
    ROOT["📁 Group-9-DS-and-AI-Lab-Project"]

    ROOT --> BA["📁 backend-api\nFastAPI · Python 3.11"]
    ROOT --> FA["📁 frontend-app\nReact 19 · Vite · Node 22"]
    ROOT --> RD["📄 README_Deployment.md"]

    BA --> APP["app.py\nEntry point · CORS · Router registration"]
    BA --> REQ["requirements.txt\nPip dependencies"]
    BA --> DOCK["Dockerfile\nHuggingFace Spaces image"]
    BA --> CORE["📁 core/"]
    BA --> ROUT["📁 routers/"]
    BA --> SERV["📁 services/"]
    BA --> TDIR["📁 tools/  ×13 modules"]

    CORE --> LC["llm_config.py\nenv → override → default resolution"]
    CORE --> TR["tool_registry.py\nCentral tool catalogue"]

    ROUT --> RM["tools_meta.py\nGET /api/v1/tools"]
    ROUT --> RE["tools_exec.py\nPOST /api/v1/run/{tool}"]
    ROUT --> RL["llm.py\nPOST /api/v1/llm/* · GET /llm/status"]

    SERV --> LS["langchain_service.py\nextract · research · inference · summary"]

    FA --> SRC["📁 src/"]
    FA --> PKG["package.json"]
    FA --> VCFG["vite.config.js"]
    FA --> FDF["Dockerfile\nNode build → Nginx runtime"]
    FA --> FDIG[".dockerignore\nExcludes node_modules · dist · .env"]
    FA --> NGX["nginx.conf\nPort 7860 · SPA fallback · gzip · cache"]

    SRC --> AJSX["App.jsx\nPipeline orchestrator · All state"]
    SRC --> CTX["📁 contexts/\nSettingsContext.jsx — localStorage"]
    SRC --> APIS["📁 services/\napi.js — fetch wrappers"]
    SRC --> COMP["📁 components/"]

    COMP --> C1["Header.jsx\nLogo · Settings trigger"]
    COMP --> C2["JDInput.jsx\nText area · Drag-and-drop"]
    COMP --> C3["PipelineProgress.jsx\n5-step tracker"]
    COMP --> C4["ExtractedInfo.jsx\nJD fields · Deep Research"]
    COMP --> C5["ToolGrid.jsx\nCategory filter · Card grid"]
    COMP --> C6["ToolCard.jsx\nRun · Inference · Copy JSON"]
    COMP --> C7["FinalReport.jsx\nVerdict banner · Markdown"]
    COMP --> C8["SettingsModal.jsx\nLLM API configuration"]

    style ROOT fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff
    style BA fill:#172554,stroke:#3b82f6,color:#bfdbfe
    style FA fill:#14532d,stroke:#22c55e,color:#bbf7d0
    style CORE fill:#0f172a,stroke:#475569,color:#94a3b8
    style ROUT fill:#0f172a,stroke:#475569,color:#94a3b8
    style SERV fill:#0f172a,stroke:#475569,color:#94a3b8
    style TDIR fill:#0f172a,stroke:#475569,color:#94a3b8
    style SRC fill:#0f172a,stroke:#475569,color:#94a3b8
    style CTX fill:#0f172a,stroke:#475569,color:#94a3b8
    style APIS fill:#0f172a,stroke:#475569,color:#94a3b8
    style COMP fill:#0f172a,stroke:#475569,color:#94a3b8
```

## Deployment Flow

```mermaid
flowchart LR
    subgraph DEV["🧑‍💻 Local Development"]
        L_BE["Backend\npython app.py\nlocalhost:7860"]
        L_FE["Frontend\nnpm run dev\nlocalhost:5173"]
        L_BE <-- "HTTP/JSON" --> L_FE
    end

    subgraph BUILD["🔨 Build & Package"]
        B_BE["backend-api/Dockerfile\npython:3.11-slim → uvicorn"]
        B_FE["frontend-app/Dockerfile\nnode:22-alpine → nginx:1.27-alpine"]
    end

    subgraph PROD["🚀 Production — HuggingFace Spaces"]
        subgraph HF_BE_BOX["Backend Space  (Docker SDK)"]
            HF_BE["FastAPI + uvicorn\nhttps://user-fraudguard-api.hf.space\nport 7860"]
            HF_SEC["Repository Secrets\nOPENAI_API_KEY\nOPENAI_BASE_URL · LLM_MODEL"]
            HF_SEC -.->|"env vars at runtime"| HF_BE
        end
        subgraph HF_FE_BOX["Frontend Space  (Docker SDK)"]
            HF_FE["Nginx serving React SPA\nhttps://user-fraudguard-ui.hf.space\nport 7860"]
            HF_CFG["No secrets needed\nBackend URL set in\n⚙️ Settings modal at runtime"]
            HF_CFG -.-> HF_FE
        end
        HF_BE <-- "CORS: *\nHTTP/JSON" --> HF_FE
    end

    subgraph ALT["🔀 Alternative Frontend Hosts"]
        VER["Vercel"]
        NET["Netlify"]
        GHP["GitHub Pages"]
    end

    subgraph EXT2["☁️ External APIs"]
        AIPIPE2["AIPipe / OpenRouter\ngpt-4.1-mini"]
        HFH["HuggingFace Hub\nRoBERTa model"]
    end

    DEV -->|"git push"| BUILD
    B_BE -->|"auto-build\n& deploy"| HF_BE_BOX
    B_FE -->|"auto-build\n& deploy"| HF_FE_BOX
    B_FE -->|"npm run build\ndeploy dist/"| ALT

    HF_BE --> AIPIPE2
    HF_BE --> HFH

    USER(["👤 End User\nbrowser"])
    USER --> HF_FE
    USER --> ALT

    style DEV fill:#1e1b4b,stroke:#6366f1,color:#e0e7ff
    style BUILD fill:#1c1917,stroke:#78716c,color:#e7e5e4
    style PROD fill:#052e16,stroke:#16a34a,color:#dcfce7
    style ALT fill:#1e293b,stroke:#64748b,color:#cbd5e1
    style EXT2 fill:#0c1a2e,stroke:#0ea5e9,color:#bae6fd
    style HF_BE_BOX fill:#0f172a,stroke:#22c55e,color:#94a3b8
    style HF_FE_BOX fill:#0f172a,stroke:#6366f1,color:#94a3b8
    style USER fill:#6366f1,stroke:#818cf8,color:#fff
```
