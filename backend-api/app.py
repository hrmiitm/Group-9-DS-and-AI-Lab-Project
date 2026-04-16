"""
app.py — FraudGuard Backend API
Async FastAPI application. Modular, HuggingFace Spaces ready.

Endpoints:
  GET  /                        — health check  
  GET  /api/v1/tools            — list all tools + metadata
  GET  /api/v1/tools/{name}     — single tool metadata
  POST /api/v1/run/{tool_name}  — execute any tool
  POST /api/v1/run-batch        — execute multiple tools concurrently
  POST /api/v1/llm/extract      — LLM: parse JD → structured fields
  POST /api/v1/llm/deep-research — LLM: recover missing fields via web
  POST /api/v1/llm/tool-inference — LLM: 2-4 bullet analysis of tool output
  POST /api/v1/llm/final-summary  — LLM: compile all → fraud report + verdict
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import tools_meta, tools_exec, llm

# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FraudGuard Backend API",
    description=(
        "Modular API for fake job posting detection. "
        "Provides 13 investigative tools + 4 LangChain LLM services "
        "(openai/gpt-4.1-mini via AIPipe). "
        "Deployed on HuggingFace Spaces."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow all origins (frontend on GitHub Pages + local dev) ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten to your GH Pages domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(tools_meta.router)
app.include_router(tools_exec.router)
app.include_router(llm.router)


# ── Root health check ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    from core.llm_config import llm_settings_available
    return {
        "status":       "ok",
        "service":      "FraudGuard Backend API",
        "version":      "1.1.0",
        "docs":         "/docs",
        "llm_settings": llm_settings_available(),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry point for local dev ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))   # HF Spaces uses 7860
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
