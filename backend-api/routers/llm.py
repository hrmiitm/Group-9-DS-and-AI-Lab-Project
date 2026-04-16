"""
routers/llm.py
LangChain LLM endpoints:

  POST /api/v1/llm/extract        — parse JD text → structured fields
  POST /api/v1/llm/deep-research  — enrich missing fields via DuckDuckGo + LLM
  POST /api/v1/llm/tool-inference — summarise tool output in bullet points
  POST /api/v1/llm/final-summary  — compile all inferences → fraud report + verdict

All endpoints accept an optional "llm_config" block for per-request model override
when HuggingFace Spaces environment variables are not set.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from core.llm_config import LLMConfig
from services.langchain_service import (
    extract_jd,
    deep_research,
    tool_inference,
    final_summary,
)

router = APIRouter(prefix="/api/v1/llm", tags=["LLM Services"])


# ── Shared LLM config model ────────────────────────────────────────────────────
class LLMConfigPayload(BaseModel):
    api_key:     Optional[str]   = None
    base_url:    Optional[str]   = None
    model:       Optional[str]   = None
    temperature: Optional[float] = None


# ── 1. Extract JD ─────────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    raw_text:   str
    llm_config: Optional[LLMConfigPayload] = None


@router.post("/extract")
async def api_extract_jd(body: ExtractRequest):
    """
    Parse raw job description text into a structured JobPosting dict.

    Input : raw_text (string), optional llm_config
    Output: {ok, data: {title, company_name, contact_email, ...all 19 fields}}
    """
    cfg = LLMConfig(**body.llm_config.model_dump()) if body.llm_config else None
    return await extract_jd(body.raw_text, llm_config=cfg)


# ── 2. Deep Research ──────────────────────────────────────────────────────────
class DeepResearchRequest(BaseModel):
    job_dict:   dict
    raw_text:   str = ""
    llm_config: Optional[LLMConfigPayload] = None


@router.post("/deep-research")
async def api_deep_research(body: DeepResearchRequest):
    """
    Use DuckDuckGo + LLM to recover missing contact and website fields.

    Input : job_dict (extracted fields), raw_text, optional llm_config
    Output: {ok, data: {missing_from_jd, applied_overrides, candidates, summary, sources}}
    """
    cfg = LLMConfig(**body.llm_config.model_dump()) if body.llm_config else None
    return await deep_research(body.job_dict, body.raw_text, llm_config=cfg)


# ── 3. Tool Inference ─────────────────────────────────────────────────────────
class ToolInferenceRequest(BaseModel):
    tool_name:   str
    tool_label:  str = ""
    tool_result: dict
    job_dict:    dict = {}
    llm_config:  Optional[LLMConfigPayload] = None


@router.post("/tool-inference")
async def api_tool_inference(body: ToolInferenceRequest):
    """
    Generate 2-4 plain-English bullet points analysing one tool's output.

    Input : tool_name, tool_result (raw tool JSON), job_dict, optional llm_config
    Output: {ok, inference: str, bullets: [str]}
    """
    cfg = LLMConfig(**body.llm_config.model_dump()) if body.llm_config else None
    label = body.tool_label or body.tool_name.replace("_", " ").title()
    return await tool_inference(
        body.tool_name, label, body.tool_result, body.job_dict, llm_config=cfg
    )


# ── 4. Final Summary ──────────────────────────────────────────────────────────
class FinalSummaryRequest(BaseModel):
    job_dict:            dict
    tool_results:        dict[str, Any] = {}
    tool_inferences:     dict[str, str] = {}
    web_search_snippets: Optional[list[dict]] = None
    llm_config:          Optional[LLMConfigPayload] = None


@router.post("/final-summary")
async def api_final_summary(body: FinalSummaryRequest):
    """
    Compile all tool inferences into a comprehensive fraud report with verdict.

    Input : job_dict, tool_inferences, optional web_search_snippets, optional llm_config
    Output: {ok, verdict: "SAFE"|"SUSPICIOUS"|"LIKELY_FAKE", report: markdown_string}
    """
    cfg = LLMConfig(**body.llm_config.model_dump()) if body.llm_config else None
    return await final_summary(
        body.job_dict,
        body.tool_results,
        body.tool_inferences,
        body.web_search_snippets,
        llm_config=cfg,
    )
