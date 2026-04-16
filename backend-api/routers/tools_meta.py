"""
routers/tools_meta.py
GET /api/v1/tools   — returns the full tool registry metadata.
"""
from fastapi import APIRouter
from core.tool_registry import get_tool_meta
from core.llm_config import llm_settings_available

router = APIRouter(prefix="/api/v1", tags=["Tools Meta"])


@router.get("/tools")
async def list_tools():
    """
    Returns metadata for all 13 available tools:
    label, icon, description, input_schema, output_fields, category.
    Also returns LLM configuration status (which settings come from env).
    """
    return {
        "ok": True,
        "total": len(get_tool_meta()),
        "llm_settings": llm_settings_available(),
        "tools": get_tool_meta(),
    }


@router.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Return metadata for a single tool by name."""
    meta = get_tool_meta()
    if tool_name not in meta:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return {"ok": True, "tool": meta[tool_name]}
