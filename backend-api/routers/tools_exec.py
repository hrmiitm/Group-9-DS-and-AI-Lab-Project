"""
routers/tools_exec.py
POST /api/v1/run/{tool_name}  — dynamically execute any tool from the registry.

The request body is a free-form JSON object matching the tool's input_schema.
The tool function is called in a thread pool executor (asyncio.to_thread) so
blocking I/O tools don't stall the async event loop.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.tool_registry import get_tool_fn, get_tool_meta, TOOL_REGISTRY

router = APIRouter(prefix="/api/v1", tags=["Tool Execution"])


class RunRequest(BaseModel):
    """Generic tool execution request — forward all fields to the tool function."""
    model_config = {"extra": "allow"}   # accept any keys


@router.post("/run/{tool_name}")
async def run_tool(tool_name: str, body: RunRequest):
    """
    Dynamically execute a registered tool.

    - Validates tool exists
    - Maps request body fields to tool function kwargs
    - Runs blocking tool in thread pool
    - Returns raw tool output + tool metadata
    """
    fn = get_tool_fn(tool_name)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found. Call GET /api/v1/tools to see available tools.")

    meta = TOOL_REGISTRY[tool_name]
    input_schema: dict = meta.get("input_schema", {})

    # Build kwargs from request body
    body_data: dict = body.model_dump(exclude_none=True)
    kwargs: dict = {}

    for param_name, param_meta in input_schema.items():
        if param_name in body_data:
            kwargs[param_name] = body_data[param_name]
        elif param_meta.get("required", False):
            raise HTTPException(
                status_code=422,
                detail=f"Required field '{param_name}' missing. Tool '{tool_name}' requires: {list(input_schema.keys())}"
            )
        elif "default" in param_meta:
            kwargs[param_name] = param_meta["default"]

    # Run tool in thread pool (all tools are sync/blocking)
    try:
        result = await asyncio.to_thread(fn, **kwargs)
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    return {
        "ok":      result.get("ok", False),
        "tool":    tool_name,
        "label":   meta.get("label"),
        "result":  result,
    }


@router.post("/run-batch")
async def run_batch(requests: list[dict]):
    """
    Run multiple tools concurrently. Each item: {"tool_name": str, ...kwargs}
    Returns list of results in the same order.
    """
    async def _run_one(req: dict):
        tool_name = req.pop("tool_name", None)
        if not tool_name:
            return {"ok": False, "error": "tool_name is required in each batch item"}
        fn = get_tool_fn(tool_name)
        if fn is None:
            return {"ok": False, "tool": tool_name, "error": "Tool not found"}
        try:
            result = await asyncio.to_thread(fn, **req)
            return {"ok": result.get("ok", False), "tool": tool_name, "result": result}
        except Exception as e:
            return {"ok": False, "tool": tool_name, "error": str(e)}

    tasks = [_run_one(r.copy()) for r in requests]
    results = await asyncio.gather(*tasks)
    return {"ok": True, "results": list(results)}
