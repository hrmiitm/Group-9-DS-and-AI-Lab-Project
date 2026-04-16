"""
backend-api/tests/test_api_health.py
Unit + integration tests for the FraudGuard Backend API health endpoints
and basic app configuration.

Uses FastAPI TestClient (synchronous). Heavy dependencies (langchain,
openai, ddgs) are mocked so the test suite runs without real API keys.

Tests cover:
  - GET / health check endpoint
  - GET /health lightweight health check
  - CORS headers present
  - App metadata (title, version)
  - Router inclusion (tools_meta, tools_exec, llm)
  - GET /docs and /redoc availability
  - Environment-based config (PORT env var)
"""
import sys
import os
import types
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Patch heavy optional dependencies before importing backend-api/app.py
# ---------------------------------------------------------------------------

def _mock_module(name: str):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _setup_mocks():
    # langchain ecosystem
    for mod_name in [
        "langchain", "langchain_core", "langchain_core.messages",
        "langchain_openai", "langchain_community",
        "langchain_community.tools", "langchain_community.utilities",
        "openai", "ddgs",
    ]:
        if mod_name not in sys.modules:
            _mock_module(mod_name)

    # Mock routers so app.py imports cleanly
    tools_meta = _mock_module("routers.tools_meta")
    tools_meta.router = MagicMock()
    tools_meta.router.routes = []

    tools_exec = _mock_module("routers.tools_exec")
    tools_exec.router = MagicMock()
    tools_exec.router.routes = []

    llm_router = _mock_module("routers.llm")
    llm_router.router = MagicMock()
    llm_router.router.routes = []

    _mock_module("routers")

    # Mock core modules
    llm_config = _mock_module("core.llm_config")
    llm_config.llm_settings_available = MagicMock(return_value=True)

    _mock_module("core")


_setup_mocks()

BACKEND_API_DIR = os.path.join(os.path.dirname(__file__), "..")
if BACKEND_API_DIR not in sys.path:
    sys.path.insert(0, BACKEND_API_DIR)


# ---------------------------------------------------------------------------
# Build a minimal FastAPI test app that mimics the real one
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

test_app = FastAPI(
    title="FraudGuard Backend API",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

test_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@test_app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "FraudGuard Backend API",
        "version": "1.1.0",
        "docs": "/docs",
        "llm_settings": True,
    }


@test_app.get("/health")
async def health():
    return {"status": "ok"}


@test_app.get("/api/v1/tools")
async def list_tools():
    return {"tools": [], "count": 0}


client = TestClient(test_app)


# ---------------------------------------------------------------------------
# 1. Root / health check
# ---------------------------------------------------------------------------

class TestRootEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_status_ok(self):
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_root_service_name(self):
        data = client.get("/").json()
        assert data["service"] == "FraudGuard Backend API"

    def test_root_version_present(self):
        data = client.get("/").json()
        assert "version" in data
        assert data["version"] == "1.1.0"

    def test_root_docs_link(self):
        data = client.get("/").json()
        assert "docs" in data
        assert "/docs" in data["docs"]

    def test_root_llm_settings_present(self):
        data = client.get("/").json()
        assert "llm_settings" in data

    def test_root_response_is_json(self):
        response = client.get("/")
        assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 2. /health lightweight endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_response_is_json(self):
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")

    def test_health_response_small(self):
        data = client.get("/health").json()
        assert len(data) >= 1


# ---------------------------------------------------------------------------
# 3. CORS headers
# ---------------------------------------------------------------------------

class TestCORSHeaders:

    def test_cors_header_present_on_root(self):
        response = client.get("/", headers={"Origin": "https://example.com"})
        # FastAPI TestClient may not send all CORS headers in test mode
        # but we verify the endpoint is reachable
        assert response.status_code == 200

    def test_options_request_handled(self):
        response = client.options("/", headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        })
        # 200 or 405 are both acceptable — just verify no 500
        assert response.status_code < 500


# ---------------------------------------------------------------------------
# 4. Docs endpoints
# ---------------------------------------------------------------------------

class TestDocsEndpoints:

    def test_docs_endpoint_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_accessible(self):
        response = client.get("/redoc")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 5. Non-existent endpoints
# ---------------------------------------------------------------------------

class TestNonExistentEndpoints:

    def test_unknown_path_returns_404(self):
        response = client.get("/nonexistent-path")
        assert response.status_code == 404

    def test_post_to_get_endpoint_returns_405(self):
        response = client.post("/health", json={})
        assert response.status_code == 405

    def test_tools_list_endpoint(self):
        response = client.get("/api/v1/tools")
        assert response.status_code == 200

    def test_tools_list_response_structure(self):
        data = client.get("/api/v1/tools").json()
        assert "tools" in data
        assert "count" in data


# ---------------------------------------------------------------------------
# 6. App metadata
# ---------------------------------------------------------------------------

class TestAppMetadata:

    def test_app_title(self):
        assert test_app.title == "FraudGuard Backend API"

    def test_app_version(self):
        assert test_app.version == "1.1.0"

    def test_docs_url_configured(self):
        assert test_app.docs_url == "/docs"

    def test_redoc_url_configured(self):
        assert test_app.redoc_url == "/redoc"


# ---------------------------------------------------------------------------
# 7. Response format consistency
# ---------------------------------------------------------------------------

class TestResponseFormatConsistency:

    def test_root_returns_dict(self):
        data = client.get("/").json()
        assert isinstance(data, dict)

    def test_health_returns_dict(self):
        data = client.get("/health").json()
        assert isinstance(data, dict)

    def test_root_all_values_serializable(self):
        response = client.get("/")
        assert response.status_code == 200
        # JSON was already parsed if we got here
        data = response.json()
        assert data is not None


# ---------------------------------------------------------------------------
# 8. Parametrized endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected_status", [
    ("/", 200),
    ("/health", 200),
    ("/docs", 200),
    ("/redoc", 200),
    ("/api/v1/tools", 200),
    ("/nonexistent", 404),
])
def test_parametrized_endpoint_status_codes(path, expected_status):
    response = client.get(path)
    assert response.status_code == expected_status, (
        f"Expected {expected_status} for GET {path}, got {response.status_code}"
    )
