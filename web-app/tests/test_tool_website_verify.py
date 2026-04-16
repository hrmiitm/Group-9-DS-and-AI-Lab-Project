"""
tests/test_tool_website_verify.py
Unit tests for tools/tool_website_verify.py

Tests cover:
  - Input validation (empty, None, no URL)
  - URL normalisation (adding https:// prefix)
  - Live / SSL / redirect detection (mocked)
  - Return structure validation
  - Status code edge cases (2xx / 3xx / 4xx / 5xx)
  - SSL error handling
  - Connection error handling
  - Parametrized URL inputs
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_website_verify import verify_website


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(url) -> dict:
    return verify_website(url)


def _mock_response(
    status_code: int = 200,
    final_url: str = "https://example.com",
    history: list | None = None,
    elapsed_ms: float = 0.050,
    server: str = "nginx",
    content_type: str = "text/html",
) -> MagicMock:
    res = MagicMock()
    res.status_code = status_code
    res.url = final_url
    res.history = history or []

    elapsed = MagicMock()
    elapsed.total_seconds.return_value = elapsed_ms
    res.elapsed = elapsed

    res.headers = {"Server": server, "Content-Type": content_type}
    res.headers.get = lambda k, d=None: {"Server": server, "Content-Type": content_type}.get(k, d)
    return res


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_string_returns_error(self):
        result = _run("")
        assert result["ok"] is False
        assert "error" in result

    def test_none_returns_error(self):
        result = _run(None)
        assert result["ok"] is False
        assert "error" in result

    def test_error_message_non_empty(self):
        result = _run("")
        assert len(result["error"]) > 0


# ---------------------------------------------------------------------------
# 2. URL normalisation
# ---------------------------------------------------------------------------

class TestURLNormalisation:

    def test_bare_domain_gets_https_prefix(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(final_url="https://example.com")
            result = _run("example.com")
        assert result["ok"] is True
        # input_url should have https://
        assert result["data"]["input_url"].startswith("https://")

    def test_www_domain_gets_https_prefix(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(final_url="https://www.example.com")
            result = _run("www.example.com")
        assert result["ok"] is True

    def test_http_url_not_modified(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                status_code=200,
                final_url="http://example.com"
            )
            result = _run("http://example.com")
        assert result["ok"] is True
        assert result["data"]["input_url"] == "http://example.com"

    def test_https_url_not_modified(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(final_url="https://example.com")
            result = _run("https://example.com")
        assert result["data"]["input_url"] == "https://example.com"


# ---------------------------------------------------------------------------
# 3. Successful response (mocked)
# ---------------------------------------------------------------------------

class TestSuccessfulResponse:

    def test_200_response_is_live(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert data["is_live"] is True

    def test_200_ok_true(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            result = _run("https://example.com")
        assert result["ok"] is True

    def test_status_code_captured(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert data["status_code"] == 200

    def test_final_url_captured(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://redirected.com")
            data = _run("https://example.com")["data"]
        assert data["final_url"] == "https://redirected.com"

    def test_ssl_valid_for_https_final_url(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert data["ssl_valid"] is True

    def test_ssl_invalid_for_http_final_url(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "http://example.com")
            data = _run("http://example.com")["data"]
        assert data["ssl_valid"] is False

    def test_response_time_ms_is_int(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com", elapsed_ms=0.123)
            data = _run("https://example.com")["data"]
        assert isinstance(data["response_time_ms"], int)
        assert data["response_time_ms"] == 123

    def test_server_header_captured(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com", server="Apache")
            data = _run("https://example.com")["data"]
        assert data["server"] == "Apache"

    def test_content_type_captured(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, "https://example.com", content_type="text/html; charset=utf-8"
            )
            data = _run("https://example.com")["data"]
        assert "text/html" in data["content_type"]

    def test_no_redirects_gives_empty_chain(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com", history=[])
            data = _run("https://example.com")["data"]
        assert data["redirect_count"] == 0
        assert data["redirect_chain"] == []

    def test_redirect_chain_captured(self):
        redirect1 = MagicMock()
        redirect1.url = "http://example.com"
        redirect1.status_code = 301

        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, "https://example.com", history=[redirect1]
            )
            data = _run("http://example.com")["data"]

        assert data["redirect_count"] == 1
        assert len(data["redirect_chain"]) == 1
        assert data["redirect_chain"][0]["url"] == "http://example.com"
        assert data["redirect_chain"][0]["status"] == 301


# ---------------------------------------------------------------------------
# 4. Status code edge cases
# ---------------------------------------------------------------------------

class TestStatusCodes:

    @pytest.mark.parametrize("status_code", [200, 201, 204, 301, 302, 304])
    def test_2xx_3xx_is_live(self, status_code):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(status_code, "https://example.com")
            data = _run("https://example.com")["data"]
        assert data["is_live"] is True

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_5xx_is_not_live(self, status_code):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(status_code, "https://example.com")
            data = _run("https://example.com")["data"]
        assert data["is_live"] is False

    def test_404_is_live_since_server_responded(self):
        # 404 is < 500, so is_live = True per the tool logic
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404, "https://example.com")
            data = _run("https://example.com")["data"]
        assert data["is_live"] is True


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_ssl_error_returns_ok_false(self):
        import requests as req
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.SSLError("SSL handshake failed")
            result = _run("https://bad-ssl.example.com")
        assert result["ok"] is False
        assert "ssl" in result["error"].lower() or "SSL" in result["error"]

    def test_connection_error_returns_ok_false(self):
        import requests as req
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.ConnectionError("Connection refused")
            result = _run("https://unreachable.example.com")
        assert result["ok"] is False
        assert "error" in result

    def test_timeout_returns_ok_false(self):
        import requests as req
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.Timeout("Request timed out")
            result = _run("https://slow.example.com")
        assert result["ok"] is False

    def test_generic_exception_returns_ok_false(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.side_effect = Exception("Unexpected error")
            result = _run("https://example.com")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 6. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_all_keys_present_on_success(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            result = _run("https://example.com")

        assert result["ok"] is True
        required = {
            "input_url", "final_url", "status_code", "is_live",
            "ssl_valid", "redirect_count", "redirect_chain",
            "response_time_ms", "server", "content_type"
        }
        assert required.issubset(result["data"].keys())

    def test_is_live_is_bool(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert isinstance(data["is_live"], bool)

    def test_ssl_valid_is_bool(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert isinstance(data["ssl_valid"], bool)

    def test_redirect_count_is_int(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert isinstance(data["redirect_count"], int)

    def test_redirect_chain_is_list(self):
        with patch("tools.tool_website_verify.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, "https://example.com")
            data = _run("https://example.com")["data"]
        assert isinstance(data["redirect_chain"], list)


# ---------------------------------------------------------------------------
# 7. Parametrized URL inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://example.com",
    "http://example.com",
    "example.com",
    "www.example.com",
    "https://subdomain.example.com/path?query=1",
])
def test_parametrized_url_inputs_ok(url):
    with patch("tools.tool_website_verify.requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, "https://example.com")
        result = _run(url)
    assert result["ok"] is True
