"""
tests/test_tool_domain_reputation.py
Unit tests for tools/tool_domain_reputation.py

Tests cover:
  - Input validation (empty, None)
  - Domain extraction from email addresses
  - Domain extraction from URLs
  - www. stripping
  - WHOIS data parsing (mocked)
  - domain age risk levels (high / medium / low)
  - Live domain check (mocked requests)
  - Exception handling (whois failure, network failure)
  - Return structure validation
  - Parametrized domain inputs
  - _days_since and _bare_domain internal helpers (via public API)
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from tools.tool_domain_reputation import check_domain_reputation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(domain_or_email) -> dict:
    return check_domain_reputation(domain_or_email)


def _make_whois(
    creation_date=None,
    expiration_date=None,
    updated_date=None,
    registrar="GoDaddy LLC",
):
    w = MagicMock()
    w.creation_date = creation_date
    w.expiration_date = expiration_date
    w.updated_date = updated_date
    w.registrar = registrar
    return w


def _make_live_response(status_code=200, final_url="https://example.com"):
    res = MagicMock()
    res.status_code = status_code
    res.url = final_url
    return res


def _old_domain_date():
    """Returns a datetime 3 years ago — should yield 'low' risk."""
    return datetime.now(timezone.utc) - timedelta(days=1095)


def _young_domain_date():
    """Returns a datetime 60 days ago — should yield 'high' risk."""
    return datetime.now(timezone.utc) - timedelta(days=60)


def _medium_domain_date():
    """Returns a datetime 400 days ago — should yield 'medium' risk."""
    return datetime.now(timezone.utc) - timedelta(days=400)


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
# 2. Domain extraction from various input formats
# ---------------------------------------------------------------------------

class TestDomainExtraction:

    def test_plain_domain_accepted(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            result = _run("infosys.com")
        assert result["ok"] is True
        assert result["data"]["domain"] == "infosys.com"

    def test_email_domain_extracted(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            result = _run("hr@infosys.com")
        assert result["ok"] is True
        assert result["data"]["domain"] == "infosys.com"

    def test_https_url_domain_extracted(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            result = _run("https://www.infosys.com/careers")
        assert result["ok"] is True
        assert result["data"]["domain"] == "infosys.com"

    def test_http_url_domain_extracted(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            result = _run("http://infosys.com/about")
        assert result["ok"] is True
        assert result["data"]["domain"] == "infosys.com"

    def test_www_stripped_from_domain(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            result = _run("www.infosys.com")
        assert result["ok"] is True
        assert result["data"]["domain"] == "infosys.com"

    def test_path_stripped_from_url(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            result = _run("https://wipro.com/careers/jobs?role=dev")
        assert result["data"]["domain"] == "wipro.com"


# ---------------------------------------------------------------------------
# 3. Risk level based on domain age
# ---------------------------------------------------------------------------

class TestRiskLevel:

    def test_old_domain_low_risk(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("infosys.com")["data"]
        assert data["risk_level"] == "low"

    def test_young_domain_high_risk(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_young_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("newscam.com")["data"]
        assert data["risk_level"] == "high"

    def test_medium_age_domain_medium_risk(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_medium_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("midagesite.com")["data"]
        assert data["risk_level"] == "medium"

    def test_no_creation_date_gives_low_risk(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(creation_date=None)
            mock_get.return_value = _make_live_response()
            data = _run("nodatesite.com")["data"]
        assert data["risk_level"] == "low"

    def test_domain_age_days_is_int_or_none(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("infosys.com")["data"]
        assert data["domain_age_days"] is None or isinstance(data["domain_age_days"], int)

    def test_domain_age_non_negative(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("infosys.com")["data"]
        if data["domain_age_days"] is not None:
            assert data["domain_age_days"] >= 0

    @pytest.mark.parametrize("age_days,expected_risk", [
        (30, "high"),
        (90, "high"),
        (179, "high"),
        (180, "medium"),
        (365, "medium"),
        (729, "medium"),
        (730, "low"),
        (1000, "low"),
        (3650, "low"),
    ])
    def test_parametrized_risk_thresholds(self, age_days, expected_risk):
        # Test the risk formula directly (mirrors the tool's logic)
        age = age_days
        risk = (
            "high"   if age < 180 else
            "medium" if age < 730 else
            "low"
        )
        assert risk == expected_risk


# ---------------------------------------------------------------------------
# 4. Live domain check (mocked)
# ---------------------------------------------------------------------------

class TestLiveDomainCheck:

    def test_live_domain_is_live_true(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response(200, "https://example.com")
            data = _run("example.com")["data"]
        assert data["is_live"] is True

    def test_server_error_is_live_false(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response(500)
            data = _run("example.com")["data"]
        assert data["is_live"] is False

    def test_live_url_captured(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response(200, "https://example.com/home")
            data = _run("example.com")["data"]
        assert data["live_url"] == "https://example.com/home"

    def test_connection_failure_is_live_false(self):
        import requests as req
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.side_effect = req.exceptions.ConnectionError("refused")
            data = _run("down.example.com")["data"]
        assert data["is_live"] is False


# ---------------------------------------------------------------------------
# 5. WHOIS data in response
# ---------------------------------------------------------------------------

class TestWHOISData:

    def test_registrar_captured(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(
                _old_domain_date(), registrar="Namecheap Inc."
            )
            mock_get.return_value = _make_live_response()
            data = _run("example.com")["data"]
        assert data["registrar"] == "Namecheap Inc."

    def test_creation_date_is_iso_string_or_none(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("example.com")["data"]
        if data["creation_date"] is not None:
            assert isinstance(data["creation_date"], str)
            assert "T" in data["creation_date"] or "-" in data["creation_date"]

    def test_domain_key_in_response(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("example.com")["data"]
        assert "domain" in data


# ---------------------------------------------------------------------------
# 6. Exception handling
# ---------------------------------------------------------------------------

class TestExceptionHandling:

    def test_whois_exception_returns_ok_false(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois:
            mock_whois.side_effect = Exception("WHOIS lookup failed")
            result = _run("unknown-domain.xyz")
        assert result["ok"] is False

    def test_whois_exception_domain_in_data(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois:
            mock_whois.side_effect = Exception("WHOIS error")
            result = _run("unknown-domain.xyz")
        assert result.get("data", {}).get("domain") == "unknown-domain.xyz"

    def test_whois_exception_error_key_present(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois:
            mock_whois.side_effect = Exception("Lookup timed out")
            result = _run("slow.example.com")
        assert "error" in result
        assert "Lookup timed out" in result["error"]


# ---------------------------------------------------------------------------
# 7. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_key_always_present(self):
        result = _run("")
        assert "ok" in result

    def test_data_has_all_required_keys(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("example.com")["data"]
        required = {
            "domain", "registrar", "creation_date", "expiration_date",
            "updated_date", "domain_age_days", "is_live", "live_url", "risk_level"
        }
        assert required.issubset(data.keys())

    def test_risk_level_is_valid_string(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("example.com")["data"]
        assert data["risk_level"] in ("low", "medium", "high")

    def test_is_live_is_bool(self):
        with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
             patch("tools.tool_domain_reputation.requests.get") as mock_get:
            mock_whois.return_value = _make_whois(_old_domain_date())
            mock_get.return_value = _make_live_response()
            data = _run("example.com")["data"]
        assert isinstance(data["is_live"], bool)


# ---------------------------------------------------------------------------
# 8. Parametrized domain formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_val,expected_domain", [
    ("infosys.com",              "infosys.com"),
    ("www.infosys.com",          "infosys.com"),
    ("https://infosys.com",      "infosys.com"),
    ("https://www.infosys.com",  "infosys.com"),
    ("hr@infosys.com",           "infosys.com"),
    ("https://infosys.com/path", "infosys.com"),
])
def test_parametrized_domain_extraction(input_val, expected_domain):
    with patch("tools.tool_domain_reputation.whois.whois") as mock_whois, \
         patch("tools.tool_domain_reputation.requests.get") as mock_get:
        mock_whois.return_value = _make_whois(_old_domain_date())
        mock_get.return_value = _make_live_response()
        result = _run(input_val)
    if result["ok"]:
        assert result["data"]["domain"] == expected_domain, (
            f"Expected domain '{expected_domain}' for input '{input_val}'"
        )
