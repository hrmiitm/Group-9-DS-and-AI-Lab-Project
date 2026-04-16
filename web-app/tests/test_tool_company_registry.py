"""
tests/test_tool_company_registry.py
Unit tests for tools/tool_company_registry.py

The registry tool is a stub that always returns a "Coming Soon" error.
Tests verify the contract is stable so callers can rely on it.
"""
import pytest

from tools.tool_company_registry import get_company_registry


# ---------------------------------------------------------------------------
# 1. Always-error contract
# ---------------------------------------------------------------------------

class TestAlwaysErrorContract:

    def test_returns_dict(self):
        result = get_company_registry()
        assert isinstance(result, dict)

    def test_ok_is_false(self):
        result = get_company_registry()
        assert result["ok"] is False

    def test_error_key_present(self):
        result = get_company_registry()
        assert "error" in result

    def test_error_is_string(self):
        result = get_company_registry()
        assert isinstance(result["error"], str)

    def test_error_is_non_empty(self):
        result = get_company_registry()
        assert len(result["error"]) > 0

    def test_error_mentions_coming_soon(self):
        result = get_company_registry()
        assert "Coming Soon" in result["error"] or "coming soon" in result["error"].lower()

    def test_no_data_key_or_data_is_none(self):
        result = get_company_registry()
        assert "data" not in result or result.get("data") is None


# ---------------------------------------------------------------------------
# 2. Accepts arbitrary arguments (stub signature)
# ---------------------------------------------------------------------------

class TestAcceptsArbitraryArgs:

    def test_called_with_no_args(self):
        result = get_company_registry()
        assert result["ok"] is False

    def test_called_with_positional_args(self):
        result = get_company_registry("Infosys", "IN")
        assert result["ok"] is False

    def test_called_with_keyword_args(self):
        result = get_company_registry(company_name="Infosys", country="IN")
        assert result["ok"] is False

    def test_called_with_mixed_args(self):
        result = get_company_registry("Infosys", country="IN", cin="U12345")
        assert result["ok"] is False

    def test_called_with_none_args(self):
        result = get_company_registry(None, None)
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 3. Idempotency — same result every call
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_multiple_calls_same_result(self):
        results = [get_company_registry() for _ in range(5)]
        for r in results:
            assert r["ok"] is False
            assert "error" in r

    def test_error_message_consistent(self):
        results = [get_company_registry() for _ in range(3)]
        errors = [r["error"] for r in results]
        assert len(set(errors)) == 1  # all identical


# ---------------------------------------------------------------------------
# 4. Parametrized calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args,kwargs", [
    ((), {}),
    (("Wipro",), {}),
    (("TCS", "IN"), {}),
    ((), {"company": "HCL", "country": "IN"}),
    (("Reliance",), {"cin": "L17110MH1973PLC019786"}),
])
def test_parametrized_calls_always_fail(args, kwargs):
    result = get_company_registry(*args, **kwargs)
    assert result["ok"] is False
    assert "error" in result
