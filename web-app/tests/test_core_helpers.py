"""
tests/test_core_helpers.py
Unit tests for core/helpers.py

Tests cover:
  - safe_call: success path, exception path, non-dict return wrapping
  - normalize_website: various URL formats
  - infer_company_name: all 5 priority levels
  - _domain_to_company (via infer_company_name)
  - Edge cases and parametrized inputs
"""
import sys
import os
import pytest

# Ensure web-app root is on path
WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "..")
if WEBAPP_DIR not in sys.path:
    sys.path.insert(0, WEBAPP_DIR)

from core.helpers import safe_call, normalize_website, infer_company_name


# ---------------------------------------------------------------------------
# 1. safe_call
# ---------------------------------------------------------------------------

class TestSafeCall:

    def test_successful_call_returns_dict_result(self):
        def good_fn(x): return {"ok": True, "data": {"value": x}}
        result = safe_call("good_fn", good_fn, 42)
        assert result == {"ok": True, "data": {"value": 42}}

    def test_non_dict_result_wrapped_in_ok_data(self):
        def string_fn(): return "hello"
        result = safe_call("string_fn", string_fn)
        assert result["ok"] is True
        assert result["data"] == "hello"

    def test_none_result_wrapped(self):
        def none_fn(): return None
        result = safe_call("none_fn", none_fn)
        assert result["ok"] is True
        assert result["data"] is None

    def test_int_result_wrapped(self):
        def int_fn(): return 99
        result = safe_call("int_fn", int_fn)
        assert result["ok"] is True
        assert result["data"] == 99

    def test_list_result_wrapped(self):
        def list_fn(): return [1, 2, 3]
        result = safe_call("list_fn", list_fn)
        assert result["ok"] is True
        assert result["data"] == [1, 2, 3]

    def test_exception_returns_ok_false(self):
        def bad_fn(): raise ValueError("Something broke")
        result = safe_call("bad_fn", bad_fn)
        assert result["ok"] is False

    def test_exception_error_contains_tool_name(self):
        def bad_fn(): raise RuntimeError("crash")
        result = safe_call("my_tool", bad_fn)
        assert "my_tool" in result["error"]

    def test_exception_error_contains_exception_message(self):
        def bad_fn(): raise RuntimeError("specific crash message")
        result = safe_call("my_tool", bad_fn)
        assert "specific crash message" in result["error"]

    def test_passes_positional_args(self):
        def add(a, b): return {"ok": True, "data": a + b}
        result = safe_call("add", add, 3, 4)
        assert result["data"] == 7

    def test_passes_keyword_args(self):
        def greet(name="World"): return {"ok": True, "data": f"Hello, {name}!"}
        result = safe_call("greet", greet, name="Alice")
        assert result["data"] == "Hello, Alice!"

    def test_zero_division_caught(self):
        def divide(a, b): return {"ok": True, "data": a / b}
        result = safe_call("divide", divide, 10, 0)
        assert result["ok"] is False

    def test_type_error_caught(self):
        def typed_fn(x: int): return {"ok": True, "data": x + "wrong"}
        result = safe_call("typed_fn", typed_fn, 5)
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 2. normalize_website
# ---------------------------------------------------------------------------

class TestNormalizeWebsite:

    def test_none_returns_none(self):
        assert normalize_website(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_website("") is None

    def test_whitespace_returns_none(self):
        assert normalize_website("   ") is None

    def test_www_prefix_gets_https(self):
        result = normalize_website("www.infosys.com")
        assert result == "https://www.infosys.com"

    def test_http_url_unchanged(self):
        result = normalize_website("http://example.com")
        assert result == "http://example.com"

    def test_https_url_unchanged(self):
        result = normalize_website("https://example.com")
        assert result == "https://example.com"

    def test_bare_domain_with_dot_gets_https(self):
        result = normalize_website("infosys.com")
        assert result == "https://infosys.com"

    def test_domain_with_path_gets_https(self):
        result = normalize_website("infosys.com/careers")
        assert result == "https://infosys.com/careers"

    def test_string_with_spaces_returns_none(self):
        result = normalize_website("not a url")
        assert result is None

    def test_string_without_dot_returns_none(self):
        result = normalize_website("justtext")
        assert result is None

    def test_subdomain_gets_https(self):
        result = normalize_website("careers.infosys.com")
        assert result == "https://careers.infosys.com"

    def test_trailing_whitespace_stripped(self):
        result = normalize_website("  https://example.com  ")
        assert result == "https://example.com"

    @pytest.mark.parametrize("url,expected", [
        ("www.google.com", "https://www.google.com"),
        ("https://google.com", "https://google.com"),
        ("http://google.com", "http://google.com"),
        ("google.com", "https://google.com"),
        (None, None),
        ("", None),
        ("   ", None),
        ("no spaces here.com", None),
    ])
    def test_parametrized_normalize(self, url, expected):
        assert normalize_website(url) == expected


# ---------------------------------------------------------------------------
# 3. infer_company_name
# ---------------------------------------------------------------------------

class TestInferCompanyName:

    # ── Priority 1: parsed_job fields ──────────────────────────────────────

    def test_company_name_field_used_first(self):
        result = infer_company_name(
            raw_text="Company: SomeOtherCo",
            parsed_job={"company_name": "Infosys"},
            website=None,
            email=None,
        )
        assert result == "Infosys"

    def test_company_field_used_if_no_company_name(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={"company": "Wipro"},
            website=None,
            email=None,
        )
        assert result == "Wipro"

    def test_organization_field_used_as_fallback(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={"organization": "TCS"},
            website=None,
            email=None,
        )
        assert result == "TCS"

    def test_company_name_takes_priority_over_company(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={"company_name": "Infosys", "company": "Wipro"},
            website=None,
            email=None,
        )
        assert result == "Infosys"

    def test_parsed_name_strips_extra_whitespace(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={"company_name": "  Infosys  "},
            website=None,
            email=None,
        )
        assert result == "Infosys"

    # ── Priority 3: email domain ────────────────────────────────────────────

    def test_email_domain_used_when_no_parsed_name(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website=None,
            email="hr@infosys.com",
        )
        # "infosys.com" → base "infosys" → title "Infosys"
        assert result == "Infosys"

    def test_email_domain_titlecased(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website=None,
            email="jobs@acmecorp.com",
        )
        assert result == "Acmecorp"

    def test_email_with_hyphen_in_domain(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website=None,
            email="hr@acme-corp.com",
        )
        # "acme-corp" → base "acme" (split on ".") → title "Acme"
        assert result is not None

    # ── Priority 4: website domain ──────────────────────────────────────────

    def test_website_domain_used_when_no_email(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website="https://wipro.com/careers",
            email=None,
        )
        assert result == "Wipro"

    def test_website_www_stripped(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website="https://www.infosys.com",
            email=None,
        )
        assert result == "Infosys"

    def test_website_domain_titlecased(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website="https://hcltech.com",
            email=None,
        )
        assert result == "Hcltech"

    # ── Priority 5: raw_text regex ──────────────────────────────────────────

    def test_raw_text_company_colon_pattern(self):
        result = infer_company_name(
            raw_text="Company: Acme Technologies Pvt Ltd",
            parsed_job={},
            website=None,
            email=None,
        )
        assert result is not None
        assert "Acme" in result

    def test_raw_text_organization_colon_pattern(self):
        result = infer_company_name(
            raw_text="Organization: Global IT Solutions",
            parsed_job={},
            website=None,
            email=None,
        )
        assert result is not None

    def test_raw_text_company_dash_pattern(self):
        result = infer_company_name(
            raw_text="Company - TechStart Inc",
            parsed_job={},
            website=None,
            email=None,
        )
        assert result is not None

    # ── Returns None when nothing matches ──────────────────────────────────

    def test_all_none_returns_none(self):
        result = infer_company_name(
            raw_text="Random text with no company clues",
            parsed_job={},
            website=None,
            email=None,
        )
        assert result is None

    def test_empty_raw_text_no_other_sources_returns_none(self):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website=None,
            email=None,
        )
        assert result is None

    # ── Parametrized inputs ────────────────────────────────────────────────

    @pytest.mark.parametrize("company_field,expected", [
        ("Infosys", "Infosys"),
        ("Tata Consultancy Services", "Tata Consultancy Services"),
        ("  HCL  ", "HCL"),
        ("Wipro Limited", "Wipro Limited"),
    ])
    def test_parametrized_parsed_company_name(self, company_field, expected):
        result = infer_company_name(
            raw_text="",
            parsed_job={"company_name": company_field},
            website=None,
            email=None,
        )
        assert result == expected

    @pytest.mark.parametrize("email,expected_base", [
        ("hr@infosys.com", "Infosys"),
        ("jobs@wipro.net", "Wipro"),
        ("careers@acme.io", "Acme"),
    ])
    def test_parametrized_email_domain_inference(self, email, expected_base):
        result = infer_company_name(
            raw_text="",
            parsed_job={},
            website=None,
            email=email,
        )
        assert result is not None
        assert expected_base in result
