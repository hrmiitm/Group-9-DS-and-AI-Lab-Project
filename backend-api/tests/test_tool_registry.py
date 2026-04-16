"""
backend-api/tests/test_tool_registry.py
Unit tests for core/tool_registry.py

Tests cover:
  - TOOL_REGISTRY structure and required keys for every tool
  - get_tool_meta(): strips 'fn' key by default, includes it when asked
  - get_tool_fn(): returns callable for valid tool, None for unknown
  - All 13 expected tools are registered
  - Categories are valid strings
  - Input schema structure (type, required, description)
  - Output fields are lists
  - Callable is actually callable
  - Parametrized per-tool tests
"""
import sys
import os
import types
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Patch external dependencies before importing registry
# ---------------------------------------------------------------------------

def _mock_if_missing(name: str):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]


# Patch ddgs
ddgs_mod = _mock_if_missing("ddgs")
ddgs_mod.DDGS = MagicMock()

# Patch trafilatura
traf = _mock_if_missing("trafilatura")
traf.fetch_url = MagicMock(return_value=None)
traf.extract = MagicMock(return_value=None)
traf.extract_metadata = MagicMock(return_value=None)

# Patch whois
whois_mod = _mock_if_missing("whois")
whois_mod.whois = MagicMock(return_value=MagicMock())

# Patch phonenumbers
pn = _mock_if_missing("phonenumbers")
pn.parse = MagicMock()
pn.format_number = MagicMock(return_value="+911234567890")
pn.is_valid_number = MagicMock(return_value=True)
pn.is_possible_number = MagicMock(return_value=True)
pn.region_code_for_number = MagicMock(return_value="IN")
pn.number_type = MagicMock(return_value=1)
pn.PhoneNumberFormat = MagicMock()
pn.NumberParseException = Exception
_mock_if_missing("phonenumbers.carrier")
_mock_if_missing("phonenumbers.geocoder")
_mock_if_missing("phonenumbers.timezone")

# Patch email_validator
ev = _mock_if_missing("email_validator")
ev.validate_email = MagicMock()
ev.EmailNotValidError = Exception

# Patch requests (used by several tools)
req_mod = _mock_if_missing("requests")
req_mod.get = MagicMock(return_value=MagicMock(status_code=200, url="https://example.com"))
exc_mod = types.ModuleType("requests.exceptions")
exc_mod.SSLError = Exception
exc_mod.ConnectionError = Exception
exc_mod.Timeout = Exception
sys.modules["requests.exceptions"] = exc_mod
req_mod.exceptions = exc_mod

# Patch tool_roberta (ML model tool — heavy deps)
roberta_mod = _mock_if_missing("tools.tool_roberta")
roberta_mod.classify_job_roberta = MagicMock(return_value={"ok": True, "data": {}})

# Ensure backend-api is on path
BACKEND_API_DIR = os.path.join(os.path.dirname(__file__), "..")
if BACKEND_API_DIR not in sys.path:
    sys.path.insert(0, BACKEND_API_DIR)

from core.tool_registry import TOOL_REGISTRY, get_tool_meta, get_tool_fn


# ---------------------------------------------------------------------------
# Expected tools
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "scam_signals",
    "email_verify",
    "domain_reputation",
    "website_verify",
    "website_content",
    "company_wikipedia",
    "company_web_search",
    "company_news",
    "social_profiles",
    "job_boards",
    "phone_check",
    "company_registry",
    "roberta_classifier",
}

VALID_CATEGORIES = {
    "text_analysis", "contact", "website", "company", "ml_model"
}


# ---------------------------------------------------------------------------
# 1. Registry completeness
# ---------------------------------------------------------------------------

class TestRegistryCompleteness:

    def test_registry_is_dict(self):
        assert isinstance(TOOL_REGISTRY, dict)

    def test_registry_not_empty(self):
        assert len(TOOL_REGISTRY) > 0

    def test_all_expected_tools_registered(self):
        for tool in EXPECTED_TOOLS:
            assert tool in TOOL_REGISTRY, f"Tool '{tool}' not in TOOL_REGISTRY"

    def test_registry_has_13_tools(self):
        assert len(TOOL_REGISTRY) == 13

    def test_no_unexpected_tools(self):
        registered = set(TOOL_REGISTRY.keys())
        extra = registered - EXPECTED_TOOLS
        assert extra == set(), f"Unexpected tools found: {extra}"


# ---------------------------------------------------------------------------
# 2. Required keys in every tool entry
# ---------------------------------------------------------------------------

class TestRequiredKeys:

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_label_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "label" in TOOL_REGISTRY[tool_name]

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_icon_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "icon" in TOOL_REGISTRY[tool_name]

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_description_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "description" in TOOL_REGISTRY[tool_name]

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_fn_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "fn" in TOOL_REGISTRY[tool_name]

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_input_schema_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "input_schema" in TOOL_REGISTRY[tool_name]

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_output_fields_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "output_fields" in TOOL_REGISTRY[tool_name]

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_category_present(self, tool_name):
        if tool_name not in TOOL_REGISTRY:
            pytest.skip(f"Tool '{tool_name}' not registered")
        assert "category" in TOOL_REGISTRY[tool_name]


# ---------------------------------------------------------------------------
# 3. Field type validation
# ---------------------------------------------------------------------------

class TestFieldTypes:

    def test_all_labels_are_strings(self):
        for name, entry in TOOL_REGISTRY.items():
            assert isinstance(entry["label"], str), f"Tool '{name}' label not str"

    def test_all_labels_non_empty(self):
        for name, entry in TOOL_REGISTRY.items():
            assert len(entry["label"]) > 0, f"Tool '{name}' has empty label"

    def test_all_icons_are_strings(self):
        for name, entry in TOOL_REGISTRY.items():
            assert isinstance(entry["icon"], str)

    def test_all_descriptions_are_strings(self):
        for name, entry in TOOL_REGISTRY.items():
            assert isinstance(entry["description"], str)

    def test_all_descriptions_non_empty(self):
        for name, entry in TOOL_REGISTRY.items():
            assert len(entry["description"]) > 0

    def test_all_fns_are_callable(self):
        for name, entry in TOOL_REGISTRY.items():
            assert callable(entry["fn"]), f"Tool '{name}' fn is not callable"

    def test_all_input_schemas_are_dicts(self):
        for name, entry in TOOL_REGISTRY.items():
            assert isinstance(entry["input_schema"], dict), (
                f"Tool '{name}' input_schema not dict"
            )

    def test_all_output_fields_are_lists(self):
        for name, entry in TOOL_REGISTRY.items():
            assert isinstance(entry["output_fields"], list), (
                f"Tool '{name}' output_fields not list"
            )

    def test_all_categories_are_valid(self):
        for name, entry in TOOL_REGISTRY.items():
            assert entry["category"] in VALID_CATEGORIES, (
                f"Tool '{name}' has invalid category: {entry['category']}"
            )


# ---------------------------------------------------------------------------
# 4. Input schema structure
# ---------------------------------------------------------------------------

class TestInputSchemaStructure:

    def test_each_input_param_has_type(self):
        for tool_name, entry in TOOL_REGISTRY.items():
            for param, meta in entry["input_schema"].items():
                assert "type" in meta, (
                    f"Tool '{tool_name}', param '{param}' missing 'type'"
                )

    def test_each_input_param_has_required_flag(self):
        for tool_name, entry in TOOL_REGISTRY.items():
            for param, meta in entry["input_schema"].items():
                assert "required" in meta, (
                    f"Tool '{tool_name}', param '{param}' missing 'required'"
                )

    def test_each_input_param_has_description(self):
        for tool_name, entry in TOOL_REGISTRY.items():
            for param, meta in entry["input_schema"].items():
                assert "description" in meta, (
                    f"Tool '{tool_name}', param '{param}' missing 'description'"
                )

    def test_required_flag_is_bool(self):
        for tool_name, entry in TOOL_REGISTRY.items():
            for param, meta in entry["input_schema"].items():
                assert isinstance(meta["required"], bool), (
                    f"Tool '{tool_name}', param '{param}' required is not bool"
                )

    def test_type_is_valid_string(self):
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        for tool_name, entry in TOOL_REGISTRY.items():
            for param, meta in entry["input_schema"].items():
                assert meta["type"] in valid_types, (
                    f"Tool '{tool_name}', param '{param}' has invalid type '{meta['type']}'"
                )


# ---------------------------------------------------------------------------
# 5. get_tool_meta() function
# ---------------------------------------------------------------------------

class TestGetToolMeta:

    def test_returns_dict(self):
        result = get_tool_meta()
        assert isinstance(result, dict)

    def test_contains_all_tools(self):
        result = get_tool_meta()
        for tool in EXPECTED_TOOLS:
            assert tool in result

    def test_fn_excluded_by_default(self):
        result = get_tool_meta(include_fn=False)
        for name, entry in result.items():
            assert "fn" not in entry, f"Tool '{name}' has 'fn' when it should be excluded"

    def test_fn_included_when_requested(self):
        result = get_tool_meta(include_fn=True)
        for name, entry in result.items():
            assert "fn" in entry, f"Tool '{name}' missing 'fn' when include_fn=True"

    def test_label_preserved_without_fn(self):
        result = get_tool_meta(include_fn=False)
        for name, entry in result.items():
            assert "label" in entry

    def test_category_preserved_without_fn(self):
        result = get_tool_meta(include_fn=False)
        for name, entry in result.items():
            assert "category" in entry


# ---------------------------------------------------------------------------
# 6. get_tool_fn() function
# ---------------------------------------------------------------------------

class TestGetToolFn:

    def test_returns_callable_for_valid_tool(self):
        fn = get_tool_fn("scam_signals")
        assert callable(fn)

    def test_returns_none_for_unknown_tool(self):
        fn = get_tool_fn("nonexistent_tool")
        assert fn is None

    def test_returns_none_for_empty_string(self):
        fn = get_tool_fn("")
        assert fn is None

    def test_returns_none_for_none(self):
        fn = get_tool_fn(None)
        assert fn is None

    @pytest.mark.parametrize("tool_name", list(EXPECTED_TOOLS))
    def test_all_tools_return_callable(self, tool_name):
        fn = get_tool_fn(tool_name)
        assert fn is not None, f"get_tool_fn('{tool_name}') returned None"
        assert callable(fn), f"get_tool_fn('{tool_name}') returned non-callable"


# ---------------------------------------------------------------------------
# 7. Specific tool metadata spot-checks
# ---------------------------------------------------------------------------

class TestSpecificToolMetadata:

    def test_scam_signals_category_text_analysis(self):
        assert TOOL_REGISTRY["scam_signals"]["category"] == "text_analysis"

    def test_scam_signals_required_input_is_job_text(self):
        schema = TOOL_REGISTRY["scam_signals"]["input_schema"]
        assert "job_text" in schema
        assert schema["job_text"]["required"] is True

    def test_email_verify_required_input_is_email(self):
        schema = TOOL_REGISTRY["email_verify"]["input_schema"]
        assert "email" in schema
        assert schema["email"]["required"] is True

    def test_job_boards_has_optional_location(self):
        schema = TOOL_REGISTRY["job_boards"]["input_schema"]
        assert "location" in schema
        assert schema["location"]["required"] is False

    def test_company_news_has_default_max_results(self):
        schema = TOOL_REGISTRY["company_news"]["input_schema"]
        assert "max_results" in schema
        assert schema["max_results"].get("default") == 8

    def test_company_registry_is_stub(self):
        entry = TOOL_REGISTRY["company_registry"]
        assert entry.get("is_stub") is True

    def test_roberta_classifier_is_ml_category(self):
        assert TOOL_REGISTRY["roberta_classifier"]["category"] == "ml_model"

    def test_phone_check_has_region_with_default(self):
        schema = TOOL_REGISTRY["phone_check"]["input_schema"]
        assert "region" in schema
        assert schema["region"].get("default") == "IN"
