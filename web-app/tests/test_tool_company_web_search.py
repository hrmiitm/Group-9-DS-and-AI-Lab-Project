"""
tests/test_tool_company_web_search.py
Unit tests for tools/tool_company_web_search.py

Tests cover:
  - Input validation (empty, None)
  - Return structure validation
  - Search angles presence and structure
  - Mocked DDGS responses
  - Error handling within individual search angles
  - company_name preserved in output
  - Parametrized company names
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_company_web_search import search_company_web, SEARCH_ANGLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(company_name) -> dict:
    return search_company_web(company_name)


def _mock_ddgs_result(n: int = 2):
    return [
        {
            "title": f"Result {i}",
            "href": f"https://example.com/result-{i}",
            "body": f"Snippet for result {i}",
        }
        for i in range(n)
    ]


class _MockDDGS:
    """Context manager mock for DDGS that returns fixed results."""

    def __init__(self, results=None, raise_for=None):
        self._results = results or _mock_ddgs_result(2)
        self._raise_for = raise_for or set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def text(self, query, max_results=4):
        for angle in self._raise_for:
            if angle in query:
                raise Exception(f"Search failed for: {query}")
        return self._results


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
# 2. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_true_on_success(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
            result = _run("Infosys")
        assert result["ok"] is True

    def test_data_has_company_name(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
            data = _run("Infosys")["data"]
        assert data["company_name"] == "Infosys"

    def test_data_has_searches_key(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
            data = _run("Infosys")["data"]
        assert "searches" in data

    def test_searches_is_dict(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
            data = _run("Infosys")["data"]
        assert isinstance(data["searches"], dict)

    def test_all_search_angles_present(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
            data = _run("Infosys")["data"]
        for angle in SEARCH_ANGLES.keys():
            assert angle in data["searches"], f"Missing angle: {angle}"

    def test_search_angle_results_are_lists(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            assert isinstance(results, list), f"Angle '{angle}' results not a list"

    def test_search_result_has_title_url_snippet(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS(2)):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            for r in results:
                assert "title" in r or "error" in r
                assert "url" in r or "error" in r


# ---------------------------------------------------------------------------
# 3. DDGS result mapping
# ---------------------------------------------------------------------------

class TestDDGSResultMapping:

    def test_title_mapped_correctly(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS(1)):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            if results:
                assert results[0]["title"] == "Result 0"
                break

    def test_url_mapped_correctly(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS(1)):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            if results:
                assert results[0]["url"] == "https://example.com/result-0"
                break

    def test_snippet_mapped_correctly(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS(1)):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            if results:
                assert results[0]["snippet"] == "Snippet for result 0"
                break

    def test_no_results_gives_empty_list(self):
        with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS(0)):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            assert isinstance(results, list)

    def test_none_results_gives_empty_list(self):
        mock = _MockDDGS()
        mock._results = None
        with patch("tools.tool_company_web_search.DDGS", return_value=mock):
            data = _run("Infosys")["data"]
        for angle, results in data["searches"].items():
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 4. Error handling within individual angles
# ---------------------------------------------------------------------------

class TestAngleErrorHandling:

    def test_error_in_one_angle_does_not_crash(self):
        """If one angle raises, the others should still succeed."""

        class PartialFailDDGS(_MockDDGS):
            def text(self, query, max_results=4):
                if "scam" in query.lower():
                    raise Exception("Rate limited")
                return _mock_ddgs_result(2)

        with patch("tools.tool_company_web_search.DDGS", return_value=PartialFailDDGS()):
            result = _run("Infosys")

        assert result["ok"] is True
        assert "scam_fraud" in result["data"]["searches"]
        assert "error" in result["data"]["searches"]["scam_fraud"][0]

    def test_error_result_contains_error_key(self):
        class AlwaysFailDDGS(_MockDDGS):
            def text(self, query, max_results=4):
                raise Exception("All searches failed")

        with patch("tools.tool_company_web_search.DDGS", return_value=AlwaysFailDDGS()):
            result = _run("Infosys")

        assert result["ok"] is True  # function-level ok is still True
        for angle, results in result["data"]["searches"].items():
            assert len(results) > 0
            assert "error" in results[0]


# ---------------------------------------------------------------------------
# 5. SEARCH_ANGLES integrity
# ---------------------------------------------------------------------------

class TestSearchAnglesIntegrity:

    def test_search_angles_is_dict(self):
        assert isinstance(SEARCH_ANGLES, dict)

    def test_search_angles_not_empty(self):
        assert len(SEARCH_ANGLES) > 0

    def test_all_angle_values_are_format_strings(self):
        for angle, template in SEARCH_ANGLES.items():
            assert "{}" in template, (
                f"Angle '{angle}' template missing '{{}}' placeholder"
            )

    def test_expected_angles_exist(self):
        expected = {"general_info", "employee_review", "scam_fraud", "glassdoor", "linkedin_page"}
        assert expected.issubset(set(SEARCH_ANGLES.keys()))


# ---------------------------------------------------------------------------
# 6. Parametrized company names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("company_name", [
    "Infosys",
    "Wipro",
    "Tata Consultancy Services",
    "Reliance Industries",
    "HDFC Bank",
])
def test_parametrized_company_names_ok(company_name):
    with patch("tools.tool_company_web_search.DDGS", return_value=_MockDDGS()):
        result = _run(company_name)
    assert result["ok"] is True
    assert result["data"]["company_name"] == company_name


@pytest.mark.parametrize("bad_input", ["", None])
def test_parametrized_bad_inputs_error(bad_input):
    result = _run(bad_input)
    assert result["ok"] is False
