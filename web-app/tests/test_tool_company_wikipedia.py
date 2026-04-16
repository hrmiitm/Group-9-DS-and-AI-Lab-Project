"""
tests/test_tool_company_wikipedia.py
Unit tests for tools/tool_company_wikipedia.py

Tests cover:
  - Input validation (empty, None)
  - Successful Wikipedia fetch (mocked requests)
  - 404 fallback to opensearch
  - Opensearch returning no results
  - HTTP errors
  - Return structure validation
  - Parametrized company names
  - Slug generation (spaces → underscores)
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_company_wikipedia import get_company_wikipedia


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(company_name) -> dict:
    return get_company_wikipedia(company_name)


def _make_wikipedia_response(title="Infosys", description="Indian IT company"):
    return {
        "title": title,
        "description": description,
        "extract": f"{title} is a global technology company.",
        "content_urls": {
            "desktop": {"page": f"https://en.wikipedia.org/wiki/{title}"}
        },
        "thumbnail": {"source": f"https://upload.wikimedia.org/wikipedia/{title}.jpg"},
    }


def _mock_http_200(title="Infosys"):
    res = MagicMock()
    res.status_code = 200
    res.json.return_value = _make_wikipedia_response(title)
    res.raise_for_status = MagicMock()
    return res


def _mock_http_404():
    res = MagicMock()
    res.status_code = 404
    res.raise_for_status.side_effect = Exception("404 Not Found")
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
# 2. Successful fetch (mocked)
# ---------------------------------------------------------------------------

class TestSuccessfulFetch:

    def test_ok_true_on_200(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            result = _run("Infosys")
        assert result["ok"] is True

    def test_title_extracted(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            data = _run("Infosys")["data"]
        assert data["title"] == "Infosys"

    def test_description_extracted(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            data = _run("Infosys")["data"]
        assert data["description"] == "Indian IT company"

    def test_extract_text_returned(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            data = _run("Infosys")["data"]
        assert "Infosys" in data["extract"]

    def test_wikipedia_url_returned(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            data = _run("Infosys")["data"]
        assert "wikipedia.org" in (data["wikipedia_url"] or "")

    def test_thumbnail_url_returned(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            data = _run("Infosys")["data"]
        assert data["thumbnail_url"] is not None

    def test_data_has_all_required_keys(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Infosys")
            data = _run("Infosys")["data"]
        required = {"title", "description", "extract", "wikipedia_url", "thumbnail_url"}
        assert required.issubset(data.keys())

    def test_space_in_name_converted_to_underscore_in_slug(self):
        """The slug 'Tata_Consultancy_Services' should be used in the URL."""
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200("Tata Consultancy Services")
            _run("Tata Consultancy Services")
        call_url = mock_get.call_args[0][0]
        assert "Tata_Consultancy_Services" in call_url


# ---------------------------------------------------------------------------
# 3. 404 fallback to opensearch
# ---------------------------------------------------------------------------

class TestOpenSearchFallback:

    def test_404_triggers_opensearch(self):
        opensearch_res = MagicMock()
        opensearch_res.status_code = 200
        opensearch_res.json.return_value = [
            "Wipro",
            ["Wipro Limited"],
            ["Indian IT company"],
            ["https://en.wikipedia.org/wiki/Wipro_Limited"],
        ]
        opensearch_res.raise_for_status = MagicMock()

        summary_res = _mock_http_200("Wipro Limited")

        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_http_404(),   # first: direct slug → 404
                opensearch_res,     # second: opensearch
                summary_res,        # third: summary for best result
            ]
            result = _run("Wipro")

        assert result["ok"] is True

    def test_opensearch_no_results_returns_error(self):
        opensearch_res = MagicMock()
        opensearch_res.status_code = 200
        opensearch_res.json.return_value = ["query", [], [], []]
        opensearch_res.raise_for_status = MagicMock()

        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_http_404(),
                opensearch_res,
            ]
            result = _run("NonExistentCompanyXYZ123")

        assert result["ok"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# 4. Exception handling
# ---------------------------------------------------------------------------

class TestExceptionHandling:

    def test_network_error_returns_ok_false(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection timeout")
            result = _run("Infosys")
        assert result["ok"] is False
        assert "error" in result

    def test_json_parse_error_returns_ok_false(self):
        res = MagicMock()
        res.status_code = 200
        res.raise_for_status = MagicMock()
        res.json.side_effect = ValueError("Invalid JSON")

        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = res
            result = _run("Infosys")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 5. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_key_always_present(self):
        result = _run("")
        assert "ok" in result

    def test_error_result_has_no_required_data_keys(self):
        result = _run("")
        assert result["ok"] is False
        assert "data" not in result or result.get("data") is None

    def test_data_is_dict_on_success(self):
        with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
            mock_get.return_value = _mock_http_200()
            result = _run("Infosys")
        assert isinstance(result.get("data"), dict)


# ---------------------------------------------------------------------------
# 6. Parametrized company names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("company_name", [
    "Infosys",
    "Tata Consultancy Services",
    "Wipro Limited",
    "HCL Technologies",
    "Tech Mahindra",
])
def test_parametrized_company_names_ok(company_name):
    slug = company_name.replace(" ", "_")
    with patch("tools.tool_company_wikipedia.requests.get") as mock_get:
        mock_get.return_value = _mock_http_200(company_name)
        result = _run(company_name)
    assert result["ok"] is True, (
        f"Expected ok=True for company '{company_name}'"
    )


@pytest.mark.parametrize("invalid_input", ["", None])
def test_parametrized_invalid_inputs(invalid_input):
    result = _run(invalid_input)
    assert result["ok"] is False
