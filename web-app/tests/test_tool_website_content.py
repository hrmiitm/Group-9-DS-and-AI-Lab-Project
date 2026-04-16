"""
tests/test_tool_website_content.py
Unit tests for tools/tool_website_content.py

Tests cover:
  - Input validation (empty, None, no URL)
  - URL normalisation (https:// prefix)
  - Successful content extraction (mocked trafilatura)
  - Fetch failure handling
  - Empty content handling
  - Word count calculation
  - Metadata extraction structure
  - Return structure validation
  - Parametrized URL inputs
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_website_content import extract_website_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(url) -> dict:
    return extract_website_content(url)


SAMPLE_HTML = """
<html>
<head><title>Acme Corp – Careers</title></head>
<body>
<h1>Join Acme Corp</h1>
<p>We are looking for talented engineers to join our team.
Apply today and become part of a world-class engineering culture.
We offer competitive salaries, flexible work, and great benefits.</p>
</body>
</html>
"""

SAMPLE_TEXT = (
    "Join Acme Corp. We are looking for talented engineers to join our team. "
    "Apply today and become part of a world-class engineering culture. "
    "We offer competitive salaries, flexible work, and great benefits."
)


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
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            result = _run("example.com")
        assert result["ok"] is True
        assert result["data"]["url"].startswith("https://")

    def test_https_url_unchanged(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            result = _run("https://example.com")
        assert result["data"]["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# 3. Successful extraction (mocked)
# ---------------------------------------------------------------------------

class TestSuccessfulExtraction:

    def _make_mock_meta(self):
        meta = MagicMock()
        meta.title = "Acme Corp – Careers"
        meta.description = "Careers at Acme Corp"
        meta.author = "Acme Corp"
        meta.sitename = "acmecorp.com"
        meta.date = "2024-01-15"
        meta.language = "en"
        meta.categories = ["Jobs", "Careers"]
        meta.tags = ["engineering", "hiring"]
        return meta

    def test_ok_true_on_success(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = self._make_mock_meta()
            result = _run("https://example.com")
        assert result["ok"] is True

    def test_extracted_text_returned(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert data["extracted_text"] == SAMPLE_TEXT

    def test_word_count_calculated(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        expected_words = len(SAMPLE_TEXT.split())
        assert data["word_count"] == expected_words

    def test_word_count_is_int(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert isinstance(data["word_count"], int)

    def test_metadata_dict_populated(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = self._make_mock_meta()
            data = _run("https://example.com")["data"]
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["title"] == "Acme Corp – Careers"

    def test_metadata_empty_when_none(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert data["metadata"] == {}

    def test_metadata_has_expected_keys(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = self._make_mock_meta()
            data = _run("https://example.com")["data"]
        expected_keys = {"title", "description", "author", "sitename", "date", "language"}
        assert expected_keys.issubset(data["metadata"].keys())


# ---------------------------------------------------------------------------
# 4. Fetch failure
# ---------------------------------------------------------------------------

class TestFetchFailure:

    def test_fetch_url_returns_none_gives_error(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch:
            mock_fetch.return_value = None
            result = _run("https://unreachable.example.com")
        assert result["ok"] is False
        assert "error" in result

    def test_fetch_url_returns_none_includes_url_in_data(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch:
            mock_fetch.return_value = None
            result = _run("https://unreachable.example.com")
        assert result.get("data", {}).get("url") == "https://unreachable.example.com"

    def test_exception_in_fetch_returns_ok_false(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            result = _run("https://example.com")
        assert result["ok"] is False

    def test_exception_message_in_error(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch:
            mock_fetch.side_effect = Exception("Timeout reached")
            result = _run("https://example.com")
        assert "Timeout reached" in result["error"]


# ---------------------------------------------------------------------------
# 5. Empty / None extracted text
# ---------------------------------------------------------------------------

class TestEmptyContent:

    def test_none_extracted_text_gives_word_count_zero(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = None
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert data["word_count"] == 0

    def test_empty_extracted_text_gives_word_count_zero(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = ""
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert data["word_count"] == 0

    def test_none_text_ok_still_true(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = None
            mock_meta.return_value = None
            result = _run("https://example.com")
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 6. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_key_always_present(self):
        result = _run("")
        assert "ok" in result

    def test_data_has_required_keys(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        required = {"url", "extracted_text", "word_count", "metadata"}
        assert required.issubset(data.keys())

    def test_url_in_data_matches_input(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert data["url"] == "https://example.com"

    def test_metadata_is_dict(self):
        with patch("tools.tool_website_content.fetch_url") as mock_fetch, \
             patch("tools.tool_website_content.extract") as mock_extract, \
             patch("tools.tool_website_content.extract_metadata") as mock_meta:
            mock_fetch.return_value = SAMPLE_HTML
            mock_extract.return_value = SAMPLE_TEXT
            mock_meta.return_value = None
            data = _run("https://example.com")["data"]
        assert isinstance(data["metadata"], dict)
