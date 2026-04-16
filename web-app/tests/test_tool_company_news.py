"""
tests/test_tool_company_news.py
Unit tests for tools/tool_company_news.py

Tests cover:
  - Input validation (empty, None)
  - Return structure validation
  - Mocked DDGS news results
  - Article field mapping (date, title, url, source, snippet)
  - max_results parameter respected
  - Empty results handling
  - Exception handling
  - total_articles count accuracy
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_company_news import search_company_news


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(company_name, max_results=8) -> dict:
    return search_company_news(company_name, max_results)


def _make_articles(n: int = 4) -> list:
    return [
        {
            "date":   f"2024-01-{i+1:02d}",
            "title":  f"News about company #{i}",
            "url":    f"https://news.example.com/article-{i}",
            "source": f"NewsSource{i}",
            "body":   f"This is a snippet for article {i}.",
        }
        for i in range(n)
    ]


class _MockDDGS:
    def __init__(self, articles=None, raise_exc=False):
        self._articles = articles
        self._raise = raise_exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def news(self, query, max_results=8):
        if self._raise:
            raise Exception("News fetch failed")
        return self._articles


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
# 2. Successful result (mocked)
# ---------------------------------------------------------------------------

class TestSuccessfulResult:

    def test_ok_true_on_success(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(4))):
            result = _run("Infosys")
        assert result["ok"] is True

    def test_company_name_in_data(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(4))):
            data = _run("Infosys")["data"]
        assert data["company_name"] == "Infosys"

    def test_total_articles_count(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(4))):
            data = _run("Infosys")["data"]
        assert data["total_articles"] == 4

    def test_articles_is_list(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(4))):
            data = _run("Infosys")["data"]
        assert isinstance(data["articles"], list)

    def test_articles_length_matches_total(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(6))):
            data = _run("Infosys")["data"]
        assert len(data["articles"]) == data["total_articles"]


# ---------------------------------------------------------------------------
# 3. Article field mapping
# ---------------------------------------------------------------------------

class TestArticleFieldMapping:

    def test_article_has_date_field(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert "date" in data["articles"][0]

    def test_article_has_title_field(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert "title" in data["articles"][0]

    def test_article_has_url_field(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert "url" in data["articles"][0]

    def test_article_has_source_field(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert "source" in data["articles"][0]

    def test_article_has_snippet_field(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert "snippet" in data["articles"][0]

    def test_article_title_value(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert data["articles"][0]["title"] == "News about company #0"

    def test_article_url_value(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert data["articles"][0]["url"] == "https://news.example.com/article-0"

    def test_article_source_value(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert data["articles"][0]["source"] == "NewsSource0"

    def test_article_snippet_value(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(1))):
            data = _run("Infosys")["data"]
        assert "snippet for article 0" in data["articles"][0]["snippet"]

    def test_body_mapped_to_snippet(self):
        """The raw 'body' field from DDGS should be mapped to 'snippet'."""
        articles = [{"date": "2024-01-01", "title": "Test", "url": "https://x.com",
                     "source": "X", "body": "The body text."}]
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(articles)):
            data = _run("Infosys")["data"]
        assert data["articles"][0]["snippet"] == "The body text."


# ---------------------------------------------------------------------------
# 4. Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:

    def test_none_results_gives_zero_articles(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(None)):
            data = _run("Infosys")["data"]
        assert data["total_articles"] == 0
        assert data["articles"] == []

    def test_empty_list_gives_zero_articles(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS([])):
            data = _run("Infosys")["data"]
        assert data["total_articles"] == 0

    def test_ok_true_even_with_no_articles(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS([])):
            result = _run("Infosys")
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 5. Exception handling
# ---------------------------------------------------------------------------

class TestExceptionHandling:

    def test_ddgs_exception_returns_ok_false(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(raise_exc=True)):
            result = _run("Infosys")
        assert result["ok"] is False

    def test_ddgs_exception_has_error_key(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(raise_exc=True)):
            result = _run("Infosys")
        assert "error" in result

    def test_ddgs_exception_data_has_company_name(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(raise_exc=True)):
            result = _run("Infosys")
        assert result.get("data", {}).get("company_name") == "Infosys"


# ---------------------------------------------------------------------------
# 6. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_key_always_present(self):
        result = _run("")
        assert "ok" in result

    def test_data_has_required_keys(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(2))):
            data = _run("Infosys")["data"]
        required = {"company_name", "total_articles", "articles"}
        assert required.issubset(data.keys())

    def test_total_articles_is_int(self):
        with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(3))):
            data = _run("Infosys")["data"]
        assert isinstance(data["total_articles"], int)


# ---------------------------------------------------------------------------
# 7. Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("company_name", [
    "Infosys", "Wipro", "TCS", "HCL", "Tech Mahindra"
])
def test_parametrized_company_names(company_name):
    with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(3))):
        result = _run(company_name)
    assert result["ok"] is True
    assert result["data"]["company_name"] == company_name


@pytest.mark.parametrize("max_results", [1, 5, 8, 10])
def test_parametrized_max_results_accepted(max_results):
    with patch("tools.tool_company_news.DDGS", return_value=_MockDDGS(_make_articles(max_results))):
        result = _run("Infosys", max_results)
    assert result["ok"] is True
    assert result["data"]["total_articles"] == max_results
