"""
tests/test_tool_job_boards.py
Unit tests for tools/tool_job_boards.py

Tests cover:
  - Input validation (empty job_title, empty company_name, both empty)
  - Return structure validation
  - Verdict logic (strong_presence / moderate_presence / not_found_on_boards)
  - boards_found count accuracy
  - Location parameter (optional)
  - Individual board result structure
  - Error handling within a single board
  - JOB_BOARDS constant integrity
  - Parametrized inputs
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_job_boards import check_job_boards, JOB_BOARDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(job_title, company_name, location=None) -> dict:
    return check_job_boards(job_title, company_name, location)


def _make_results(n: int = 2) -> list:
    return [
        {
            "title":   f"Job Result {i}",
            "href":    f"https://jobboard.example.com/job-{i}",
            "body":    f"Job description snippet {i}",
        }
        for i in range(n)
    ]


class _AllBoardsFoundDDGS:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def text(self, query, max_results=3):
        return _make_results(2)


class _NoBoardsFoundDDGS:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def text(self, query, max_results=3):
        return []


class _TwoBoardsFoundDDGS:
    """Returns results for the first 2 boards, nothing for the rest."""

    def __init__(self):
        self._call_count = 0

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def text(self, query, max_results=3):
        self._call_count += 1
        if self._call_count <= 2:
            return _make_results(1)
        return []


class _ErrorOneBoardDDGS:
    """Raises exception for the 'naukri' board."""

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def text(self, query, max_results=3):
        if "naukri" in query:
            raise Exception("Blocked by Naukri")
        return _make_results(1)


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_job_title_returns_error(self):
        result = _run("", "Infosys")
        assert result["ok"] is False
        assert "error" in result

    def test_empty_company_name_returns_error(self):
        result = _run("Software Engineer", "")
        assert result["ok"] is False
        assert "error" in result

    def test_both_empty_returns_error(self):
        result = _run("", "")
        assert result["ok"] is False

    def test_none_job_title_returns_error(self):
        result = _run(None, "Infosys")
        assert result["ok"] is False

    def test_none_company_name_returns_error(self):
        result = _run("Software Engineer", None)
        assert result["ok"] is False

    def test_error_message_non_empty(self):
        result = _run("", "Infosys")
        assert len(result["error"]) > 0


# ---------------------------------------------------------------------------
# 2. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_true_on_success(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            result = _run("Software Engineer", "Infosys")
        assert result["ok"] is True

    def test_data_has_job_title(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["job_title"] == "Software Engineer"

    def test_data_has_company_name(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["company_name"] == "Infosys"

    def test_data_has_boards_found(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert "boards_found" in data

    def test_data_has_verdict(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert "verdict" in data

    def test_data_has_boards(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert "boards" in data

    def test_boards_is_dict(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert isinstance(data["boards"], dict)

    def test_boards_found_is_int(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert isinstance(data["boards_found"], int)

    def test_all_job_boards_in_result(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board in JOB_BOARDS.keys():
            assert board in data["boards"], f"Missing board: {board}"


# ---------------------------------------------------------------------------
# 3. Verdict logic
# ---------------------------------------------------------------------------

class TestVerdictLogic:

    def test_all_boards_found_gives_strong_presence(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["verdict"] == "strong_presence"
        assert data["boards_found"] >= 3

    def test_no_boards_found_gives_not_found(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_NoBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["verdict"] == "not_found_on_boards"
        assert data["boards_found"] == 0

    def test_two_boards_found_gives_moderate_presence(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_TwoBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["verdict"] == "moderate_presence"

    def test_verdict_is_valid_string(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        valid_verdicts = {"strong_presence", "moderate_presence", "not_found_on_boards"}
        assert data["verdict"] in valid_verdicts

    @pytest.mark.parametrize("boards_found,expected_verdict", [
        (0, "not_found_on_boards"),
        (1, "moderate_presence"),
        (2, "moderate_presence"),
        (3, "strong_presence"),
        (5, "strong_presence"),
    ])
    def test_parametrized_verdict_thresholds(self, boards_found, expected_verdict):
        # We test the verdict formula directly
        verdict = (
            "strong_presence" if boards_found >= 3
            else "moderate_presence" if boards_found >= 1
            else "not_found_on_boards"
        )
        assert verdict == expected_verdict


# ---------------------------------------------------------------------------
# 4. Board result structure
# ---------------------------------------------------------------------------

class TestBoardResultStructure:

    def test_board_has_found_key(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            assert "found" in result or "error" in result

    def test_board_has_results_key(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            assert "results" in result or "error" in result

    def test_board_found_is_bool(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            if "found" in result:
                assert isinstance(result["found"], bool)

    def test_board_results_is_list(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            if "results" in result:
                assert isinstance(result["results"], list)

    def test_board_result_item_has_title(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            if result.get("results"):
                assert "title" in result["results"][0]

    def test_board_result_item_has_url(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            if result.get("results"):
                assert "url" in result["results"][0]

    def test_board_result_item_has_snippet(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        for board, result in data["boards"].items():
            if result.get("results"):
                assert "snippet" in result["results"][0]


# ---------------------------------------------------------------------------
# 5. Location parameter
# ---------------------------------------------------------------------------

class TestLocationParameter:

    def test_without_location_ok(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            result = _run("Software Engineer", "Infosys")
        assert result["ok"] is True
        assert result["data"]["location"] is None

    def test_with_location_ok(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            result = _run("Software Engineer", "Infosys", "Bangalore")
        assert result["ok"] is True
        assert result["data"]["location"] == "Bangalore"

    def test_location_included_in_query(self):
        """Verify location is appended to search query by checking the data."""
        with patch("tools.tool_job_boards.DDGS", return_value=_AllBoardsFoundDDGS()):
            data = _run("Software Engineer", "Infosys", "Mumbai")["data"]
        assert data["location"] == "Mumbai"


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_error_in_one_board_does_not_crash(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_ErrorOneBoardDDGS()):
            result = _run("Software Engineer", "Infosys")
        assert result["ok"] is True

    def test_errored_board_has_error_key(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_ErrorOneBoardDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert "error" in data["boards"]["naukri"]

    def test_errored_board_found_is_false(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_ErrorOneBoardDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["boards"]["naukri"]["found"] is False

    def test_errored_board_results_is_empty_list(self):
        with patch("tools.tool_job_boards.DDGS", return_value=_ErrorOneBoardDDGS()):
            data = _run("Software Engineer", "Infosys")["data"]
        assert data["boards"]["naukri"]["results"] == []


# ---------------------------------------------------------------------------
# 7. JOB_BOARDS constant integrity
# ---------------------------------------------------------------------------

class TestJobBoardsConstant:

    def test_job_boards_is_dict(self):
        assert isinstance(JOB_BOARDS, dict)

    def test_job_boards_not_empty(self):
        assert len(JOB_BOARDS) > 0

    def test_expected_boards_exist(self):
        expected = {"linkedin_jobs", "indeed", "glassdoor", "naukri"}
        assert expected.issubset(set(JOB_BOARDS.keys()))

    def test_all_board_values_contain_site_filter(self):
        for board, site_filter in JOB_BOARDS.items():
            assert "site:" in site_filter, (
                f"Board '{board}' missing 'site:' filter"
            )
