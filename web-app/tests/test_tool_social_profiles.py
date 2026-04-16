"""
tests/test_tool_social_profiles.py
Unit tests for tools/tool_social_profiles.py

Tests cover:
  - Input validation (empty, None)
  - Return structure validation
  - Platform presence detection (mocked DDGS)
  - platforms_found count accuracy
  - Individual platform profile structure
  - Error within a single platform does not crash
  - PLATFORMS constant integrity
  - Parametrized company names
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_social_profiles import check_social_profiles, PLATFORMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(company_name) -> dict:
    return check_social_profiles(company_name)


def _make_search_results(n: int = 2, platform: str = "linkedin") -> list:
    return [
        {
            "href": f"https://{platform}.com/company/acme-{i}",
            "body": f"Snippet for {platform} result {i}",
        }
        for i in range(n)
    ]


class _AllFoundDDGS:
    """Returns 2 results for every platform query."""

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def text(self, query, max_results=3):
        platform = query.split("site:")[1].split(".")[0] if "site:" in query else "example"
        return _make_search_results(2, platform)


class _NoneFoundDDGS:
    """Returns no results for every platform query."""

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def text(self, query, max_results=3):
        return []


class _PartialFoundDDGS:
    """Returns results only for linkedin, nothing for the rest."""

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def text(self, query, max_results=3):
        if "linkedin" in query:
            return _make_search_results(1, "linkedin")
        return []


class _ErrorDDGS:
    """Raises exception for one platform."""

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def text(self, query, max_results=3):
        if "github" in query:
            raise Exception("Rate limited")
        return _make_search_results(1)


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
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            result = _run("Infosys")
        assert result["ok"] is True

    def test_data_has_company_name(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        assert data["company_name"] == "Infosys"

    def test_data_has_platforms_found(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        assert "platforms_found" in data

    def test_data_has_profiles(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        assert "profiles" in data

    def test_profiles_is_dict(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        assert isinstance(data["profiles"], dict)

    def test_platforms_found_is_int(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        assert isinstance(data["platforms_found"], int)

    def test_all_platforms_in_profiles(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform in PLATFORMS.keys():
            assert platform in data["profiles"], f"Missing platform: {platform}"


# ---------------------------------------------------------------------------
# 3. Platform profile structure
# ---------------------------------------------------------------------------

class TestPlatformProfileStructure:

    def test_profile_has_found_key(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            assert "found" in profile, f"Platform '{platform}' missing 'found'"

    def test_profile_has_links_key(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            assert "links" in profile or "error" in profile

    def test_profile_found_is_bool(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            if "found" in profile:
                assert isinstance(profile["found"], bool)

    def test_profile_links_is_list(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            if "links" in profile:
                assert isinstance(profile["links"], list)

    def test_profile_snippets_is_list(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            if "snippets" in profile:
                assert isinstance(profile["snippets"], list)


# ---------------------------------------------------------------------------
# 4. Platforms found count accuracy
# ---------------------------------------------------------------------------

class TestPlatformsFoundCount:

    def test_all_found_max_count(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        assert data["platforms_found"] == len(PLATFORMS)

    def test_none_found_zero_count(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_NoneFoundDDGS()):
            data = _run("Infosys")["data"]
        assert data["platforms_found"] == 0

    def test_partial_found_correct_count(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_PartialFoundDDGS()):
            data = _run("Infosys")["data"]
        assert data["platforms_found"] == 1

    def test_found_flag_true_when_results_exist(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            if "found" in profile:
                assert profile["found"] is True

    def test_found_flag_false_when_no_results(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_NoneFoundDDGS()):
            data = _run("Infosys")["data"]
        for platform, profile in data["profiles"].items():
            if "found" in profile:
                assert profile["found"] is False


# ---------------------------------------------------------------------------
# 5. Error handling within platform
# ---------------------------------------------------------------------------

class TestPlatformErrorHandling:

    def test_error_in_one_platform_does_not_crash(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_ErrorDDGS()):
            result = _run("Infosys")
        assert result["ok"] is True

    def test_errored_platform_has_error_key(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_ErrorDDGS()):
            data = _run("Infosys")["data"]
        assert "error" in data["profiles"]["github"]

    def test_errored_platform_found_is_false(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_ErrorDDGS()):
            data = _run("Infosys")["data"]
        assert data["profiles"]["github"]["found"] is False

    def test_errored_platform_links_is_empty(self):
        with patch("tools.tool_social_profiles.DDGS", return_value=_ErrorDDGS()):
            data = _run("Infosys")["data"]
        assert data["profiles"]["github"]["links"] == []


# ---------------------------------------------------------------------------
# 6. PLATFORMS constant integrity
# ---------------------------------------------------------------------------

class TestPlatformsConstantIntegrity:

    def test_platforms_is_dict(self):
        assert isinstance(PLATFORMS, dict)

    def test_platforms_not_empty(self):
        assert len(PLATFORMS) > 0

    def test_expected_platforms_exist(self):
        expected = {"linkedin", "twitter_x", "github", "facebook", "glassdoor"}
        assert expected.issubset(set(PLATFORMS.keys()))

    def test_all_platform_values_contain_site_filter(self):
        for platform, site_filter in PLATFORMS.items():
            assert "site:" in site_filter, (
                f"Platform '{platform}' missing 'site:' filter"
            )


# ---------------------------------------------------------------------------
# 7. Parametrized company names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("company_name", [
    "Infosys",
    "Wipro",
    "Google",
    "Microsoft",
    "Amazon",
])
def test_parametrized_company_names_ok(company_name):
    with patch("tools.tool_social_profiles.DDGS", return_value=_AllFoundDDGS()):
        result = _run(company_name)
    assert result["ok"] is True
    assert result["data"]["company_name"] == company_name
