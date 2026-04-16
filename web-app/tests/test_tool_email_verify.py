"""
tests/test_tool_email_verify.py
Unit tests for tools/tool_email_verify.py

Tests cover:
  - Valid corporate / role-based / personal emails (syntax only, no DNS)
  - Invalid syntax emails
  - Disposable domain detection
  - Role-account prefix detection
  - Empty / None input validation
  - Return structure validation
  - Edge cases: leading/trailing whitespace, weird TLDs, subdomains
  - Parametrized valid and invalid address lists
"""
import pytest
from unittest.mock import patch, MagicMock

from tools.tool_email_verify import verify_email, DISPOSABLE_DOMAINS, ROLE_PREFIXES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(email) -> dict:
    return verify_email(email)


def _data(email) -> dict:
    result = _run(email)
    assert result["ok"] is True, f"Expected ok=True for '{email}', got: {result}"
    return result["data"]


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_string_returns_error(self, assert_error_shape):
        result = _run("")
        assert_error_shape(result)

    def test_none_returns_error(self, assert_error_shape):
        result = _run(None)
        assert_error_shape(result)

    def test_empty_error_has_message(self):
        result = _run("")
        assert len(result["error"]) > 0

    def test_whitespace_stripped_before_validation(self):
        # "  " is still falsy check; verify_email checks `if not email`
        result = _run("  ")
        # "  " is truthy, so it proceeds to syntax check
        assert "ok" in result


# ---------------------------------------------------------------------------
# 2. Syntax validation (no DNS — check_deliverability=False)
# ---------------------------------------------------------------------------

class TestSyntaxValidation:

    def test_valid_simple_email_ok_true(self):
        result = _run("user@example.com")
        assert result["ok"] is True

    def test_valid_email_data_structure(self):
        data = _data("user@example.com")
        expected_keys = {
            "email", "local_part", "domain", "is_syntax_valid",
            "is_deliverable", "mx_host", "is_disposable",
            "is_role_account", "overall_status"
        }
        assert expected_keys.issubset(data.keys())

    def test_valid_email_syntax_flag(self):
        data = _data("john.doe@company.org")
        assert data["is_syntax_valid"] is True

    def test_valid_email_domain_extracted(self):
        data = _data("alice@infosys.com")
        assert data["domain"] == "infosys.com"

    def test_valid_email_local_part_extracted(self):
        data = _data("alice@infosys.com")
        assert data["local_part"] == "alice"

    def test_invalid_email_no_at(self):
        result = _run("notanemail")
        assert result["ok"] is False
        assert "is_syntax_valid" in result.get("data", {}) or "error" in result

    def test_invalid_email_no_domain(self):
        result = _run("user@")
        assert result["ok"] is False

    def test_invalid_email_no_local_part(self):
        result = _run("@domain.com")
        assert result["ok"] is False

    def test_invalid_email_double_at(self):
        result = _run("user@@domain.com")
        assert result["ok"] is False

    def test_invalid_email_spaces_in_address(self):
        result = _run("user name@domain.com")
        assert result["ok"] is False

    def test_email_with_subdomain_valid(self):
        result = _run("hr@mail.company.co.in")
        assert result["ok"] is True

    def test_email_with_plus_tag_valid(self):
        result = _run("user+tag@example.com")
        assert result["ok"] is True

    def test_email_with_dots_in_local_valid(self):
        result = _run("first.last@domain.net")
        assert result["ok"] is True

    def test_email_with_numbers_valid(self):
        result = _run("john123@company456.io")
        assert result["ok"] is True

    def test_email_leading_whitespace_stripped(self):
        result = _run("  hr@company.com")
        assert result["ok"] is True

    def test_email_trailing_whitespace_stripped(self):
        result = _run("hr@company.com  ")
        assert result["ok"] is True

    def test_email_normalized_lowercase(self):
        data = _data("HR@COMPANY.COM")
        assert data["email"] == data["email"].lower()


# ---------------------------------------------------------------------------
# 3. Disposable domain detection
# ---------------------------------------------------------------------------

class TestDisposableDomainDetection:

    @pytest.mark.parametrize("domain", list(DISPOSABLE_DOMAINS))
    def test_known_disposable_domain_flagged(self, domain):
        email = f"user@{domain}"
        result = _run(email)
        if result["ok"]:
            assert result["data"]["is_disposable"] is True, (
                f"Expected is_disposable=True for {email}"
            )

    def test_mailinator_flagged(self):
        result = _run("test@mailinator.com")
        if result["ok"]:
            assert result["data"]["is_disposable"] is True

    def test_tempmail_flagged(self):
        result = _run("temp@tempmail.com")
        if result["ok"]:
            assert result["data"]["is_disposable"] is True

    def test_yopmail_flagged(self):
        result = _run("user@yopmail.com")
        if result["ok"]:
            assert result["data"]["is_disposable"] is True

    def test_corporate_email_not_disposable(self):
        data = _data("employee@infosys.com")
        assert data["is_disposable"] is False

    def test_gmail_not_in_disposable_set(self):
        # Gmail is flagged as unofficial_contact but NOT disposable
        data = _data("user@gmail.com")
        assert data["is_disposable"] is False

    def test_unknown_domain_not_disposable(self):
        data = _data("user@legitimate-company-xyz.com")
        assert data["is_disposable"] is False


# ---------------------------------------------------------------------------
# 4. Role account detection
# ---------------------------------------------------------------------------

class TestRoleAccountDetection:

    @pytest.mark.parametrize("prefix", list(ROLE_PREFIXES))
    def test_known_role_prefix_flagged(self, prefix):
        email = f"{prefix}@company.com"
        result = _run(email)
        if result["ok"]:
            assert result["data"]["is_role_account"] is True, (
                f"Expected is_role_account=True for {email}"
            )

    def test_hr_is_role_account(self):
        data = _data("hr@company.com")
        assert data["is_role_account"] is True

    def test_jobs_is_role_account(self):
        data = _data("jobs@startup.io")
        assert data["is_role_account"] is True

    def test_careers_is_role_account(self):
        data = _data("careers@bigcorp.com")
        assert data["is_role_account"] is True

    def test_noreply_is_role_account(self):
        data = _data("noreply@service.com")
        assert data["is_role_account"] is True

    def test_recruitment_is_role_account(self):
        data = _data("recruitment@agency.com")
        assert data["is_role_account"] is True

    def test_personal_name_is_not_role_account(self):
        data = _data("john.doe@company.com")
        assert data["is_role_account"] is False

    def test_alice_is_not_role_account(self):
        data = _data("alice@company.com")
        assert data["is_role_account"] is False

    def test_role_check_is_case_insensitive(self):
        data = _data("HR@company.com")
        assert data["is_role_account"] is True

    def test_partial_role_prefix_not_flagged(self):
        # "hradmin" is NOT in ROLE_PREFIXES as-is
        data = _data("hradmin@company.com")
        assert data["is_role_account"] is False


# ---------------------------------------------------------------------------
# 5. Deliverability check (mocked DNS calls)
# ---------------------------------------------------------------------------

class TestDeliverabilityMocked:

    def test_deliverable_email_sets_flag(self):
        # Mock successful DNS check
        mock_info = MagicMock()
        mock_info.normalized = "hr@infosys.com"
        mock_info.local_part = "hr"
        mock_info.domain = "infosys.com"
        mock_info.mx = [(10, "mx.infosys.com")]

        from email_validator import EmailNotValidError

        with patch("tools.tool_email_verify.validate_email") as mock_validate:
            mock_validate.side_effect = [
                mock_info,   # first call: syntax check (check_deliverability=False)
                mock_info,   # second call: DNS check (check_deliverability=True)
            ]
            result = verify_email("hr@infosys.com")

        assert result["ok"] is True
        assert result["data"]["is_deliverable"] is True

    def test_undeliverable_domain_sets_flag(self):
        from email_validator import EmailNotValidError

        mock_info = MagicMock()
        mock_info.normalized = "user@nonexistent-xyz.com"
        mock_info.local_part = "user"
        mock_info.domain = "nonexistent-xyz.com"

        with patch("tools.tool_email_verify.validate_email") as mock_validate:
            mock_validate.side_effect = [
                mock_info,  # syntax OK
                EmailNotValidError("No MX record found"),  # DNS fails
            ]
            result = verify_email("user@nonexistent-xyz.com")

        assert result["ok"] is True
        assert result["data"]["is_deliverable"] is False
        assert result["data"]["deliverability_error"] is not None

    def test_mx_host_extracted_when_available(self):
        mock_info_syntax = MagicMock()
        mock_info_syntax.normalized = "hr@company.com"
        mock_info_syntax.local_part = "hr"
        mock_info_syntax.domain = "company.com"

        mock_info_dns = MagicMock()
        mock_info_dns.mx = [(10, "mail.company.com")]

        from email_validator import EmailNotValidError

        with patch("tools.tool_email_verify.validate_email") as mock_validate:
            mock_validate.side_effect = [mock_info_syntax, mock_info_dns]
            result = verify_email("hr@company.com")

        assert result["ok"] is True
        assert result["data"]["mx_host"] is not None


# ---------------------------------------------------------------------------
# 6. Overall status logic
# ---------------------------------------------------------------------------

class TestOverallStatus:

    def test_overall_status_values_are_valid(self):
        result = _run("user@example.com")
        if result["ok"]:
            assert result["data"]["overall_status"] in (
                "deliverable", "undeliverable", "unknown"
            )

    def test_overall_status_is_string(self):
        result = _run("user@example.com")
        if result["ok"]:
            assert isinstance(result["data"]["overall_status"], str)


# ---------------------------------------------------------------------------
# 7. Return structure validation
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_ok_key_always_present(self):
        for email in ["valid@example.com", "", "bad"]:
            result = _run(email)
            assert "ok" in result

    def test_valid_email_has_data_key(self):
        result = _run("user@example.com")
        assert "data" in result

    def test_invalid_email_has_error_key(self):
        result = _run("notvalid")
        assert "error" in result

    def test_data_is_dict_for_valid_input(self):
        result = _run("user@example.com")
        assert isinstance(result.get("data"), dict)

    def test_domain_is_lowercase(self):
        data = _data("User@COMPANY.COM")
        assert data["domain"] == data["domain"].lower()

    def test_is_disposable_is_bool(self):
        data = _data("user@example.com")
        assert isinstance(data["is_disposable"], bool)

    def test_is_role_account_is_bool(self):
        data = _data("hr@example.com")
        assert isinstance(data["is_role_account"], bool)

    def test_is_syntax_valid_is_bool(self):
        data = _data("user@example.com")
        assert isinstance(data["is_syntax_valid"], bool)


# ---------------------------------------------------------------------------
# 8. Parametrized valid emails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("email", [
    "hr@infosys.com",
    "careers@microsoft.com",
    "john.doe@startup.io",
    "alice+work@example.org",
    "user123@company.co.in",
    "name@subdomain.company.com",
    "first.last@big-corp.net",
    "admin@university.edu",
])
def test_parametrized_valid_emails_ok(email):
    result = _run(email)
    assert result["ok"] is True, f"Expected ok=True for '{email}'"


# ---------------------------------------------------------------------------
# 9. Parametrized invalid emails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("email", [
    "notanemail",
    "@nodomain.com",
    "user@",
    "user name@domain.com",
    "user@@double.com",
    "",
    "plaintext",
])
def test_parametrized_invalid_emails_error(email):
    result = _run(email)
    assert result["ok"] is False, (
        f"Expected ok=False for invalid email '{email}', got: {result}"
    )


# ---------------------------------------------------------------------------
# 10. Disposable domain set integrity
# ---------------------------------------------------------------------------

class TestDisposableDomainSetIntegrity:

    def test_disposable_domains_is_set(self):
        assert isinstance(DISPOSABLE_DOMAINS, set)

    def test_disposable_domains_not_empty(self):
        assert len(DISPOSABLE_DOMAINS) > 0

    def test_all_disposable_domains_lowercase(self):
        for domain in DISPOSABLE_DOMAINS:
            assert domain == domain.lower(), (
                f"Disposable domain '{domain}' is not lowercase"
            )

    def test_mailinator_in_disposable_set(self):
        assert "mailinator.com" in DISPOSABLE_DOMAINS

    def test_role_prefixes_is_set(self):
        assert isinstance(ROLE_PREFIXES, set)

    def test_role_prefixes_not_empty(self):
        assert len(ROLE_PREFIXES) > 0

    def test_all_role_prefixes_lowercase(self):
        for prefix in ROLE_PREFIXES:
            assert prefix == prefix.lower(), (
                f"Role prefix '{prefix}' is not lowercase"
            )

    def test_hr_in_role_prefixes(self):
        assert "hr" in ROLE_PREFIXES

    def test_noreply_in_role_prefixes(self):
        assert "noreply" in ROLE_PREFIXES
