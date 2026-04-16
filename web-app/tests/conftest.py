"""
conftest.py
Shared pytest fixtures and helpers for the web-app test suite.

All fixtures are designed to be fast and deterministic — no live network
calls are made here; individual test modules patch where needed.
"""
import sys
import os
import pytest

# ---------------------------------------------------------------------------
# Make the web-app package importable when tests are run from the repo root
# ---------------------------------------------------------------------------
WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "..")
if WEBAPP_DIR not in sys.path:
    sys.path.insert(0, WEBAPP_DIR)


# ---------------------------------------------------------------------------
# Sample job posting fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_job_text():
    """A realistic, legitimate-looking job posting with no scam signals."""
    return (
        "Software Engineer – Backend\n\n"
        "Company: Acme Technologies Pvt. Ltd.\n"
        "Location: Bangalore, India\n"
        "Employment Type: Full-Time\n"
        "Salary: ₹12–18 LPA\n\n"
        "About Us:\n"
        "Acme Technologies is a 12-year-old product company specializing in "
        "enterprise SaaS solutions. We are ISO 27001 certified and listed on NSE.\n\n"
        "Responsibilities:\n"
        "- Design and maintain scalable REST APIs using Python/FastAPI\n"
        "- Work closely with frontend and DevOps teams\n"
        "- Participate in code reviews and architecture discussions\n\n"
        "Requirements:\n"
        "- 2+ years of experience with Python backend development\n"
        "- Strong understanding of SQL and NoSQL databases\n"
        "- Experience with Docker and Kubernetes is a plus\n\n"
        "How to Apply:\n"
        "Send your resume to careers@acmetechnologies.com\n"
        "We will schedule a technical interview within 5 business days."
    )


@pytest.fixture
def scam_job_text_money():
    """Job posting that explicitly asks for upfront money."""
    return (
        "Urgent Work From Home Opportunity!\n\n"
        "Earn daily up to ₹5000 from the comfort of your home!\n"
        "No experience needed. Guaranteed income every week.\n\n"
        "Requirements:\n"
        "- A smartphone or laptop\n"
        "- Pay a registration fee of ₹999 to activate your account\n"
        "- Processing fee of ₹499 for training kit\n\n"
        "Contact us on WhatsApp only: +91 9999999999\n"
        "Limited slots available — apply immediately!"
    )


@pytest.fixture
def scam_job_text_banking():
    """Job posting that requests sensitive banking information."""
    return (
        "Data Entry Operator – Remote\n\n"
        "Work from home, earn ₹800 per hour.\n"
        "To get started, share your bank account number and IFSC code "
        "so we can set up direct salary transfers.\n"
        "Also provide your UPI ID for daily payouts.\n\n"
        "Send details via Western Union or wire transfer confirmation.\n"
        "Immediate joining — only 10 seats remaining!"
    )


@pytest.fixture
def scam_job_text_docs():
    """Job posting that demands identity documents before an interview."""
    return (
        "HR Executive Position – MNC Company Hiring\n\n"
        "US based company hiring locally for work-from-home positions.\n"
        "Salary: ₹40,000/month guaranteed income.\n\n"
        "To proceed, please send:\n"
        "- Aadhaar copy\n"
        "- PAN card copy\n"
        "- Passport copy\n\n"
        "Documents before hiring are mandatory per our policy.\n"
        "Contact on Telegram only: @fakejobposter"
    )


@pytest.fixture
def scam_job_text_high_pressure():
    """Job posting using urgency and high-pressure tactics."""
    return (
        "Marketing Executive – Urgent Hiring\n\n"
        "Last day today! Hurry — only a few openings left.\n"
        "Apply immediately. Interview on the same day.\n"
        "Make money from home — passive income guaranteed.\n\n"
        "Work 2 hours a day and earn lakhs weekly!"
    )


@pytest.fixture
def multi_signal_scam_text(
    scam_job_text_money, scam_job_text_banking, scam_job_text_docs
):
    """A job posting that combines multiple scam signals for a high total score."""
    return "\n\n".join([
        scam_job_text_money,
        scam_job_text_banking,
        scam_job_text_docs,
    ])


# ---------------------------------------------------------------------------
# Sample email fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_corporate_email():
    return "hr@infosys.com"


@pytest.fixture
def valid_role_email():
    """Valid syntax but role-based account prefix."""
    return "careers@microsoft.com"


@pytest.fixture
def disposable_email():
    return "applicant@mailinator.com"


@pytest.fixture
def invalid_email_no_at():
    return "notanemail"


@pytest.fixture
def invalid_email_no_domain():
    return "user@"


@pytest.fixture
def empty_string():
    return ""


# ---------------------------------------------------------------------------
# Sample phone fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_indian_mobile():
    return "+919876543210"


@pytest.fixture
def valid_us_number():
    return "+12125551234"


@pytest.fixture
def invalid_phone_string():
    return "not-a-phone"


@pytest.fixture
def short_phone():
    return "123"


# ---------------------------------------------------------------------------
# Sample domain / URL fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def well_known_domain():
    return "google.com"


@pytest.fixture
def email_for_domain_check():
    return "hr@infosys.com"


@pytest.fixture
def url_for_domain_check():
    return "https://www.infosys.com/careers"


@pytest.fixture
def bare_domain_with_www():
    return "www.tcs.com"


# ---------------------------------------------------------------------------
# Helpers exposed as fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_ok_shape():
    """
    Returns a callable that asserts the standard {ok: True, data: {...}} shape.
    Usage:
        def test_foo(assert_ok_shape):
            result = some_tool("input")
            data = assert_ok_shape(result)
            assert data["field"] == expected
    """
    def _check(result: dict) -> dict:
        assert isinstance(result, dict), "Result must be a dict"
        assert result.get("ok") is True, (
            f"Expected ok=True but got: {result}"
        )
        assert "data" in result, "Result must contain 'data' key"
        assert isinstance(result["data"], dict), "'data' must be a dict"
        return result["data"]
    return _check


@pytest.fixture
def assert_error_shape():
    """
    Returns a callable that asserts the standard {ok: False, error: '...'} shape.
    """
    def _check(result: dict) -> str:
        assert isinstance(result, dict), "Result must be a dict"
        assert result.get("ok") is False, (
            f"Expected ok=False but got: {result}"
        )
        assert "error" in result, "Error result must contain 'error' key"
        assert isinstance(result["error"], str), "'error' must be a string"
        assert len(result["error"]) > 0, "'error' must not be empty"
        return result["error"]
    return _check
