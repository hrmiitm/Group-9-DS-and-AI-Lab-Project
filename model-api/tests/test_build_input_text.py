"""
model-api/tests/test_build_input_text.py
Unit tests for the build_input_text() preprocessor in model-api/app.py

We import just the pure function — no model loading occurs because
we monkey-patch the HuggingFace imports before importing app.py.

Tests cover:
  - All fields None → empty string
  - Individual structured fields formatted correctly
  - Individual free-text fields included
  - [SEP] token separator between fields
  - Empty / whitespace fields skipped
  - Binary flags (0 / 1) formatted correctly
  - Combined realistic job posting
  - Parametrized structured fields
  - Parametrized free-text fields
  - Order: structured fields before free-text
"""
import sys
import os
import types
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Patch heavy dependencies before importing app.py
# ---------------------------------------------------------------------------

def _patch_transformers():
    """Create a fake transformers module so app.py can be imported."""
    fake_transformers = types.ModuleType("transformers")
    fake_model = MagicMock()
    fake_model.eval.return_value = None
    fake_model.to.return_value = fake_model

    fake_tokenizer = MagicMock()

    fake_auto_model = MagicMock(return_value=fake_model)
    fake_auto_tokenizer = MagicMock(return_value=fake_tokenizer)

    fake_transformers.AutoModelForSequenceClassification = MagicMock()
    fake_transformers.AutoModelForSequenceClassification.from_pretrained = fake_auto_model
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = fake_auto_tokenizer

    sys.modules["transformers"] = fake_transformers


def _patch_torch():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = MagicMock()
    fake_torch.cuda.is_available = MagicMock(return_value=False)
    fake_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
    sys.modules["torch"] = fake_torch


# Patch before import
_patch_torch()
_patch_transformers()

# Now we can safely import
MODEL_API_DIR = os.path.join(os.path.dirname(__file__), "..")
if MODEL_API_DIR not in sys.path:
    sys.path.insert(0, MODEL_API_DIR)

from app import build_input_text, JobPosting, _make_response, THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(**kwargs) -> JobPosting:
    return JobPosting(**kwargs)


def _build(**kwargs) -> str:
    return build_input_text(_job(**kwargs))


# ---------------------------------------------------------------------------
# 1. Empty job posting
# ---------------------------------------------------------------------------

class TestEmptyJobPosting:

    def test_all_none_returns_empty_string(self):
        text = _build()
        assert text == ""

    def test_all_empty_strings_returns_empty_string(self):
        text = _build(
            title="", description="", requirements="",
            company_profile="", benefits="",
            location="", salary_range="", employment_type="",
        )
        assert text == ""

    def test_whitespace_only_fields_ignored(self):
        text = _build(title="   ", description="  \n  ")
        assert text == ""


# ---------------------------------------------------------------------------
# 2. Individual structured field formatting
# ---------------------------------------------------------------------------

class TestStructuredFields:

    def test_location_label_present(self):
        text = _build(location="Bangalore, India")
        assert "Location: Bangalore, India" in text

    def test_salary_range_label_present(self):
        text = _build(salary_range="80000-100000")
        assert "Salary Range: 80000-100000" in text

    def test_employment_type_label_present(self):
        text = _build(employment_type="Full-time")
        assert "Employment Type: Full-time" in text

    def test_required_experience_label_present(self):
        text = _build(required_experience="Mid-Senior level")
        assert "Required Experience: Mid-Senior level" in text

    def test_required_education_label_present(self):
        text = _build(required_education="Bachelor's Degree")
        assert "Required Education: Bachelor's Degree" in text

    def test_department_label_present(self):
        text = _build(department="Engineering")
        assert "Department: Engineering" in text

    def test_industry_label_present(self):
        text = _build(industry="Information Technology")
        assert "Industry: Information Technology" in text

    def test_function_label_present(self):
        text = _build(function="Software Development")
        assert "Function: Software Development" in text

    def test_has_company_logo_1_label_present(self):
        text = _build(has_company_logo=1)
        assert "Has Company Logo: 1" in text

    def test_has_company_logo_0_label_present(self):
        text = _build(has_company_logo=0)
        assert "Has Company Logo: 0" in text

    def test_telecommuting_1_label_present(self):
        text = _build(telecommuting=1)
        assert "Telecommuting: 1" in text

    def test_has_questions_1_label_present(self):
        text = _build(has_questions=1)
        assert "Has Questions: 1" in text


# ---------------------------------------------------------------------------
# 3. Individual free-text fields
# ---------------------------------------------------------------------------

class TestFreeTextField:

    def test_title_included_without_label(self):
        text = _build(title="Software Engineer")
        assert "Software Engineer" in text

    def test_description_included(self):
        text = _build(description="We are looking for great engineers.")
        assert "We are looking for great engineers." in text

    def test_requirements_included(self):
        text = _build(requirements="5 years Python experience required.")
        assert "5 years Python experience required." in text

    def test_company_profile_included(self):
        text = _build(company_profile="Acme Corp is a global tech company.")
        assert "Acme Corp is a global tech company." in text

    def test_benefits_included(self):
        text = _build(benefits="Health insurance, flexible hours.")
        assert "Health insurance, flexible hours." in text


# ---------------------------------------------------------------------------
# 4. SEP token separator
# ---------------------------------------------------------------------------

class TestSEPTokenSeparator:

    def test_sep_token_used_as_separator(self):
        text = _build(location="Bangalore", title="Engineer")
        assert "[SEP]" in text

    def test_sep_between_structured_and_freetext(self):
        text = _build(location="Mumbai", description="Great opportunity.")
        parts = text.split(" [SEP] ")
        assert len(parts) == 2

    def test_multiple_fields_multiple_sep_tokens(self):
        text = _build(
            location="Delhi",
            employment_type="Full-time",
            title="Developer",
        )
        parts = text.split(" [SEP] ")
        assert len(parts) == 3

    def test_no_sep_when_single_field(self):
        text = _build(title="Engineer")
        assert "[SEP]" not in text

    def test_sep_format_has_spaces(self):
        text = _build(location="City", title="Role")
        assert " [SEP] " in text


# ---------------------------------------------------------------------------
# 5. Structured fields come before free-text fields
# ---------------------------------------------------------------------------

class TestFieldOrdering:

    def test_location_before_title(self):
        text = _build(location="Bangalore", title="Software Engineer")
        loc_pos = text.find("Location:")
        title_pos = text.find("Software Engineer")
        assert loc_pos < title_pos

    def test_salary_before_description(self):
        text = _build(salary_range="50000-80000", description="Some description here.")
        salary_pos = text.find("Salary Range:")
        desc_pos = text.find("Some description here.")
        assert salary_pos < desc_pos

    def test_all_structured_before_all_freetext(self):
        text = _build(
            location="City",
            salary_range="50k",
            title="Engineer",
            description="Do stuff.",
        )
        loc_pos = text.find("Location:")
        desc_pos = text.find("Do stuff.")
        assert loc_pos < desc_pos


# ---------------------------------------------------------------------------
# 6. Combined realistic job posting
# ---------------------------------------------------------------------------

class TestRealisticJobPosting:

    def test_full_legitimate_job_posting(self):
        text = _build(
            title="Senior Python Developer",
            description="We are looking for an experienced Python developer.",
            requirements="5+ years of Python. Strong knowledge of FastAPI.",
            company_profile="Acme Technologies is a 10-year-old IT company.",
            benefits="Health insurance, 25 days annual leave.",
            location="Bangalore, India",
            salary_range="20-30 LPA",
            employment_type="Full-time",
            required_experience="Senior level",
            required_education="Bachelor's in CS",
            department="Engineering",
            industry="Information Technology",
            function="Software Development",
            has_company_logo=1,
            telecommuting=0,
            has_questions=1,
        )
        assert "Location: Bangalore, India" in text
        assert "Senior Python Developer" in text
        assert "Acme Technologies" in text
        assert "[SEP]" in text

    def test_minimal_legitimate_posting(self):
        text = _build(title="Data Analyst", location="Mumbai")
        assert "Data Analyst" in text
        assert "Location: Mumbai" in text

    def test_scam_like_posting(self):
        text = _build(
            title="Work From Home – Earn Daily",
            description="No experience needed. Guaranteed income. Pay registration fee.",
        )
        assert "Work From Home" in text
        assert "Pay registration fee" in text


# ---------------------------------------------------------------------------
# 7. Parametrized structured fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,label,value", [
    ("location",            "Location",            "New York, NY"),
    ("salary_range",        "Salary Range",        "100000-150000"),
    ("employment_type",     "Employment Type",     "Part-time"),
    ("required_experience", "Required Experience", "Entry level"),
    ("required_education",  "Required Education",  "Master's Degree"),
    ("department",          "Department",          "Marketing"),
    ("industry",            "Industry",            "Finance"),
    ("function",            "Function",            "Accounting"),
])
def test_parametrized_structured_fields(field, label, value):
    text = _build(**{field: value})
    assert f"{label}: {value}" in text, (
        f"Expected '{label}: {value}' in text for field '{field}'"
    )


# ---------------------------------------------------------------------------
# 8. _make_response tests
# ---------------------------------------------------------------------------

class TestMakeResponse:

    def test_high_fraud_prob_verdict_fraudulent(self):
        resp = _make_response(0.95, 50.0)
        assert resp.verdict == "FRAUDULENT"

    def test_low_fraud_prob_verdict_legitimate(self):
        resp = _make_response(0.10, 50.0)
        assert resp.verdict == "LEGITIMATE"

    def test_exactly_at_threshold_is_fraudulent(self):
        resp = _make_response(THRESHOLD, 50.0)
        assert resp.verdict == "FRAUDULENT"

    def test_just_below_threshold_is_legitimate(self):
        resp = _make_response(THRESHOLD - 0.001, 50.0)
        assert resp.verdict == "LEGITIMATE"

    def test_high_confidence_band(self):
        # distance > 0.25 from threshold
        resp = _make_response(0.99, 50.0)
        assert resp.confidence == "HIGH"

    def test_medium_confidence_band(self):
        resp = _make_response(THRESHOLD + 0.15, 50.0)
        assert resp.confidence == "MEDIUM"

    def test_low_confidence_band(self):
        resp = _make_response(THRESHOLD + 0.05, 50.0)
        assert resp.confidence == "LOW"

    def test_fraud_probability_rounded(self):
        resp = _make_response(0.123456789, 50.0)
        assert resp.fraud_probability == round(0.123456789, 4)

    def test_fraud_percent_is_probability_times_100(self):
        resp = _make_response(0.75, 50.0)
        assert abs(resp.fraud_percent - 75.0) < 0.2

    def test_latency_ms_returned(self):
        resp = _make_response(0.50, 123.456)
        assert resp.latency_ms == round(123.456, 1)

    def test_model_id_in_response(self):
        resp = _make_response(0.50, 50.0)
        assert isinstance(resp.model_id, str)
        assert len(resp.model_id) > 0

    def test_threshold_in_response(self):
        resp = _make_response(0.50, 50.0)
        assert resp.threshold == THRESHOLD


# ---------------------------------------------------------------------------
# 9. JobPosting schema validation
# ---------------------------------------------------------------------------

class TestJobPostingSchema:

    def test_all_fields_optional(self):
        job = JobPosting()
        assert job.title is None
        assert job.description is None

    def test_has_company_logo_accepts_0(self):
        job = JobPosting(has_company_logo=0)
        assert job.has_company_logo == 0

    def test_has_company_logo_accepts_1(self):
        job = JobPosting(has_company_logo=1)
        assert job.has_company_logo == 1

    def test_telecommuting_accepts_0_and_1(self):
        for val in [0, 1]:
            job = JobPosting(telecommuting=val)
            assert job.telecommuting == val

    def test_all_text_fields_set(self):
        job = JobPosting(
            title="T", description="D", requirements="R",
            company_profile="CP", benefits="B"
        )
        assert job.title == "T"
        assert job.description == "D"
        assert job.requirements == "R"
