"""
core/job_parser_agent.py
JobPosting Pydantic schema + document loader.
Copied from src/job_analyzer/job_parser_agent.py and trimmed to webapp essentials:
  - JobPosting  (Pydantic model with 16 fields)
  - load_document(file_path) -> str
No sys.path hacks. No external imports. Self-contained.
"""

from __future__ import annotations

import pathlib
from typing import Literal, Optional

from pydantic import BaseModel, Field

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)


# ── Pydantic schema ───────────────────────────────────────────────────────────

class JobPosting(BaseModel):
    """
    Structured extraction of a job posting.
    Mirrors the Kaggle 'Real or Fake Job Postings' dataset schema.
    Optional fields are None when not present in the source text.
    """

    # Group A: Free-text fields
    title: str = Field(
        description="Job title exactly as posted, e.g. 'Senior Software Engineer'."
    )
    description: Optional[str] = Field(
        default=None,
        description="Full job description / responsibilities / role overview."
    )
    requirements: Optional[str] = Field(
        default=None,
        description="Required qualifications, skills, certifications and experience."
    )
    company_profile: Optional[str] = Field(
        default=None,
        description="'About the company' section or description of the hiring organisation."
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Official company/employer name as written in the posting."
    )
    company_website: Optional[str] = Field(
        default=None,
        description="Official company website URL if present (prefer full URL)."
    )
    contact_email: Optional[str] = Field(
        default=None,
        description="Recruiter or hiring contact email address if explicitly present."
    )
    contact_phone: Optional[str] = Field(
        default=None,
        description="Recruiter or hiring contact phone number if explicitly present."
    )
    benefits: Optional[str] = Field(
        default=None,
        description="Benefits, perks and compensation extras offered with the role."
    )

    # Group B: Structured metadata
    location: Optional[str] = Field(
        default=None,
        description="Job location as a string, e.g. 'San Francisco, CA, US'."
    )
    department: Optional[str] = Field(
        default=None,
        description="Organisational department or team, e.g. 'Engineering', 'Finance'."
    )
    salary_range: Optional[str] = Field(
        default=None,
        description="Salary band as a plain string, e.g. '50000-70000' or '$80k-$100k'."
    )
    employment_type: Optional[Literal[
        "Full-time", "Part-time", "Contract", "Temporary", "Other"
    ]] = Field(
        default=None,
        description="Employment type. Set to null if not stated."
    )
    required_experience: Optional[Literal[
        "Entry level", "Mid-Senior level", "Associate",
        "Director", "Executive", "Internship", "Not Applicable"
    ]] = Field(
        default=None,
        description="Required experience level. Set to null if not stated."
    )
    required_education: Optional[Literal[
        "Bachelor's Degree", "Master's Degree", "Doctorate",
        "High School or equivalent", "Associate Degree",
        "Some College Coursework Completed", "Professional",
        "Certification", "Vocational", "Some High School Coursework",
        "Unspecified"
    ]] = Field(
        default=None,
        description="Minimum education requirement. Set to null if not stated."
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry classification, e.g. 'Information Technology and Services'."
    )
    function: Optional[str] = Field(
        default=None,
        description="Job function / category, e.g. 'Engineering', 'Sales', 'Marketing'."
    )

    # Group C: Binary flags
    has_company_logo: Optional[Literal[0, 1]] = Field(
        default=None,
        description="1 if a company logo is present, 0 if absent, null if unknown."
    )
    telecommuting: Optional[Literal[0, 1]] = Field(
        default=None,
        description="1 if remote/work-from-home is offered, 0 if office-only, null if not mentioned."
    )
    has_questions: Optional[Literal[0, 1]] = Field(
        default=None,
        description="1 if screening questions are included, 0 if not, null if unknown."
    )


# ── Document loader ───────────────────────────────────────────────────────────

_LOADER_MAP: dict[str, type] = {
    ".docx": Docx2txtLoader,
    ".doc":  Docx2txtLoader,
    ".pdf":  PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm":  UnstructuredHTMLLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".txt":  TextLoader,
}


def load_document(file_path: str) -> str:
    """
    Return the plain-text content of a document file.
    Picks the right LangChain loader by file extension; falls back to TextLoader.
    Supported: .pdf, .docx, .doc, .html, .htm, .md, .txt
    """
    suffix = pathlib.Path(file_path).suffix.lower()
    loader_cls = _LOADER_MAP.get(suffix, TextLoader)
    docs = loader_cls(file_path).load()
    return "\n\n".join(doc.page_content for doc in docs)
