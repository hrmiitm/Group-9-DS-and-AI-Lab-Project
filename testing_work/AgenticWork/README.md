# AgentTesting – Job Posting Feature Extraction Agent

This module contains a **LangChain ReAct agent** that can read a job-posting
document in any common format and extract all 18 structured features defined
in the Milestone-2 dataset schema.

## Features extracted

| Group | Column | Description |
|-------|--------|-------------|
| Free-text | `title` | Job title as posted |
| Free-text | `description` | Full job description |
| Free-text | `requirements` | Required qualifications & skills |
| Free-text | `company_profile` | Hiring-company description |
| Free-text | `benefits` | Offered benefits / perks |
| Structured | `location` | City, State, Country |
| Structured | `department` | Organisational department |
| Structured | `salary_range` | Salary band (e.g. "50000-70000") |
| Structured | `employment_type` | Full-time / Part-time / Contract / Other |
| Structured | `required_experience` | Entry level / Mid-Senior / Director … |
| Structured | `required_education` | Bachelor's / Master's … |
| Structured | `industry` | Industry classification |
| Structured | `function` | Job function / category |
| Binary | `has_company_logo` | 1 if logo present, else 0 |
| Binary | `telecommuting` | 1 if remote work offered, else 0 |
| Binary | `has_questions` | 1 if screening questions present, else 0 |

## Supported input formats

`.docx` · `.doc` · `.pdf` · `.html` · `.htm` · `.md` · `.txt`

## Installation

```bash
pip install langchain langchain-openai langchain-community docx2txt pypdf unstructured
```

> **Note:** Set your OpenAI API key before running:
> ```bash
> export OPENAI_API_KEY="sk-..."
> ```

## Usage

```bash
python job_parser_agent.py path/to/job_description.pdf
python job_parser_agent.py path/to/job_description.docx
python job_parser_agent.py path/to/job_description.md
```

The agent prints a JSON object with all extracted features.

## Architecture

```
User supplies file path
        │
        ▼
  ReAct Agent  (LangGraph create_react_agent)
        │  decides to call ↓
        ▼
  extract_job_posting_features(@tool)
        │  1. Loads raw text via LangChain document loaders
        │  2. Sends text + structured prompt to GPT-4o-mini
        │  3. Returns validated JSON
        ▼
  Agent formats & prints result
```
