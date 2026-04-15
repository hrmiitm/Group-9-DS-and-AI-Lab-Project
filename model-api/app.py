"""
model-api/app.py

FastAPI service wrapping the fine-tuned RoBERTa fraud classifier.
Designed to run on HuggingFace Spaces (Docker SDK).

Endpoints:
    GET  /              → health check
    POST /predict       → single job posting → fraud probability + verdict
    POST /predict/batch → list of job postings → list of results
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID    = os.getenv("MODEL_ID",    "aditya963/fraud-job-classifier")
THRESHOLD   = float(os.getenv("FRAUD_THRESHOLD", "0.87"))
MAX_LENGTH  = int(os.getenv("MAX_LENGTH", "512"))
MAX_BATCH   = int(os.getenv("MAX_BATCH_SIZE", "16"))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudGuard — RoBERTa Fraud Classifier API",
    description=(
        "Fine-tuned RoBERTa-base model for detecting fraudulent job postings. "
        "Trained on the EMSCAD dataset (17,880 samples, 4.84% fraud). "
        "F1=0.907 | Precision=0.957 | ROC-AUC=0.993 at threshold=0.87."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Model loading (once at startup) ──────────────────────────────────────────

print(f"[startup] Loading model '{MODEL_ID}' on {DEVICE} ...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
_model.to(DEVICE)
_model.eval()
print(f"[startup] Model ready. Threshold={THRESHOLD}, device={DEVICE}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class JobPosting(BaseModel):
    """Input schema — mirrors the EMSCAD / RoBERTa training schema."""

    # Free-text fields
    title:           Optional[str] = Field(None, example="Software Engineer")
    description:     Optional[str] = Field(None, example="We are looking for...")
    requirements:    Optional[str] = Field(None, example="5 years Python experience")
    company_profile: Optional[str] = Field(None, example="A leading tech company...")
    benefits:        Optional[str] = Field(None, example="Health insurance, WFH")

    # Structured metadata
    location:             Optional[str] = Field(None, example="New York, NY, US")
    salary_range:         Optional[str] = Field(None, example="80000-100000")
    employment_type:      Optional[str] = Field(None, example="Full-time")
    required_experience:  Optional[str] = Field(None, example="Mid-Senior level")
    required_education:   Optional[str] = Field(None, example="Bachelor's Degree")
    department:           Optional[str] = Field(None, example="Engineering")
    industry:             Optional[str] = Field(None, example="Information Technology")
    function:             Optional[str] = Field(None, example="Engineering")

    # Binary / categorical
    has_company_logo: Optional[int] = Field(None, example=1, ge=0, le=1)
    telecommuting:    Optional[int] = Field(None, example=0, ge=0, le=1)
    has_questions:    Optional[int] = Field(None, example=1, ge=0, le=1)


class PredictResponse(BaseModel):
    fraud_probability: float = Field(..., example=0.9247)
    fraud_percent:     float = Field(..., example=92.5)
    verdict:           str   = Field(..., example="FRAUDULENT")
    confidence:        str   = Field(..., example="HIGH")
    threshold:         float = Field(..., example=0.87)
    model_id:          str   = Field(..., example="aditya963/fraud-job-classifier")
    latency_ms:        float = Field(..., example=43.2)


class BatchRequest(BaseModel):
    postings: list[JobPosting] = Field(..., max_length=16)


class BatchResponse(BaseModel):
    results:    list[PredictResponse]
    count:      int
    latency_ms: float


class HealthResponse(BaseModel):
    status:    str
    model_id:  str
    threshold: float
    device:    str
    version:   str


# ── Core logic ────────────────────────────────────────────────────────────────

def build_input_text(job: JobPosting) -> str:
    """
    Recreates the exact preprocessing used during training:
    structured key-value fields first, then free-text fields,
    all joined with ' [SEP] '.
    """
    structured = [
        ("Location",            job.location),
        ("Salary Range",        job.salary_range),
        ("Employment Type",     job.employment_type),
        ("Required Experience", job.required_experience),
        ("Required Education",  job.required_education),
        ("Department",          job.department),
        ("Industry",            job.industry),
        ("Function",            job.function),
        ("Has Company Logo",    str(job.has_company_logo) if job.has_company_logo is not None else None),
        ("Telecommuting",       str(job.telecommuting)    if job.telecommuting    is not None else None),
        ("Has Questions",       str(job.has_questions)    if job.has_questions    is not None else None),
    ]
    free_text = [
        job.title,
        job.company_profile,
        job.description,
        job.requirements,
        job.benefits,
    ]
    parts = []
    for label, val in structured:
        v = (val or "").strip()
        if v:
            parts.append(f"{label}: {v}")
    for val in free_text:
        v = (val or "").strip()
        if v:
            parts.append(v)

    return " [SEP] ".join(parts)


def _infer(texts: list[str]) -> list[float]:
    """Run tokenization + forward pass. Returns fraud probabilities."""
    encoded = _tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        logits = _model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()  # LABEL_1 = fraud
    return probs


def _make_response(prob: float, latency_ms: float) -> PredictResponse:
    is_fraud = prob >= THRESHOLD
    dist     = abs(prob - THRESHOLD)
    conf     = "HIGH" if dist > 0.25 else "MEDIUM" if dist > 0.10 else "LOW"
    return PredictResponse(
        fraud_probability = round(prob, 4),
        fraud_percent     = round(prob * 100, 1),
        verdict           = "FRAUDULENT" if is_fraud else "LEGITIMATE",
        confidence        = conf,
        threshold         = THRESHOLD,
        model_id          = MODEL_ID,
        latency_ms        = round(latency_ms, 1),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check — confirms the model is loaded and ready."""
    return HealthResponse(
        status    = "ok",
        model_id  = MODEL_ID,
        threshold = THRESHOLD,
        device    = DEVICE,
        version   = "1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(job: JobPosting):
    """
    Predict whether a single job posting is fraudulent.

    - Accepts the full EMSCAD schema (all fields optional).
    - Returns fraud_probability (0–1), verdict, and confidence band.
    - Threshold: 0.87 (calibrated on validation set for Precision ≥ 0.93).
    """
    t0   = time.perf_counter()
    text = build_input_text(job)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No usable text fields provided.")
    prob = _infer([text])[0]
    return _make_response(prob, (time.perf_counter() - t0) * 1000)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(req: BatchRequest):
    """
    Predict fraud for a batch of up to 16 job postings in one call.
    More efficient than calling /predict in a loop.
    """
    if len(req.postings) > MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(req.postings)} exceeds maximum {MAX_BATCH}."
        )
    t0    = time.perf_counter()
    texts = [build_input_text(p) for p in req.postings]
    # Replace empty texts with a placeholder so batch stays aligned
    texts = [t if t.strip() else "[EMPTY]" for t in texts]
    probs = _infer(texts)
    total_ms = (time.perf_counter() - t0) * 1000
    per_ms   = total_ms / len(probs)
    return BatchResponse(
        results    = [_make_response(p, per_ms) for p in probs],
        count      = len(probs),
        latency_ms = round(total_ms, 1),
    )
