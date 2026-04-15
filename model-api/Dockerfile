# ── FraudGuard RoBERTa API — HuggingFace Spaces Dockerfile ──────────────────
#
# HuggingFace Spaces (Docker SDK) requirements:
#   - Must expose port 7860
#   - App must listen on 0.0.0.0:7860
#   - User must be non-root (UID 1000)
#
# Build locally:
#   docker build -t fraudguard-api .
#
# Run locally:
#   docker run -p 7860:7860 fraudguard-api
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (required by HuggingFace Spaces) ───────────────────────────
RUN useradd -m -u 1000 appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app.py .

# ── Pre-download model weights at build time (bakes into image layer) ─────────
# This means the container starts instantly — no cold-start download delay.
COPY download_model.py .
RUN python download_model.py

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Expose HuggingFace Spaces port ───────────────────────────────────────────
EXPOSE 7860

# ── Environment defaults (overridable at runtime) ────────────────────────────
ENV MODEL_ID="aditya963/fraud-job-classifier" \
    FRAUD_THRESHOLD="0.87" \
    MAX_LENGTH="512" \
    MAX_BATCH_SIZE="16" \
    PYTHONUNBUFFERED="1"

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
