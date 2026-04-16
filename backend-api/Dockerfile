# ─────────────────────────────────────────────────────────────────────────────
# FraudGuard Backend API — Dockerfile
# Optimised for HuggingFace Spaces (Docker SDK, port 7860).
#
# Build stages:
#   builder  — install Python deps into a virtual-env (cached layer)
#   runtime  — minimal runtime image, copies only the venv + app code
#
# Key decisions:
#   • python:3.11-slim keeps the image small (~200 MB vs ~1 GB for full image)
#   • Two-stage copy (requirements first, then src) maximises layer cache reuse
#   • Non-root user (appuser) follows container security best practices
#   • HEALTHCHECK lets HF Spaces / Docker know when the app is ready
#   • torch is CPU-only — no CUDA drivers needed on free-tier Spaces
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile Python wheels and run whois/curl at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        whois \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so we can copy it cleanly to the
# runtime stage without dragging in build tools.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the dependency manifest first so Docker caches this expensive
# layer — it is only invalidated when requirements.txt actually changes.
COPY requirements.txt .

# Install PyTorch CPU-only wheel explicitly before the rest of requirements
# to avoid pulling in CUDA variants (~3 GB) that are unnecessary on free Spaces.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.2.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Re-install only the runtime system packages (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        whois \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Security: run as non-root user ────────────────────────────────────────────
# HuggingFace Spaces runs containers as UID 1000 by default.
# Creating the user explicitly ensures consistent permissions.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# Copy application source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Environment defaults (override via HF Space Secrets in production)
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Pre-download Transformers cache to /app/.cache so it survives across
    # container restarts when a persistent storage volume is mounted.
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# ── Health check ──────────────────────────────────────────────────────────────
# Docker / HF Spaces uses this to determine when the container is healthy.
# Checks every 30s, allows 120s startup grace (model download on cold start).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
