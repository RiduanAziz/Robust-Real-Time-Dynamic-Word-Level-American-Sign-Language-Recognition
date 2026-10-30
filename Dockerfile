# ==============================================================================
# Stage 1: Build production React frontend bundle
# ==============================================================================
FROM node:24-slim AS frontend-builder
WORKDIR /build

COPY app/frontend/package*.json ./
RUN npm ci

COPY app/frontend ./
RUN npm run build

# ==============================================================================
# Stage 2: Production Python backend & static file server
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8000 \
    SIGNFLOW_DEVICE=cpu

WORKDIR /app

# Install system libraries for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies
COPY requirements-runtime.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runtime.txt && \
    pip install --no-cache-dir -e .

# Copy application source, scripts, and experiment configs
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Copy compiled React frontend bundle into static mount path
COPY --from=frontend-builder /build/dist ./app/frontend/dist

# Create runtime directories for models and logs, and set up non-root user
RUN mkdir -p /app/models /app/results && \
    useradd -m -u 1000 signflow && \
    chown -R signflow:signflow /app

USER signflow

EXPOSE 8000

# Health check verifies the live process probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health/live')" || exit 1

# Launch Uvicorn binding to 0.0.0.0 and reading dynamic cloud $PORT
CMD ["sh", "-c", "exec uvicorn sign_language.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
