# Production Backend Dockerfile for Railway / Container Environments
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install CPU-only PyTorch from the official PyTorch CPU wheel index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install application dependencies (layer-cached)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application codebase and bootstrap scripts
COPY backend /app/backend
COPY scripts /app/scripts
COPY pyproject.toml /app/

# 4. Prepare default persistent volume directories
RUN mkdir -p /data/chroma /data/storage /data/huggingface

EXPOSE 8000

# Start FastAPI application with dynamic Railway PORT expansion
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
