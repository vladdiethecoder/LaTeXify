FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# Python 3.11, Rust/Cargo (for UniMERNet extensions), PDF tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    build-essential \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
# Copy src and scripts later to utilize cache if dependencies didn't change
# But uv sync needs pyproject.toml. 
# If uv.lock doesn't match, it might resolve.

# Install dependencies
# Note: We use system python3.11 for the venv
RUN uv venv .venv --python 3.11 && \
    uv pip sync pyproject.toml --frozen || uv pip install -r pyproject.toml

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY README.md ./

# Set path to use venv
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download models (Optional, can be done at runtime)
# RUN python3 scripts/download_models.py 

CMD ["python3", "run_latexify.py"]