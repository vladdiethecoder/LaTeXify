FROM ghcr.io/astral-sh/uv:latest AS uv-base

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    build-essential \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-base /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv .venv --python 3.11 && \
    .venv/bin/uv sync

FROM base AS runtime

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY README.md ./

CMD ["uv", "run", "python", "run_latexify.py"]
