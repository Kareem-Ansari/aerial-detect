# syntax=docker/dockerfile:1

# ---------- Build stage: install dependencies with uv ----------
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: cached until pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group train --no-install-project

# Then the project code (changes often, rebuilds in seconds)
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group train

# ---------- Runtime stage: only what is needed to serve ----------
FROM python:3.12-slim-bookworm
# OpenCV (used by Ultralytics) needs these system libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MODEL_PATH=/models/best.pt
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "aerial_detect.api:app", "--host", "0.0.0.0", "--port", "8000"]