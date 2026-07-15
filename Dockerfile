FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=false

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

WORKDIR /app

COPY --chown=app:app pyproject.toml uv.lock README.md ./

USER app

RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app . .

RUN uv sync --frozen --no-dev \
    && python -m compileall -q src \
    && python -m src.indexing.ensure_indexes

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]