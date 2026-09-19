# Two stages so the model weights are baked into the image rather than downloaded on first
# request. A container that needs the network before it can answer is not reproducible, and a
# cold start that silently takes minutes is worse than one that fails fast.

FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

WORKDIR /build

# Dependency layer first: it changes far less often than the source.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Pre-fetch the pinned model revision so the runtime image needs no network.
ENV HF_HOME=/opt/hf
RUN --mount=type=cache,target=/root/.cache/huggingface \
    /build/.venv/bin/python -c "\
from transformers import AutoModel, AutoTokenizer; \
m='BAAI/bge-small-en-v1.5'; r='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'; \
AutoTokenizer.from_pretrained(m, revision=r); \
AutoModel.from_pretrained(m, revision=r); \
print('model cached')"


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    ATLASRAG_DATA_DIR=/data

RUN useradd --create-home --uid 10001 atlas \
    && mkdir -p /data /opt/hf \
    && chown -R atlas:atlas /data /opt/hf

WORKDIR /app

COPY --from=builder --chown=atlas:atlas /build/.venv /app/.venv
COPY --from=builder --chown=atlas:atlas /opt/hf /opt/hf
COPY --chown=atlas:atlas src/ /app/src/
COPY --chown=atlas:atlas fixtures/ /app/fixtures/

USER atlas

# The registry and both indexes live here. Without a volume, every restart re-ingests.
VOLUME ["/data"]

EXPOSE 8000

# Readiness, not liveness: a process that is up but whose indexes are unusable is not ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"

CMD ["uvicorn", "atlasrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
