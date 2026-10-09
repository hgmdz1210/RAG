FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Both images run as this user, so model weights and datasets that one of them
# writes to a shared ./data mount stay readable by the other.
RUN useradd --create-home app

# Dependencies come first so this layer is rebuilt only when the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# Test image: adds the dev dependency group. Mount the repository at /app to
# run the tests, the linters or the benchmark against the working tree.
FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
# The mounted tree may not be writable by this user, so keep tool caches off.
ENV PYTEST_ADDOPTS="-p no:cacheprovider" \
    RUFF_NO_CACHE=true
USER app
CMD ["pytest"]


# Runtime image: the API server.
FROM base AS runtime
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable
RUN mkdir -p /app/data && chown -R app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
