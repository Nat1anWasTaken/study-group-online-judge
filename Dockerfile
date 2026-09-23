# syntax=docker/dockerfile:1.7

FROM python:3.14-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app


FROM base AS build

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM base AS judge

RUN groupadd --gid 10001 judge \
    && useradd --uid 10001 --gid 10001 --create-home judge \
    && mkdir -p /data \
    && chown judge:judge /data

COPY --from=build --chown=judge:judge /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    JUDGE_DATABASE_PATH=/data/judge.db \
    PYTHONUNBUFFERED=1

USER 10001:10001

CMD ["uvicorn", "judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
