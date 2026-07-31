FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest   /uv   /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest   /uvx  /usr/local/bin/uvx

# Install system dependencies for mysqlclient + uv
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    pkg-config \
    gcc \
    curl \
    iputils-ping \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT="/opt/apps/venv" \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=10 \
    UV_CACHE_DIR=/root/.cache/uv \
    VIRTUAL_ENV="/opt/apps/venv" \
    PATH="/opt/apps/venv/bin:/root/.local/bin:${PATH}"

##    UV_SYSTEM_PYTHON=1 \
##    UV_LINK_MODE=copy \
##    UV_HTTP_TIMEOUT=300 \
##    UV_HTTP_RETRIES=10 \
##    UV_CACHE_DIR=/root/.cache/uv \
##    UV_DEFAULT_INDEX=https://mirror.yandex.ru/pypi/simple \
##    PIP_INDEX_URL=https://mirror.yandex.ru/pypi/s imple \


WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync #--frozen --no-dev

COPY . .
