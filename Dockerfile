# ---------- 构建阶段：装依赖（利用 wheel 缓存，减小最终镜像） ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------- 运行阶段 ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# 非 root 运行
RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . /app
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# 启动前先确保示例文档已索引（幂等：内容没变会跳过）
CMD ["sh", "-c", "python -m scripts.ingest || true; exec uvicorn app:app --host 0.0.0.0 --port 8000"]
