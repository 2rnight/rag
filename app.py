"""FastAPI 应用入口。

启动：
    uvicorn app:app --host 0.0.0.0 --port 8000          # 直接跑
    python app.py                                       # 等价（读取 RAG_PORT）
    docker compose up -d                                # 容器化

在线体验：
    http://localhost:8000          演示界面（流式打字机效果）
    http://localhost:8000/docs     OpenAPI 交互式文档
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.deps import build_pipeline, pipeline_from_request
from api.routes import router
from config import get_settings
from rag.exceptions import RagError
from rag.logging_utils import get_logger, setup_logging

settings = get_settings()
setup_logging(settings.log_level, settings.log_json)
logger = get_logger("rag.api")

STATIC_DIR = Path(__file__).resolve().parent / "api" / "static"

TAGS_METADATA = [
    {"name": "chat", "description": "问答：流式（SSE）与非流式，含拒答兜底与引用溯源"},
    {"name": "knowledge", "description": "知识库管理：上传、列表、删除、重建、状态"},
    {"name": "system", "description": "系统：健康检查、公开配置、运行指标"},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时加载索引（BM25 缓存命中则秒开），关闭时无需额外清理。"""
    settings.ensure_dirs()
    pipeline = build_pipeline(strict=False)
    app.state.pipeline = pipeline
    logger.info(
        "服务启动完成",
        extra={
            "extra_fields": {
                "version": settings.version,
                "vectors": pipeline.vector_store.size,
                "chunks": len(pipeline.chunk_store),
                "llm": pipeline.llm.info.as_dict(),
                "embedding": pipeline.embeddings.info.fingerprint(),
                "docs_ui": f"http://localhost:{settings.port}/",
            }
        },
    )
    yield
    logger.info("服务已停止")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "基于 RAG（检索增强生成）的企业/个人知识库智能问答系统。\n\n"
        "- 多格式文档解析（PDF / Word / Markdown / TXT / CSV / HTML / JSON）\n"
        "- 重叠窗口切片（Overlap 20%）+ text-embedding-v3 向量化 + FAISS 本地索引\n"
        "- 混合检索（稠密向量 + BM25）+ RRF 融合 + 拒答兜底\n"
        "- FastAPI 异步 + SSE 流式输出，首字响应时间（TTFT）大幅降低\n"
    ),
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """访问日志 + 耗时埋点（流式响应只记录到"开始返回"的时间）。"""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not request.url.path.startswith("/static"):
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed_ms, 1),
                }
            },
        )
    return response


@app.exception_handler(RagError)
async def rag_error_handler(request: Request, exc: RagError) -> JSONResponse:
    """业务异常 -> 结构化 JSON（统一错误码，前端好处理）。"""
    logger.warning("业务异常 %s: %s", type(exc).__name__, exc.message)
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": type(exc).__name__, "message": exc.message, "detail": exc.detail},
    )


# ---------------------------------------------------------------- 基础设施端点（不带前缀，方便探针直接使用）


@app.get("/health", include_in_schema=False)
async def root_health(request: Request) -> dict[str, Any]:
    pipeline = pipeline_from_request(request)
    return {
        "status": "ok" if pipeline.ready else "degraded",
        "version": settings.version,
        "vectors": pipeline.vector_store.size,
        "chunks": len(pipeline.chunk_store),
        "llm": pipeline.llm.info.as_dict(),
    }


@app.get("/metrics", include_in_schema=False)
async def root_metrics(request: Request) -> dict[str, Any]:
    from rag.metrics import metrics

    return metrics.snapshot()


# ---------------------------------------------------------------- 路由与静态资源

app.include_router(router, prefix="/api/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> Any:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html; charset=utf-8")
    return JSONResponse({"message": settings.app_name, "docs": "/docs"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Any:
    icon = STATIC_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return JSONResponse(status_code=204, content=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
