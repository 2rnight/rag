"""依赖注入：进程内单例的 RagPipeline。"""
from __future__ import annotations

import threading
from collections.abc import Iterator

from fastapi import HTTPException, Request

from config import get_settings
from rag.exceptions import RagError
from rag.logging_utils import get_logger
from rag.pipeline import RagPipeline

logger = get_logger(__name__)

_lock = threading.Lock()
_pipeline: RagPipeline | None = None


def build_pipeline(*, strict: bool = True) -> RagPipeline:
    """构建（或复用）全局流水线：加载索引 + 重建 BM25。"""
    global _pipeline
    with _lock:
        if _pipeline is None:
            settings = get_settings()
            pipeline = RagPipeline(settings)
            try:
                pipeline.bootstrap(strict=strict)
            except RagError as exc:
                # embedding 指纹不匹配等：降级为"空知识库"启动，接口层给出明确提示，
                # 而不是让整个服务起不来（可用性优先）
                logger.error("流水线启动降级（%s）：%s", type(exc).__name__, exc.message)
                pipeline.ready = False
                pipeline.last_error = exc.message
            _pipeline = pipeline
        return _pipeline


def get_pipeline() -> Iterator[RagPipeline]:
    """FastAPI 依赖：拿到全局流水线实例。"""
    pipeline = build_pipeline()
    if not pipeline.ready and pipeline.vector_store.size and pipeline.last_error:
        raise HTTPException(status_code=409, detail=pipeline.last_error)
    yield pipeline


def pipeline_from_request(request: Request) -> RagPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    return pipeline or build_pipeline()
