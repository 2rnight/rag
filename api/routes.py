"""HTTP 接口层（FastAPI 路由）。

接口地图：
    GET    /                          Web 演示界面
    GET    /health                    健康检查（K8s liveness/readiness 可直接用）
    GET    /api/v1/config             公开配置（不含密钥）
    GET    /api/v1/metrics            运行指标：TTFT / 检索耗时 / 拒答率 / 入库量
    POST   /api/v1/chat               非流式问答（JSON）
    POST   /api/v1/chat/stream        ★ 流式问答（SSE，前端主链路）
    GET    /api/v1/chat/stream        ★ 流式问答（EventSource 兼容）
    POST   /api/v1/search             纯检索（不调 LLM，用于调召回）
    POST   /api/v1/knowledge/upload   上传文档入库（多文件）
    GET    /api/v1/knowledge/documents 知识库文档列表
    DELETE /api/v1/knowledge/documents/{source_file}  删除文档（四处状态同步）
    POST   /api/v1/knowledge/rebuild  全量重建（幂等）
    POST   /api/v1/knowledge/reset     清空知识库（危险操作）
    GET    /api/v1/knowledge/status    知识库详细状态 + 当前生效配置
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from config import BASE_DIR, get_settings
from rag.exceptions import RagError, UnsupportedFileTypeError
from rag.logging_utils import get_logger
from rag.metrics import metrics
from rag.pipeline import RagPipeline

from .deps import get_pipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResultModel,
    KnowledgeStatusResponse,
    MetricsResponse,
    RebuildRequest,
    RebuildResponse,
    SearchResponse,
    UploadResponse,
)
from .sse import SSE_HEADERS, SSE_MEDIA_TYPE, stream_with_heartbeat

logger = get_logger(__name__)
router = APIRouter()

_PUBLIC_ROOT = BASE_DIR.resolve()


def _resolve_server_path(raw: str) -> Path:
    """限制服务端目录参数只能落在项目目录内（防目录穿越）。"""
    candidate = (BASE_DIR / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not str(candidate).startswith(str(_PUBLIC_ROOT)):
        raise HTTPException(status_code=400, detail="仅允许访问项目目录内的路径")
    return candidate


# ================================================================ 系统


@router.get("/health", response_model=HealthResponse, tags=["system"], summary="健康检查")
async def health(pipeline: RagPipeline = Depends(get_pipeline)) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok" if pipeline.ready else "degraded",
        version=settings.version,
        uptime_seconds=round(time.time() - metrics.started_at, 1),
        knowledge_base_ready=pipeline.vector_store.size > 0,
        knowledge_base_chunks=len(pipeline.chunk_store),
        llm_provider=pipeline.llm.info.provider,
        llm_model=pipeline.llm.info.model,
        llm_fallback=pipeline.llm.info.is_fallback,
        embedding_model=f"{pipeline.embeddings.info.provider}:{pipeline.embeddings.info.model}",
    )


@router.get("/config", tags=["system"], summary="当前生效配置（不含密钥）")
async def public_config() -> dict[str, Any]:
    return get_settings().public_dict()


@router.get("/metrics", response_model=MetricsResponse, tags=["system"], summary="运行指标")
async def get_metrics(pipeline: RagPipeline = Depends(get_pipeline)) -> MetricsResponse:
    return MetricsResponse(
        metrics=metrics.snapshot(),
        chunking=pipeline.chunking_stats().as_dict() if len(pipeline.chunk_store) else None,
    )


# ================================================================ 问答


@router.post("/chat", response_model=ChatResponse, tags=["chat"], summary="非流式问答")
async def chat(payload: ChatRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> ChatResponse:
    result = await run_in_threadpool(
        pipeline.answer,
        payload.question,
        [message.model_dump() for message in payload.history],
        top_k=payload.top_k,
        threshold=payload.threshold,
        mmr=payload.mmr,
    )
    return ChatResponse(**result.as_dict())


@router.post("/chat/stream", tags=["chat"], summary="流式问答（SSE）")
async def chat_stream_post(
    payload: ChatRequest,
    request: Request,
    pipeline: RagPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """主链路：检索 → 拒答判定 → LLM 流式生成 → SSE 逐字下发。

    事件序列：
        open  -> 连接建立（带心跳间隔）
        meta  -> 检索命中信息（先于生成返回，前端可立即渲染引用卡片）
        delta -> 文本增量（核心，首字延迟 TTFT 由这里体现）
        done  -> 汇总：答案、耗时、引用、落地性校验
        error -> 异常（不会返回 5xx，流内已可读）
    """
    return _streaming_response(pipeline, payload, request)


@router.get("/chat/stream", tags=["chat"], summary="流式问答（EventSource 兼容）")
async def chat_stream_get(
    request: Request,
    question: str = Query(..., min_length=1, max_length=1000, description="用户问题"),
    top_k: int | None = Query(default=None, ge=1, le=20),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    mmr: bool | None = Query(default=None),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    payload = ChatRequest(question=question, top_k=top_k, threshold=threshold, mmr=mmr)
    return _streaming_response(pipeline, payload, request)


def _streaming_response(pipeline: RagPipeline, payload: ChatRequest, request: Request) -> StreamingResponse:
    settings = get_settings()
    source = pipeline.astream_answer(
        payload.question,
        [message.model_dump() for message in payload.history],
        top_k=payload.top_k,
        threshold=payload.threshold,
        mmr=payload.mmr,
    )
    stream = stream_with_heartbeat(source, heartbeat=settings.sse_heartbeat_interval)
    headers = {**SSE_HEADERS, "X-Question-Length": str(len(payload.question))}
    return StreamingResponse(stream, media_type=SSE_MEDIA_TYPE, headers=headers)


@router.post("/search", response_model=SearchResponse, tags=["chat"], summary="纯检索（不调用大模型）")
async def search(payload: ChatRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> SearchResponse:
    result = await run_in_threadpool(
        pipeline.retrieve,
        payload.question,
        top_k=payload.top_k,
        threshold=payload.threshold,
        mmr=payload.mmr,
    )
    return SearchResponse(
        question=payload.question,
        answerable=result.is_answerable,
        strategy=result.strategy,
        top_score=round(result.top_score, 4),
        threshold=round(result.threshold, 4),
        timings_ms={key: round(value, 2) for key, value in result.timings.items()},
        results=[
            {**hit.to_dict(index), "index": index}
            for index, hit in enumerate(result.hits, start=1)
        ],
    )


# ================================================================ 知识库


@router.post("/knowledge/upload", response_model=UploadResponse, tags=["knowledge"], summary="上传文档入库")
async def upload(
    files: list[UploadFile] = File(..., description="支持 PDF/Word/Markdown/TXT/CSV/HTML/JSON"),
    force: bool = Query(default=False, description="内容未变化时是否强制重新入库"),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> UploadResponse:
    settings = get_settings()
    started = time.perf_counter()
    results: list[IngestResultModel] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024

    for upload_file in files:
        filename = upload_file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix not in pipeline.loader.supported_suffixes:
            error = UnsupportedFileTypeError(suffix, pipeline.loader.supported_suffixes)
            results.append(
                IngestResultModel(
                    source_file=filename,
                    skipped=True,
                    reason=f"{error.message}（{error.detail}）",
                )
            )
            continue
        data = await upload_file.read()
        if len(data) > max_bytes:
            results.append(
                IngestResultModel(
                    source_file=filename,
                    skipped=True,
                    reason=f"文件超过上限 {settings.max_upload_mb}MB",
                )
            )
            continue
        try:
            result = await run_in_threadpool(pipeline.ingest_bytes, data, filename, force=force)
            results.append(IngestResultModel(**result.as_dict()))
        except RagError as exc:
            metrics.incr("errors")
            results.append(
                IngestResultModel(source_file=filename, skipped=True, reason=f"{type(exc).__name__}: {exc.message}")
            )
        finally:
            await upload_file.close()

    return UploadResponse(
        results=results,
        documents_total=len(pipeline.registry),
        chunks_total=len(pipeline.chunk_store),
        vectors_total=pipeline.vector_store.size,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.get("/knowledge/documents", response_model=DocumentListResponse, tags=["knowledge"], summary="文档列表")
async def list_documents(pipeline: RagPipeline = Depends(get_pipeline)) -> DocumentListResponse:
    documents = [
        {
            "doc_id": str(doc.get("doc_id", "")),
            "source_file": str(doc.get("source_file", "")),
            "doc_type": str(doc.get("doc_type", "")),
            "parser": str(doc.get("parser", "")),
            "chunks": int(doc.get("chunks", 0) or 0),
            "chars": int(doc.get("chars", 0) or 0),
            "quality": float(doc.get("quality", 0.0) or 0.0),
            "ingested_at": str(doc.get("ingested_at", "")),
            "file_hash": str(doc.get("file_hash", "")),
        }
        for doc in pipeline.registry.all()
    ]
    return DocumentListResponse(
        documents=documents,
        chunks_total=len(pipeline.chunk_store),
        vectors_total=pipeline.vector_store.size,
        sources=pipeline.chunk_store.source_counts(),
    )


@router.delete(
    "/knowledge/documents/{source_file}",
    response_model=DeleteResponse,
    tags=["knowledge"],
    summary="删除文档",
)
async def delete_document(source_file: str, pipeline: RagPipeline = Depends(get_pipeline)) -> DeleteResponse:
    result = await run_in_threadpool(pipeline.delete_document, source_file)
    if result["removed_chunks"] == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档: {source_file}")
    return DeleteResponse(**result)


@router.post("/knowledge/rebuild", response_model=RebuildResponse, tags=["knowledge"], summary="全量重建索引")
async def rebuild(payload: RebuildRequest | None = None, pipeline: RagPipeline = Depends(get_pipeline)) -> RebuildResponse:
    settings = get_settings()
    directory = _resolve_server_path(payload.directory) if payload and payload.directory else settings.docs_dir
    if not Path(directory).exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {directory}")
    result = await run_in_threadpool(pipeline.rebuild_all, directory)
    return RebuildResponse(**result)


@router.post("/knowledge/reset", tags=["knowledge"], summary="清空知识库")
async def reset(pipeline: RagPipeline = Depends(get_pipeline)) -> dict[str, Any]:
    await run_in_threadpool(pipeline.reset)
    return {"status": "ok", "chunks_total": len(pipeline.chunk_store), "vectors": pipeline.vector_store.size}


@router.get(
    "/knowledge/status",
    response_model=KnowledgeStatusResponse,
    tags=["knowledge"],
    summary="知识库状态与生效配置",
)
async def knowledge_status(pipeline: RagPipeline = Depends(get_pipeline)) -> KnowledgeStatusResponse:
    return KnowledgeStatusResponse(**pipeline.status())
