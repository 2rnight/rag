"""端到端编排层（入库 / 检索 / 流式问答 / 埋点）。

一句话总结数据流：
    文档 → [混合解析] → [重叠切片] → [向量化] → FAISS + BM25 + ChunkStore
    问题 → [混合检索 + RRF] → [拒答判定] → [动态 Prompt] → [LLM 流式生成] → [答案校验]

这一层是"业务编排"，不含算法细节：解析/切分/向量/检索/生成各自独立成模块，
方便单测、替换与压测（例如压测里直接注入一个模拟 LLM 就能量化 TTFT 收益）。
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .chunking import ChunkingStats, OverlapChunker, make_doc_id
from .embeddings import RagEmbeddings, build_embeddings
from .exceptions import (
    DocumentParseError,
    KnowledgeBaseEmptyError,
    RagError,
)
from .llm import LLMClient, build_llm_client, verify_answer
from .loaders import HybridDocumentLoader
from .logging_utils import estimate_tokens, get_logger, stopwatch
from .metrics import metrics
from .prompts import build_messages, extract_context_from_messages
from .retriever import HybridRetriever, RetrievalResult
from .store import ChunkStore, DocumentRegistry
from .vector_store import FaissVectorStore


@dataclass(slots=True)
class IngestResult:
    """单个文件的入库结果（也是入库 API 的响应体）。"""

    source_file: str
    doc_id: str = ""
    doc_type: str = ""
    parser: str = ""
    blocks: int = 0
    chunks: int = 0
    chars: int = 0
    quality: float = 0.0
    chunks_total: int = 0
    skipped: bool = False
    reason: str = ""
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryResult:
    """非流式问答结果。"""

    question: str
    answer: str
    references: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    fallback: bool = False
    grounded: bool = False
    citations: list[int] = field(default_factory=list)
    coverage: float = 0.0
    notice: str = ""
    retrieval: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    provider: str = ""
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RagPipeline:
    """RAG 主流程编排器（进程内单例，见 `api/deps.py`）。"""

    def __init__(
        self,
        settings,
        *,
        embeddings: RagEmbeddings | None = None,
        chat_model: Any | None = None,
        loader: HybridDocumentLoader | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(settings)
        self.llm: LLMClient = build_llm_client(settings)
        if chat_model is not None:  # 压测/测试注入点（例如模拟 3.2s 的真实上游）
            from langchain_core.language_models.chat_models import BaseChatModel

            from .llm import ExtractiveChatModel

            assert isinstance(chat_model, BaseChatModel)
            self.llm = LLMClient(
                primary=chat_model,
                provider="injected",
                model=getattr(chat_model, "model_name", "injected"),
                temperature=settings.llm_temperature,
                first_token_timeout=getattr(settings, "llm_first_token_timeout", 20.0),
                fallback=ExtractiveChatModel(refusal_message=settings.refusal_message),
                refusal_message=settings.refusal_message,
            )

        self.loader = loader or HybridDocumentLoader()
        self.chunker = OverlapChunker(
            chunk_size=settings.chunk_size,
            overlap_ratio=settings.chunk_overlap_ratio,
            min_chunk_size=settings.chunk_min_size,
            separators=settings.separators,
        )
        self.chunk_store = ChunkStore(settings.chunk_store_path)
        self.registry = DocumentRegistry(settings.registry_path)
        self.vector_store = FaissVectorStore(self.embeddings, settings.index_dir)
        self.bm25 = BM25Index()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            chunk_store=self.chunk_store,
            bm25=self.bm25,
            top_k=settings.retrieval_top_k,
            fetch_k=settings.retrieval_fetch_k,
            hybrid=settings.hybrid_enabled,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
            expand_neighbors=settings.neighbor_expansion,
            mmr_lambda=settings.mmr_lambda,
            max_unknown_ratio=settings.max_unknown_ratio,
            sparse_rescue_min_score=settings.sparse_rescue_min_score,
            sparse_rescue_max_unknown_ratio=settings.sparse_rescue_max_unknown_ratio,
            sparse_rescue_dense_ratio=settings.sparse_rescue_dense_ratio,
            sparse_rescue_min_standout=settings.sparse_rescue_min_standout,
        )
        self.ready = False
        self.last_error: str | None = None

    # ================================================================ 启动
    def bootstrap(self, *, strict: bool = True) -> RagPipeline:
        """加载磁盘上的索引/片段仓库/文档台账，并重建 BM25。"""
        with stopwatch() as sw:
            self.chunk_store.load()
            self.registry.load()
            try:
                self.vector_store.load(strict=strict)
            except RagError as exc:
                self.last_error = exc.message
                raise
            self._load_or_build_bm25()
            self.ready = True
        # 阈值：不同 embedding 的余弦分布差异很大，用模型自带的经验阈值
        self.settings._calibrated_threshold = self.embeddings.info.recommended_threshold
        logger = get_logger(__name__)
        logger.info(
            "RAG 流水线就绪",
            extra={
                "extra_fields": {
                    "vectors": self.vector_store.size,
                    "chunks": len(self.chunk_store),
                    "documents": len(self.registry),
                    "embedding": self.embeddings.info.fingerprint(),
                    "llm": self.llm.info.as_dict(),
                    "threshold": round(self.settings.effective_score_threshold, 3),
                    "bootstrap_ms": round(sw["ms"], 2),
                }
            },
        )
        return self

    def _load_or_build_bm25(self) -> None:
        """BM25 缓存命中则直接复用，否则从片段仓库重建。"""
        if len(self.chunk_store) == 0:
            self._set_bm25(BM25Index())
            return
        cached = BM25Index.load(self.settings.bm25_path)
        fingerprint = self.chunk_store.fingerprint()
        cached_ids = set(cached._chunk_ids) if cached else set()
        if cached and cached_ids == {record.chunk_id for record in self.chunk_store}:
            self._set_bm25(cached)
            get_logger(__name__).info("BM25 索引已从缓存加载: %d 片段 (%s)", cached.size, fingerprint)
            return
        with stopwatch() as sw:
            index = BM25Index().build((record.chunk_id, record.text) for record in self.chunk_store)
        index.save(self.settings.bm25_path)
        self._set_bm25(index)
        get_logger(__name__).info("BM25 索引重建完成: %d 片段 (%.1fms)", index.size, sw["ms"])

    def _set_bm25(self, index: BM25Index) -> None:
        """替换 BM25 索引时，必须同步更新检索器持有的引用。

        ⚠️ 这是很容易踩的坑：`self.bm25 = 新对象` 之后，HybridRetriever 里
        还指着初始化时的那个空索引，稀疏召回会静默失效（表现为 bm25 分恒为 0）。
        """
        self.bm25 = index
        self.retriever.bm25 = index

    # ================================================================ 入库
    def ingest_file(self, path: str | Path, *, force: bool = False) -> IngestResult:
        """单文件入库。

        - 文件不存在于台账 -> 新增；
        - 已存在但内容变了（file_hash 不同）-> 先删除旧片段再入库（避免残留旧向量）；
        - 已存在且内容没变 -> 直接跳过（**幂等**，重复执行不产生重复向量）；
        - `force=True` -> 无视哈希强制重建（例如改了切片/向量模型参数）。
        """
        path = Path(path)
        started = time.perf_counter()
        with stopwatch():
            documents = self.loader.parse(path)
            report = self.loader.last_report[0] if self.loader.last_report else None
            file_hash = str(documents[0].metadata.get("file_hash", ""))
            doc_id = make_doc_id(path.name, file_hash)

            existing = self.registry.find_by_source(path.name)
            if existing and existing.get("file_hash") == file_hash and not force:
                return IngestResult(
                    source_file=path.name,
                    doc_id=str(existing.get("doc_id", doc_id)),
                    chunks=int(existing.get("chunks", 0)),
                    chunks_total=len(self.chunk_store),
                    skipped=True,
                    reason="文件内容未变化，已跳过（幂等：不会重复写入向量）",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )

            if existing:
                self.delete_document(path.name, persist=False)

            chunks, chunk_stats = self.chunker.chunk_documents(documents, desc=path.name)
            if not chunks:
                raise DocumentParseError(f"{path.name} 切片结果为空")

            with stopwatch() as emb_sw:
                self.vector_store.add(chunks)
                self._persist_index()
            metrics.embedding_latency.record(emb_sw["ms"])

            self.chunk_store.add_many(chunks)
            self.chunk_store.save()
            self.bm25.add((chunk.chunk_id, chunk.text) for chunk in chunks)
            self.bm25.save(self.settings.bm25_path)

            self.registry.upsert(
                {
                    "doc_id": doc_id,
                    "source_file": path.name,
                    "source_path": str(documents[0].metadata.get("source_path", path.name)),
                    "doc_type": report.doc_type if report else "unknown",
                    "parser": report.parser if report else "unknown",
                    "file_hash": file_hash,
                    "size_bytes": int(documents[0].metadata.get("size_bytes", 0) or 0),
                    "blocks": len(documents),
                    "chunks": len(chunks),
                    "chars": chunk_stats.chars,
                    "quality": report.quality if report else 0.0,
                    "ingested_at": _now(),
                }
            )
            self.registry.save()

            metrics.incr("ingest_jobs")
            metrics.incr("documents_indexed")
            metrics.incr("chunks_indexed", len(chunks))
            metrics.incr("upload_bytes", int(documents[0].metadata.get("size_bytes", 0) or 0))

        return IngestResult(
            source_file=path.name,
            doc_id=doc_id,
            doc_type=report.doc_type if report else "",
            parser=report.parser if report else "",
            blocks=len(documents),
            chunks=len(chunks),
            chars=chunk_stats.chars,
            quality=report.quality if report else 0.0,
            chunks_total=len(self.chunk_store),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            warnings=report.warnings if report else [],
        )

    def ingest_bytes(self, data: bytes, filename: str, *, force: bool = False) -> IngestResult:
        """上传文件入库：先落盘到 data/uploads（保留原始文件便于排查），再解析入库。"""
        safe_name = _safe_filename(filename)
        upload_dir = Path(self.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / safe_name
        target.write_bytes(data)
        return self.ingest_file(target, force=force)

    def ingest_directory(self, directory: str | Path, *, force: bool = False) -> list[IngestResult]:
        root = Path(directory)
        results: list[IngestResult] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not self.loader.supports(path):
                continue
            try:
                results.append(self.ingest_file(path, force=force))
            except RagError as exc:
                metrics.incr("errors")
                results.append(
                    IngestResult(
                        source_file=path.name,
                        skipped=True,
                        reason=f"{type(exc).__name__}: {exc.message}",
                        chunks_total=len(self.chunk_store),
                    )
                )
        return results

    def rebuild_all(self, directory: str | Path | None = None) -> dict[str, Any]:
        """全量重建：清空索引 -> 重新入库目录（幂等，可随时执行）。"""
        directory = directory or self.settings.docs_dir
        with stopwatch() as sw:
            self.reset(persist=True)
            results = self.ingest_directory(directory, force=True)
        return {
            "directory": str(directory),
            "documents": len([r for r in results if not r.skipped]),
            "failed": [r.as_dict() for r in results if r.skipped],
            "chunks": len(self.chunk_store),
            "vectors": self.vector_store.size,
            "elapsed_ms": round(sw["ms"], 2),
            "results": [r.as_dict() for r in results],
        }

    # ================================================================ 删除 / 重置
    def delete_document(self, source_file: str, *, persist: bool = True) -> dict[str, Any]:
        """按文件名删除文档：FAISS + ChunkStore + BM25 + 台账 四处保持一致。"""
        removed_ids = self.chunk_store.remove_sources({source_file})
        if removed_ids:
            self.vector_store.remove(removed_ids)
            self.chunk_store.save()
            self.registry.remove_source(source_file)
            self.registry.save()
            self._rebuild_bm25_from_store()
            if persist:
                self._persist_index()
            metrics.incr("chunks_deleted", len(removed_ids))
        return {
            "source_file": source_file,
            "removed_chunks": len(removed_ids),
            "chunks_total": len(self.chunk_store),
            "vectors": self.vector_store.size,
        }

    def reset(self, *, persist: bool = True) -> None:
        """清空全部知识库数据（索引目录整体重置）。"""
        self.vector_store.reset()
        self.chunk_store.clear()
        self.chunk_store.save()
        self.registry.clear()
        self.registry.save()
        self._set_bm25(BM25Index())
        if Path(self.settings.bm25_path).exists():
            Path(self.settings.bm25_path).unlink()
        if persist:
            self._persist_index()
        get_logger(__name__).warning("知识库已重置")

    def _persist_index(self) -> None:
        self.vector_store.persist(
            build_config={
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "chunk_overlap_ratio": self.settings.chunk_overlap_ratio,
                "separators": self.settings.separators,
                "hybrid": self.settings.hybrid_enabled,
            }
        )

    def _rebuild_bm25_from_store(self) -> None:
        index = BM25Index().build((record.chunk_id, record.text) for record in self.chunk_store)
        index.save(self.settings.bm25_path)
        self._set_bm25(index)

    # ================================================================ 检索
    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        mmr: bool | None = None,
    ) -> RetrievalResult:
        return self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=threshold,
            mmr=self.settings.retrieval_mmr if mmr is None else mmr,
        )

    # ================================================================ 问答（非流式）
    def answer(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None = None,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        mmr: bool | None = None,
    ) -> QueryResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            retrieval = self.retrieve(question, top_k=top_k, threshold=threshold, mmr=mmr)
        except KnowledgeBaseEmptyError:
            metrics.incr("empty_kb_hits")
            return QueryResult(
                question=question,
                answer=self.settings.empty_kb_message,
                refused=True,
                provider=self.llm.info.provider,
                model=self.llm.info.model,
                timings={"total_ms": round((time.perf_counter() - started) * 1000, 2)},
            )

        timings["retrieval_ms"] = round(sum(retrieval.timings.values()), 2)

        if not retrieval.is_answerable:
            metrics.observe_query(
                total_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=timings["retrieval_ms"],
                top_score=retrieval.top_score,
                refused=True,
                stream=False,
            )
            return QueryResult(
                question=question,
                answer=self.settings.refusal_message,
                references=[],
                refused=True,
                retrieval=retrieval.diagnostics(),
                timings={**timings, "total_ms": round((time.perf_counter() - started) * 1000, 2)},
                provider=self.llm.info.provider,
                model=self.llm.info.model,
            )

        messages, used_hits, _ = self._build_prompt(question, retrieval, history)
        with stopwatch() as llm_sw:
            answer = self.llm.invoke(messages)
        timings["llm_ms"] = round(llm_sw["ms"], 2)

        context = extract_context_from_messages(messages)
        check = verify_answer(answer, context, min_overlap=self.settings.grounding_min_overlap)
        answer, notice = self._post_process_answer(answer, check)

        total_ms = (time.perf_counter() - started) * 1000
        metrics.observe_query(
            total_ms=total_ms,
            retrieval_ms=timings["retrieval_ms"],
            top_score=retrieval.top_score,
            fallback=self.llm.used_fallback,
            stream=False,
        )
        return QueryResult(
            question=question,
            answer=answer,
            references=[hit.to_dict(index) for index, hit in enumerate(used_hits, start=1)],
            grounded=check.ok,
            citations=check.citations,
            coverage=check.coverage,
            notice=notice,
            fallback=self.llm.used_fallback,
            retrieval=retrieval.diagnostics(),
            timings={**timings, "total_ms": round(total_ms, 2)},
            provider=self.llm.info.provider,
            model=self.llm.info.model,
        )

    # ================================================================ 问答（流式）
    async def astream_answer(
        self,
        question: str,
        history: Sequence[dict[str, Any]] | None = None,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        mmr: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式问答：产出 meta / delta / done / error 四类事件（SSE 直接序列化）。

        事件顺序：meta（检索结果先给前端渲染引用卡片）→ delta 流 → done（耗时/校验信息）。
        ⚠️ 检索先于生成返回，前端可以先显示"检索到 N 条资料"，感知延迟进一步降低。
        """
        started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            try:
                retrieval = await asyncio.to_thread(
                    self.retrieve, question, top_k=top_k, threshold=threshold, mmr=mmr
                )
            except KnowledgeBaseEmptyError:
                metrics.incr("empty_kb_hits")
                yield _event("meta", {"answerable": False, "sources": [], "reason": "empty_kb", "kb_chunks": 0})
                yield _event("delta", {"text": self.settings.empty_kb_message})
                yield _event(
                    "done",
                    {
                        "answer": self.settings.empty_kb_message,
                        "refused": True,
                        "ttft_ms": round((time.perf_counter() - started) * 1000, 2),
                        "total_ms": round((time.perf_counter() - started) * 1000, 2),
                        "provider": self.llm.info.provider,
                        "model": self.llm.info.model,
                        "sources": [],
                    },
                )
                return

            timings["retrieval_ms"] = round(sum(retrieval.timings.values()), 2)
            yield _event(
                "meta",
                {
                    "answerable": retrieval.is_answerable,
                    "strategy": retrieval.strategy,
                    "top_score": round(retrieval.top_score, 4),
                    "threshold": round(retrieval.threshold, 4),
                    "retrieval_ms": timings["retrieval_ms"],
                    "sources": [
                        hit.to_dict(index) for index, hit in enumerate(retrieval.hits, start=1)
                    ],
                    "kb_chunks": len(self.chunk_store),
                },
            )

            if not retrieval.is_answerable:
                answer = self.settings.refusal_message
                yield _event("delta", {"text": answer})
                metrics.observe_query(
                    total_ms=(time.perf_counter() - started) * 1000,
                    retrieval_ms=timings["retrieval_ms"],
                    top_score=retrieval.top_score,
                    refused=True,
                )
                yield _event(
                    "done",
                    {
                        "answer": answer,
                        "refused": True,
                        "reason": retrieval.refusal_reason or "low_relevance",
                        "ttft_ms": round((time.perf_counter() - started) * 1000, 2),
                        "total_ms": round((time.perf_counter() - started) * 1000, 2),
                        "retrieval_ms": timings["retrieval_ms"],
                        "provider": self.llm.info.provider,
                        "model": self.llm.info.model,
                        "fallback": self.llm.used_fallback,
                        "sources": [],
                        "grounding": {"ok": True, "reason": "检索置信度不足，直接拒答"},
                    },
                )
                return

            messages, used_hits, _ = self._build_prompt(question, retrieval, history)
            answer_parts: list[str] = []
            ttft_ms: float | None = None
            fallback_used = False

            async for delta in self.llm.astream(messages):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                fallback_used = fallback_used or delta.fallback
                answer_parts.append(delta.text)
                yield _event("delta", {"text": delta.text, "fallback": delta.fallback})

            answer = "".join(answer_parts).strip()
            llm_ms = (time.perf_counter() - started) * 1000 - timings["retrieval_ms"]
            context = extract_context_from_messages(messages)
            check = verify_answer(answer, context, min_overlap=self.settings.grounding_min_overlap)
            if not check.ok:
                notice = f"（提示：{check.reason}，请核对下方引用原文）"
                yield _event("delta", {"text": f"\n\n{notice}"})
                answer = f"{answer}\n\n{notice}"

            total_ms = (time.perf_counter() - started) * 1000
            metrics.observe_query(
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                retrieval_ms=timings["retrieval_ms"],
                top_score=retrieval.top_score,
                fallback=self.llm.used_fallback or fallback_used,
            )
            yield _event(
                "done",
                {
                    "answer": answer,
                    "refused": False,
                    "ttft_ms": round(ttft_ms or total_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "retrieval_ms": timings["retrieval_ms"],
                    "llm_ms": round(max(llm_ms, 0.0), 2),
                    "provider": self.llm.info.provider,
                    "model": self.llm.info.model,
                    "fallback": self.llm.used_fallback or fallback_used,
                    "grounding": check.as_dict(),
                    "citations": check.citations,
                    "sources": [hit.to_dict(index) for index, hit in enumerate(used_hits, start=1)],
                    "context_tokens": estimate_tokens(context),
                    "vector_count": self.vector_store.size,
                },
            )
        except Exception as exc:  # noqa: BLE001 - 流式接口必须把错误变成事件而不是 500
            metrics.incr("errors")
            get_logger(__name__).exception("流式问答失败: %s", exc)
            yield _event(
                "error",
                {"message": f"生成失败: {exc}", "type": type(exc).__name__},
            )

    # ================================================================ 内部工具
    def _build_prompt(
        self,
        question: str,
        retrieval: RetrievalResult,
        history: Sequence[dict[str, Any]] | None,
    ) -> tuple[list[Any], list[Any], str]:
        history = list(history or [])[-self.settings.history_max_turns * 2 :]
        return build_messages(
            question,
            retrieval.hits,
            history=history,
            kb_name=self.settings.app_name,
            refusal_message=self.settings.refusal_message,
            threshold=retrieval.threshold,
            token_budget=self.settings.context_token_budget,
            per_hit_max_tokens=self.settings.context_snippet_max_tokens,
        )

    def _post_process_answer(self, answer: str, check) -> tuple[str, str]:
        """答案后处理：不达标时给出提示（strict 模式下切换为抽取式答案）。"""
        if check.ok:
            return answer, ""
        notice = f"（提示：{check.reason}，建议核对下方引用原文）"
        if self.settings.strict_grounding:
            return notice, notice
        return f"{answer}\n\n{notice}", notice

    # ================================================================ 状态
    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "last_error": self.last_error,
            "knowledge_base": {
                "chunks": len(self.chunk_store),
                "vectors": self.vector_store.size,
                "documents": len(self.registry),
                "sources": self.chunk_store.source_counts(),
                "index_path": str(self.settings.index_dir),
            },
            "documents": self.registry.all(),
            "embedding": self.embeddings.info.as_dict(),
            "llm": self.llm.info.as_dict(),
            "retrieval": {
                "strategy": "hybrid_rrf" if self.settings.hybrid_enabled else "dense",
                "top_k": self.settings.retrieval_top_k,
                "fetch_k": self.settings.retrieval_fetch_k,
                "rrf_k": self.settings.rrf_k,
                "score_threshold": round(self.settings.effective_score_threshold, 4),
                "context_token_budget": self.settings.context_token_budget,
            },
            "chunking": {
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "chunk_overlap_ratio": self.settings.chunk_overlap_ratio,
                "separators": self.settings.separators,
            },
            "supported_types": self.loader.supported_suffixes,
            "manifest": self.vector_store.manifest.get("build_config", {}),
        }

    def chunking_stats(self) -> ChunkingStats:
        """当前知识库的切片统计（在线校验切片参数是否合理）。"""
        stats = ChunkingStats()
        records = self.chunk_store.all()
        if not records:
            return stats
        lengths = [len(record.text) for record in records]
        overlaps = [int(record.metadata.get("overlap_chars", 0) or 0) for record in records]
        stats.documents = len(self.registry)
        stats.chunks = len(records)
        stats.chars = sum(lengths)
        stats.avg_chars = sum(lengths) / len(lengths)
        stats.min_chars = min(lengths)
        stats.max_chars = max(lengths)
        stats.avg_overlap_chars = sum(overlaps) / len(overlaps)
        stats.overlap_ratio = stats.avg_overlap_chars / max(1, self.chunker.chunk_size)
        return stats


def _safe_filename(filename: str) -> str:
    """文件名消毒：防目录穿越，保留中文与常见符号。"""
    name = Path(filename).name.strip().replace("\x00", "")
    allowed = "._- ()（）【】[]"
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in allowed or "\u4e00" <= ch <= "\u9fff")
    cleaned = cleaned.strip(". ") or "upload"
    return cleaned[:120]


def _event(event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event, "data": data}
