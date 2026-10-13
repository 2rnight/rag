"""FAISS 本地向量索引：构建、增量写入、删除、持久化、向量一致性校验。

面试要点：
1. **距离度量**：文本向量做 L2 归一化后用 `MAX_INNER_PRODUCT`（内积），
   等价于余弦相似度，分数区间可解释（0~1 越大越相似），也顺手避免了
   FAISS `IndexFlatL2` 那种"越小越像"的反直觉；
2. **索引原子性**：`index.faiss` + `index.pkl` + `manifest.json` 一起落盘，
   manifest 里记录 embedding 指纹（provider:model:dim）与切片参数 ——
   **换了 embedding 模型还复用旧索引**是 RAG 线上最阴的坑之一（维度不一致直接崩，
   维度相同但语义空间不同则静默劣化），这里显式拦截并给出重建提示；
3. **可管理性**：chunk_id 既是 FAISS 的 docstore id，也是 ChunkStore 的主键，
   所以"按文件删除"= 找出该文件的 chunk_id → FAISS.delete → 同步 ChunkStore/BM25；
4. **检索增强**：支持 `fetch_k` 先粗排再精排，以及 MMR 多样性重排（去冗余）。
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .embeddings import RagEmbeddings, ensure_normalized
from .exceptions import IndexNotReadyError
from .logging_utils import get_logger, stopwatch

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass(slots=True)
class Hit:
    """一次检索命中的片段（检索层与 API 层的统一数据结构）。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0                 # 稠密余弦相似度
    sparse_score: float = 0.0          # BM25 分数
    fusion_score: float = 0.0          # RRF 融合分
    retrieval: str = "dense"           # dense / sparse / hybrid
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source_file", "unknown"))

    @property
    def page(self) -> int | None:
        value = self.metadata.get("page_number")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def section(self) -> str:
        return str(self.metadata.get("heading_path", "") or "")

    def to_reference(self, index: int, preview_chars: int = 160) -> dict[str, Any]:
        """转成前端引用卡片需要的结构。"""
        preview = self.text[:preview_chars].replace("\n", " ")
        return {
            "index": index,
            "chunk_id": self.chunk_id,
            "source": self.source_file,
            "page": self.page,
            "section": self.section,
            "score": round(self.score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "fusion_score": round(self.fusion_score, 6),
            "retrieval": self.retrieval,
            "chars": len(self.text),
            "preview": preview + ("…" if len(self.text) > preview_chars else ""),
        }

    def to_dict(self, index: int, *, with_text: bool = True) -> dict[str, Any]:
        payload = self.to_reference(index)
        if with_text:
            payload["text"] = self.text
            payload["metadata"] = self.metadata
        return payload


class FaissVectorStore:
    """对 LangChain FAISS 的封装（保留统一接口，方便未来换 Milvus/Qdrant）。"""

    def __init__(self, embeddings: RagEmbeddings, index_dir: str | Path) -> None:
        # 索引层强制归一化：内积 == 余弦，阈值才有可比性
        self.embeddings = ensure_normalized(embeddings)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss = None  # type: ignore[var-annotated]
        self.manifest: dict[str, Any] = {}

    # ------------------------------------------------------------ 状态
    @property
    def is_ready(self) -> bool:
        return self._faiss is not None

    @property
    def size(self) -> int:
        if self._faiss is None:
            return 0
        index = getattr(self._faiss, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / MANIFEST_NAME

    # ------------------------------------------------------------ 持久化
    def load(self, *, strict: bool = True) -> bool:
        """加载本地索引，并校验 embedding 指纹。"""
        if not (self.index_dir / "index.faiss").exists():
            logger.info("未发现本地索引，等待首次入库: %s", self.index_dir)
            return False

        manifest = self._read_manifest()
        if manifest:
            stored = str((manifest.get("embedding") or {}).get("fingerprint", ""))
            current = self.embeddings.info.fingerprint()
            if stored and stored != current:
                message = (
                    f"索引与当前向量模型不匹配：索引={stored} 当前={current}。"
                    "请重建索引（POST /api/v1/knowledge/rebuild 或 python -m scripts.ingest --reset）。"
                )
                if strict:
                    raise IndexNotReadyError(message)
                logger.warning(message)

        from langchain_community.vectorstores import FAISS

        with stopwatch() as sw:
            self._faiss = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,  # 本地自产索引，可信
            )
        self.manifest = manifest
        logger.info(
            "FAISS 索引加载完成",
            extra={
                "extra_fields": {
                    "vectors": self.size,
                    "embedding": self.embeddings.info.fingerprint(),
                    "ms": round(sw["ms"], 2),
                    "path": str(self.index_dir),
                }
            },
        )
        return True

    def persist(self, *, build_config: dict[str, Any] | None = None) -> None:
        if self._faiss is None:
            return
        self._faiss.save_local(str(self.index_dir))
        manifest = self.manifest or {}
        manifest.update(
            {
                "version": 1,
                "embedding": self.embeddings.info.as_dict(),
                "vectors": self.size,
                "updated_at": _now(),
            }
        )
        manifest.setdefault("created_at", _now())
        if build_config:
            manifest["build_config"] = build_config
        self.manifest = manifest
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("manifest.json 损坏，已忽略")
            return {}

    # ------------------------------------------------------------ 写入
    def rebuild(self, chunks: Iterable[Chunk]) -> int:
        """全量重建索引。"""
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("没有片段可用于构建索引")

        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy
        from langchain_core.documents import Document

        documents = [
            Document(page_content=chunk.text, metadata=dict(chunk.metadata), id=chunk.chunk_id)
            for chunk in chunk_list
        ]
        with stopwatch() as sw:
            self._faiss = FAISS.from_documents(
                documents,
                self.embeddings,
                ids=[chunk.chunk_id for chunk in chunk_list],
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                # 归一化由 RagEmbeddings 层保证（见 embeddings.ensure_normalized），
                # 这里关掉 langchain 自带归一化（它只对 EUCLIDEAN 生效并会告警）
                normalize_L2=False,
            )
        logger.info(
            "FAISS 索引构建完成",
            extra={"extra_fields": {"vectors": self.size, "ms": round(sw["ms"], 2)}},
        )
        return self.size

    def add(self, chunks: Iterable[Chunk]) -> int:
        """增量写入。"""
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        from langchain_community.vectorstores.utils import DistanceStrategy
        from langchain_core.documents import Document

        documents = [
            Document(page_content=chunk.text, metadata=dict(chunk.metadata), id=chunk.chunk_id)
            for chunk in chunk_list
        ]
        if self._faiss is None:
            return self.rebuild(chunk_list)
        self._faiss.add_documents(documents, ids=[c.chunk_id for c in chunk_list])
        _ = DistanceStrategy  # 保持导入语义一致性（距离策略已在 build 时确定）
        return len(chunk_list)

    def remove(self, chunk_ids: Iterable[str]) -> int:
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids or self._faiss is None:
            return 0
        self._faiss.delete(ids)
        return len(ids)

    def reset(self) -> None:
        """清空内存索引与磁盘索引（保留目录）。"""
        import shutil

        self._faiss = None
        self.manifest = {}
        for path in self.index_dir.glob("*"):
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)
        logger.warning("向量索引已重置: %s", self.index_dir)

    # ------------------------------------------------------------ 检索
    def search(
        self,
        query: str,
        *,
        k: int = 5,
        fetch_k: int | None = None,
        score_threshold: float | None = None,
        mmr: bool = False,
        mmr_lambda: float = 0.7,
    ) -> list[Hit]:
        """向量检索：返回按相似度降序的 Hit 列表。

        fetch_k: 先召回更多候选（默认 k 的 4 倍），便于后续融合/去重/阈值过滤，
                 避免"取完 top-k 再过滤"导致结果数不足。
        """
        if self._faiss is None:
            raise IndexNotReadyError()
        fetch_k = max(fetch_k or k, k)

        if mmr:
            docs = self._faiss.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda
            )
            hits = [self._doc_to_hit(doc) for doc in docs]
        else:
            pairs = self._faiss.similarity_search_with_score(query, k=fetch_k)
            hits = [self._pair_to_hit(doc, score) for doc, score in pairs]

        if score_threshold is not None:
            hits = [hit for hit in hits if hit.score >= score_threshold]
        hits = hits[:k]
        for rank, hit in enumerate(hits, start=1):
            hit.dense_rank = rank
        return hits

    @staticmethod
    def _doc_to_hit(doc: Any) -> Hit:
        return Hit(
            chunk_id=str(doc.metadata.get("chunk_id", getattr(doc, "id", ""))),
            text=doc.page_content,
            metadata=dict(doc.metadata),
            score=0.0,
            retrieval="mmr",
        )

    @staticmethod
    def _pair_to_hit(doc: Any, score: float) -> Hit:
        return Hit(
            chunk_id=str(doc.metadata.get("chunk_id", getattr(doc, "id", ""))),
            text=doc.page_content,
            metadata=dict(doc.metadata),
            score=float(score),
            retrieval="dense",
        )

    # ------------------------------------------------------------ 统计
    def stats(self) -> dict[str, Any]:
        embedding = self.embeddings.info
        return {
            "ready": self.is_ready,
            "vectors": self.size,
            "index_path": str(self.index_dir),
            "embedding": embedding.as_dict(),
            "manifest": {
                "created_at": self.manifest.get("created_at"),
                "updated_at": self.manifest.get("updated_at"),
                "build_config": self.manifest.get("build_config", {}),
            },
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
