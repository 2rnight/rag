"""片段仓库与文档注册表（索引的全部"事实来源"）。

为什么需要它们？
    FAISS 只存向量 + 一个自增 ID，`Document` 内容存在 docstore（pkl）里，
    想做「按文件删除」「列出知识库」「统计切片数」「重建 BM25」这些管理动作会很别扭。
    所以这里额外维护两份轻量持久化数据：

    data/faiss_index/chunks.jsonl   一行一个片段（chunk_id + 原文 + 元数据）
    data/faiss_index/registry.json  一份文档台账（文件哈希、切片数、解析器、时间）

    删除文档时：先按 source_file 找到 chunk_id 列表 -> FAISS.delete(ids) -> 同步清理
    仓库与 BM25 -> 持久化。这样"增量入库/删除/重建"三件事都能做到幂等、可审计。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .logging_utils import get_logger

logger = get_logger(__name__)


def _atomic_write(path: Path, content: str) -> None:
    """原子写：先写临时文件再 rename，避免进程中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"chunk_id": self.chunk_id, "text": self.text, "metadata": self.metadata},
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChunkRecord:
        return cls(
            chunk_id=payload["chunk_id"],
            text=payload.get("text", ""),
            metadata=payload.get("metadata") or {},
        )


class ChunkStore:
    """片段仓库（JSONL，逐行 append 友好）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, ChunkRecord] = {}
        self._dirty = False

    # ---------------- 持久化 ----------------
    def load(self) -> int:
        self._records.clear()
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ChunkRecord.from_dict(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("跳过损坏的片段行: %s", line[:80])
                    continue
                self._records[record.chunk_id] = record
        logger.info("片段仓库已加载: %d 条 (%s)", len(self._records), self.path)
        return len(self._records)

    def save(self) -> None:
        content = "\n".join(record.to_json() for record in self._records.values())
        _atomic_write(self.path, content + ("\n" if content else ""))
        self._dirty = False

    # ---------------- 读写 ----------------
    def add_many(self, chunks: Iterable[Chunk]) -> int:
        added = 0
        for chunk in chunks:
            self._records[chunk.chunk_id] = ChunkRecord(
                chunk_id=chunk.chunk_id, text=chunk.text, metadata=dict(chunk.metadata)
            )
            added += 1
        self._dirty = True
        return added

    def remove_sources(self, sources: set[str]) -> list[str]:
        """按文件名删除，返回被删除的 chunk_id 列表。"""
        removed_ids = [
            chunk_id
            for chunk_id, record in self._records.items()
            if str(record.metadata.get("source_file", "")) in sources
        ]
        for chunk_id in removed_ids:
            self._records.pop(chunk_id, None)
        self._dirty = bool(removed_ids)
        return removed_ids

    def remove_doc_ids(self, doc_ids: set[str]) -> list[str]:
        removed_ids = [
            chunk_id
            for chunk_id, record in self._records.items()
            if str(record.metadata.get("doc_id", "")) in doc_ids
        ]
        for chunk_id in removed_ids:
            self._records.pop(chunk_id, None)
        self._dirty = bool(removed_ids)
        return removed_ids

    def clear(self) -> None:
        self._records.clear()
        self._dirty = True

    # ---------------- 查询 ----------------
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ChunkRecord]:
        return iter(self._records.values())

    @property
    def texts(self) -> list[str]:
        return [record.text for record in self._records.values()]

    def get(self, chunk_id: str) -> ChunkRecord | None:
        return self._records.get(chunk_id)

    def all(self) -> list[ChunkRecord]:
        return list(self._records.values())

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            source = str(record.metadata.get("source_file", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def fingerprint(self) -> str:
        """内容指纹：用于校验 BM25 缓存是否仍有效。"""
        digest = hashlib.sha1()
        for chunk_id in sorted(self._records):
            digest.update(chunk_id.encode("utf-8"))
        return digest.hexdigest()[:16]

    def stats(self) -> dict[str, Any]:
        sizes = [len(r.text) for r in self._records.values()]
        return {
            "chunks": len(self._records),
            "sources": len(self.source_counts()),
            "chars": sum(sizes),
            "avg_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "fingerprint": self.fingerprint(),
        }


class DocumentRegistry:
    """文档台账：记录每个文件解析出来的切片数、哈希、时间，支撑增量/幂等入库。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._docs: dict[str, dict[str, Any]] = {}

    def load(self) -> int:
        self._docs.clear()
        if not self.path.exists():
            return 0
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("registry.json 损坏，已忽略: %s", self.path)
            return 0
        self._docs = payload.get("documents", {})
        return len(self._docs)

    def save(self) -> None:
        payload = {"updated_at": _now(), "documents": self._docs}
        _atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2))

    def upsert(self, doc: dict[str, Any]) -> None:
        doc_id = str(doc["doc_id"])
        existing = self._docs.get(doc_id, {})
        merged = {**existing, **doc}
        merged.setdefault("first_ingested_at", doc.get("ingested_at") or _now())
        self._docs[doc_id] = merged

    def remove_source(self, source_file: str) -> None:
        for doc_id in [k for k, v in self._docs.items() if v.get("source_file") == source_file]:
            self._docs.pop(doc_id, None)

    def find_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        for doc in self._docs.values():
            if doc.get("file_hash") == file_hash:
                return doc
        return None

    def find_by_source(self, source_file: str) -> dict[str, Any] | None:
        for doc in self._docs.values():
            if doc.get("source_file") == source_file:
                return doc
        return None

    def all(self) -> list[dict[str, Any]]:
        return sorted(self._docs.values(), key=lambda d: str(d.get("source_file", "")))

    def clear(self) -> None:
        self._docs.clear()

    def __len__(self) -> int:
        return len(self._docs)

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self._docs),
            "chunks": sum(int(d.get("chunks", 0)) for d in self._docs.values()),
            "sources": [d.get("source_file") for d in self.all()],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
