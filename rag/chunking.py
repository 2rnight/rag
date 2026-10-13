"""重叠窗口切片（Chunking）—— Overlap 20% 的语义连续性保障。

核心策略（面试主线之一）：

0. **小段落聚合**：结构化文档（Markdown/员工手册这类）往往是一堆很短的小节，
   直接按节成块会导致"块太小、语义不完整、向量被噪声主导"。
   所以先贪心聚合相邻小节到接近 `chunk_size`（保留各自的 heading_path），
   再进入重叠窗口切分 —— 这是让 Overlap 策略真正生效的前提；
1. **窗口 + 边界回退**：按 `chunk_size` 开窗，窗口右边界向前回退到最近的
   句子/段落边界（"。"、"！"、"？"、"；"、换行……），避免把一句话拦腰截断；
2. **重叠窗口**：下一块起点 = 上一块终点 − `chunk_size × 20%`，
   并向**句首**对齐，保证每一块的起始都是一个完整句子，语义不断档；
3. **长度护栏**：`chunk_min_size` 以下的碎片与相邻块合并；
   连续重复（或完全被上一块包含）的片段直接丢弃，避免 Faiss 中出现"影子向量"；
4. **可解释元数据**：每个 chunk 记录 `char_start/char_end/overlap_chars/heading_path/page`，
   检索命中后能精确回溯到原文位置，前端可以做"引用高亮"。

为什么是 500/100（20%）？
    - 太小：语义碎片化，单块信息不足，召回噪声大；
    - 太大：向量被多主题稀释，且拼进 Prompt 会挤占上下文预算；
    - 500 中文字 ≈ 500 token 量级，配合 20% 重叠，既能挡住"跨块问题"，又不至于重复过多。
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .loaders import ParsedDocument, normalize_text
from .logging_utils import get_logger, stopwatch

logger = get_logger(__name__)

# 强边界：句子结束 / 段落结束
STRONG_BOUNDARIES = ("。", "！", "？", "；", "!", "?", ";", "\n")
# 次强边界（只在找不到强边界时使用）
WEAK_BOUNDARIES = ("，", ",", "：", ":", " ", "、")


@dataclass(slots=True)
class Chunk:
    """切片结果：文本 + 溯源元数据。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class ChunkingStats:
    """切片统计，用于入库报告与压测日志。"""

    documents: int = 0
    chunks: int = 0
    chars: int = 0
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    avg_overlap_chars: float = 0.0
    overlap_ratio: float = 0.0
    dropped_duplicates: int = 0
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "chars": self.chars,
            "avg_chars": round(self.avg_chars, 1),
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "avg_overlap_chars": round(self.avg_overlap_chars, 1),
            "overlap_ratio": round(self.overlap_ratio, 3),
            "dropped_duplicates": self.dropped_duplicates,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class OverlapChunker:
    """重叠窗口切片器。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int | None = None,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 60,
        separators: Iterable[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        self.chunk_size = chunk_size
        # 重叠字符数：显式传 chunk_overlap 优先，否则按比例推导（默认 20%）
        self.chunk_overlap = (
            max(1, int(chunk_overlap))
            if chunk_overlap is not None
            else max(1, round(chunk_size * overlap_ratio))
        )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size，否则滑动窗口无法前进")
        self.overlap_ratio = self.chunk_overlap / chunk_size
        self.min_chunk_size = min_chunk_size
        self.separators = tuple(separators) if separators else STRONG_BOUNDARIES + WEAK_BOUNDARIES

    # -------------------------------------------------------------- 文本切分
    def split_text(self, text: str) -> list[str]:
        """把一段长文本切成带重叠的片段。"""
        return [chunk_text for _, chunk_text, _ in self.split_text_with_offsets(text)]

    def split_text_with_offsets(self, text: str) -> list[tuple[int, str, int]]:
        """返回 [(char_start, chunk_text, overlap_chars_with_previous), ...]。"""
        text = normalize_text(text)
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [(0, text, 0)]

        windows = self._base_windows(text)
        results: list[tuple[int, str, int]] = []
        for start, end, overlap in windows:
            segment = text[start:end].strip()
            if not segment:
                continue
            results.append((start, segment, overlap))
        return self._post_process(results)

    # -------------------------------------------------------------- 内部实现
    def _base_windows(self, text: str) -> list[tuple[int, int, int]]:
        """滑动窗口：每块长度 ≤ chunk_size，相邻块重叠 ≈ chunk_overlap。

        返回 [(start, end, overlap_with_previous), ...]。
        步长 = chunk_size − overlap，所以"重叠 20%"指的是重叠部分占 chunk_size 的比例。
        """
        windows: list[tuple[int, int, int]] = []
        total = len(text)
        start = 0
        previous_end = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            if end < total:
                end = self._rewind_to_boundary(text, start, end)
            if end <= start:  # 防御性兜底：保证窗口一定前进
                end = min(start + self.chunk_size, total)
            overlap = max(0, previous_end - start)
            windows.append((start, end, overlap))
            if end >= total:
                break
            previous_end = end
            next_start = max(0, end - self.chunk_overlap)
            # 新块起点对齐到句首/行首，避免以半句话开头
            next_start = self._advance_to_boundary(text, next_start, end)
            start = max(min(next_start, end - 1), start + 1)
        return windows

    def _rewind_to_boundary(self, text: str, start: int, end: int) -> int:
        """把窗口右边界向前回退到最近的句子边界。

        先找强边界；找不到再退到弱边界；都没有就直接硬切。
        回退幅度上限为 45% 的窗口长度，防止过度回退导致块过短。
        """
        lower_bound = start + max(self.min_chunk_size, int(self.chunk_size * 0.55))
        if lower_bound >= end:
            return end
        search = slice(lower_bound, end)
        for boundary_set in (STRONG_BOUNDARIES, WEAK_BOUNDARIES):
            best = -1
            for boundary in boundary_set:
                index = text.rfind(boundary, search.start, search.stop)
                if index > best:
                    best = index
            if best != -1:
                # 段落分隔符（\n\n）优先保留为块尾
                return best + 1
        return end

    def _advance_to_boundary(self, text: str, target: int, limit: int) -> int:
        """块起点对齐：找到 [target, limit] 区间内第一个句首/行首，作为新起点。

        找不到时回退到 target（保证重叠量），而不是硬跳到 limit。
        """
        for boundary in ("\n", "。", "！", "？", "；", "!", "?", ";"):
            index = text.find(boundary, target, limit)
            if index != -1:
                return index + 1
        return target

    def _post_process(self, results: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
        """合并过短片段 + 去重（重复块会污染向量库并挤占 Prompt 预算）。"""
        cleaned: list[tuple[int, str, int]] = []
        for start, text, overlap in results:
            if not text:
                continue
            if cleaned:
                prev_start, prev_text, _ = cleaned[-1]
                if text == prev_text:  # 完全重复
                    continue
                if text in prev_text:  # 被上一块完全包含
                    continue
                if len(text) < self.min_chunk_size and prev_start + len(prev_text) + 2 >= start:
                    # 过短且与上一块相邻 -> 合并
                    merged = f"{prev_text}\n{text}"
                    cleaned[-1] = (prev_start, merged, cleaned[-1][2])
                    continue
            cleaned.append((start, text, overlap))
        return cleaned

    # -------------------------------------------------------------- 文档切片
    def group_blocks(self, documents: list[ParsedDocument]) -> list[ParsedDocument]:
        """把同一文件内过短的相邻小节聚合到接近 chunk_size，再做窗口切分。

        只聚合"同一文件 + 合并后不超过 1.5×chunk_size"的相邻块，
        避免把整篇文档糊成一坨；每个被聚合块的原 heading_path 会保留在
        `heading_paths` 里，引用溯源依然精确到小节。
        """
        groups: list[ParsedDocument] = []
        for doc in documents:
            current = groups[-1] if groups else None
            same_file = (
                current is not None
                and current.metadata.get("source_file") == doc.metadata.get("source_file")
                and current.metadata.get("file_hash") == doc.metadata.get("file_hash")
            )
            mergeable = (
                same_file
                and len(current.text) < self.chunk_size
                and len(current.text) + len(doc.text) + 2 <= int(self.chunk_size * 1.5)
            )
            if not mergeable:
                groups.append(
                    ParsedDocument(
                        text=doc.text,
                        metadata={**doc.metadata, "_paths": [str(doc.metadata.get("heading_path", ""))]},
                    )
                )
                continue
            current.text = f"{current.text}\n\n{doc.text}"
            current.metadata["_paths"].append(str(doc.metadata.get("heading_path", "")))
            if "page" in doc.metadata and "page_end" not in current.metadata:
                current.metadata["page_end"] = doc.metadata.get("page")
            current.metadata["quality"] = min(
                float(current.metadata.get("quality", 1.0) or 1.0),
                float(doc.metadata.get("quality", 1.0) or 1.0),
            )

        for group in groups:
            paths = [path for path in group.metadata.pop("_paths", []) if path]
            group.metadata["heading_paths"] = paths
            if paths:
                group.metadata["heading_path"] = paths[0]
            group.metadata["blocks_merged"] = len(paths)
        return groups

    def split_document(
        self,
        doc: ParsedDocument,
        doc_id: str | None = None,
        *,
        index_offset: int = 0,
    ) -> list[Chunk]:
        """把一个 ParsedDocument 切成 Chunk 列表（带完整溯源元数据）。

        index_offset：一个文件会被解析成多个逻辑块（按章节/按页），
        chunk 序号必须在**整个文件范围内**连续，否则不同块会生成相同 chunk_id
        （Faiss 会直接报重复 id），这是很容易踩的坑。
        """
        source_file = str(doc.metadata.get("source_file", "unknown"))
        doc_id = doc_id or make_doc_id(source_file, str(doc.metadata.get("file_hash", "")))
        pieces = self.split_text_with_offsets(doc.text)

        chunks: list[Chunk] = []
        for offset, (char_start, text, overlap) in enumerate(pieces):
            index = index_offset + offset
            metadata: dict[str, Any] = {
                **{k: v for k, v in doc.metadata.items() if v is not None},
                "doc_id": doc_id,
                "block_index": offset,
                "chunk_index": index,
                "total_chunks": len(pieces),
                "char_start": char_start,
                "char_end": char_start + len(text),
                "overlap_chars": overlap,
                "chars": len(text),
                "chunk_strategy": f"overlap-{int(self.overlap_ratio * 100)}%",
            }
            if "page" in metadata:
                try:
                    metadata["page_number"] = int(metadata["page"]) + 1
                except (TypeError, ValueError):
                    pass
            chunk_id = f"{doc_id}-{index:04d}"
            metadata["chunk_id"] = chunk_id
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
        return chunks

    def chunk_documents(
        self, documents: Iterable[ParsedDocument], *, desc: str = ""
    ) -> tuple[list[Chunk], ChunkingStats]:
        """批量切片，返回 (chunks, 统计信息)。"""
        stats = ChunkingStats()
        chunks: list[Chunk] = []
        with stopwatch() as sw:
            doc_list = self.group_blocks(list(documents))
            offsets: dict[str, int] = {}
            for doc in doc_list:
                source_file = str(doc.metadata.get("source_file", "unknown"))
                doc_id = make_doc_id(source_file, str(doc.metadata.get("file_hash", "")))
                doc_chunks = self.split_document(
                    doc, doc_id=doc_id, index_offset=offsets.get(doc_id, 0)
                )
                offsets[doc_id] = offsets.get(doc_id, 0) + len(doc_chunks)
                before = len(doc_chunks)
                # 跨块去重（同一文档内文本完全相同的块只保留一次）
                seen = {c.text for c in chunks[-64:]}
                doc_chunks = [c for c in doc_chunks if c.text not in seen]
                stats.dropped_duplicates += before - len(doc_chunks)
                chunks.extend(doc_chunks)
            # 回填每个文档的总切片数（分块过程中无法预知）
            totals: dict[str, int] = {}
            for chunk in chunks:
                doc_id = str(chunk.metadata.get("doc_id", ""))
                totals[doc_id] = max(totals.get(doc_id, 0), int(chunk.metadata.get("chunk_index", 0)) + 1)
            for chunk in chunks:
                chunk.metadata["total_chunks"] = totals.get(str(chunk.metadata.get("doc_id", "")), 1)
            stats.documents = len(totals)
            stats.chunks = len(chunks)
            stats.chars = sum(c.chars for c in chunks)
            if chunks:
                lengths = [c.chars for c in chunks]
                stats.avg_chars = sum(lengths) / len(lengths)
                stats.min_chars = min(lengths)
                stats.max_chars = max(lengths)
                overlaps = [int(c.metadata.get("overlap_chars", 0) or 0) for c in chunks[1:]]
                stats.avg_overlap_chars = sum(overlaps) / len(overlaps) if overlaps else 0.0
                stats.overlap_ratio = stats.avg_overlap_chars / self.chunk_size
        stats.elapsed_ms = round(sw["ms"], 2)
        logger.info(
            "切片完成 %s",
            desc,
            extra={"extra_fields": stats.as_dict()},
        )
        return chunks, stats


def make_doc_id(source_file: str, file_hash: str = "") -> str:
    """稳定的文档 ID（同一份文件内容重复入库时，ID 保持一致）。"""
    seed = f"{source_file}:{file_hash}".encode()
    return hashlib.sha1(seed).hexdigest()[:12]
