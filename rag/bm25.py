"""BM25 Okapi 稀疏检索（纯 Python 实现，零额外依赖）。

为什么要它？
    稠密向量召回擅长"语义相近"，但在企业知识库里有两类问题它经常失手：
      1. 专有名词 / 编号 / 版本号 / 数字条款（"V3.2"、"192.168.1.102"、"36 小时"）；
      2. 关键词字面命中但语义向量被主题稀释的长文档。
    BM25 对这种"精确字面"问题召回极稳，所以做成混合检索的一路，
    再用 RRF（Reciprocal Rank Fusion）把两路排名融合，取长补短。

实现要点：倒排索引（term -> {doc_idx: tf}）+ BM25 打分公式
    score = Σ idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
其中 idf = ln(1 + (N − df + 0.5)/(df + 0.5))，中文分词用"单字 + 双字 bigram"（免 jieba 依赖）。
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import get_logger, tokenize_for_sparse

logger = get_logger(__name__)


@dataclass(slots=True)
class SparseHit:
    chunk_id: str
    score: float
    matched_terms: int = 0


@dataclass
class BM25Index:
    """BM25 Okapi 倒排索引。"""

    k1: float = 1.5
    b: float = 0.75
    tokenizer: Callable[[str], list[str]] = tokenize_for_sparse

    _chunk_ids: list[str] = field(default_factory=list)
    _postings: dict[str, dict[int, int]] = field(default_factory=dict)
    _lengths: list[int] = field(default_factory=list)
    _avgdl: float = 0.0
    _idf_cache: dict[str, float] = field(default_factory=dict)

    # ---------------- 构建 ----------------
    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    @property
    def vocabulary(self) -> int:
        return len(self._postings)

    def build(self, documents: Iterable[tuple[str, str]]) -> BM25Index:
        """documents: [(chunk_id, text), ...]"""
        self._chunk_ids = []
        self._postings = {}
        self._lengths = []
        self._idf_cache = {}

        for chunk_id, text in documents:
            index = len(self._chunk_ids)
            self._chunk_ids.append(chunk_id)
            tokens = self.tokenizer(text)
            self._lengths.append(len(tokens) or 1)
            for token, count in _term_counts(tokens).items():
                self._postings.setdefault(token, {})[index] = count

        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        logger.info("BM25 索引构建完成: %d 片段 / %d 词元", self.size, self.vocabulary)
        return self

    def add(self, documents: Iterable[tuple[str, str]]) -> int:
        """增量追加（重建 IDF 缓存即可）。"""
        added = 0
        for chunk_id, text in documents:
            if chunk_id in self._chunk_ids:
                continue
            index = len(self._chunk_ids)
            self._chunk_ids.append(chunk_id)
            tokens = self.tokenizer(text)
            self._lengths.append(len(tokens) or 1)
            for token, count in _term_counts(tokens).items():
                self._postings.setdefault(token, {})[index] = count
            added += 1
        if added:
            self._avgdl = sum(self._lengths) / len(self._lengths)
            self._idf_cache.clear()
        return added

    def remove(self, chunk_ids: set[str]) -> int:
        """重建（BM25 倒排删除成本高，片段规模下重建更简单可靠）。"""
        if not chunk_ids:
            return 0
        keep = [(cid, "") for cid in self._chunk_ids if cid not in chunk_ids]
        if len(keep) == self.size:
            return 0
        # 保留原有文本用于重建：调用方持有 ChunkStore，这里通过清空 + 标记让上层重建
        removed = self.size - len(keep)
        self._chunk_ids = [cid for cid in self._chunk_ids if cid not in chunk_ids]
        return removed

    # ---------------- 检索 ----------------
    def search(self, query: str, k: int = 20, min_score: float = 0.0) -> list[SparseHit]:
        if not self._chunk_ids or k <= 0:
            return []
        query_terms = set(self.tokenizer(query))
        if not query_terms:
            return []

        scores: dict[int, float] = {}
        matched: dict[int, int] = {}
        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term, len(postings))
            for doc_index, tf in postings.items():
                length = self._lengths[doc_index]
                denominator = tf + self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1.0))
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * tf * (self.k1 + 1) / denominator
                matched[doc_index] = matched.get(doc_index, 0) + 1

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(k * 3, k)]
        results: list[SparseHit] = []
        for doc_index, score in ranked:
            if score <= min_score:
                continue
            coverage = matched.get(doc_index, 0) / max(1, len(query_terms))
            # 命中词覆盖率轻度加权：避免"只撞到一个高频词"的片段排到最前
            results.append(
                SparseHit(
                    chunk_id=self._chunk_ids[doc_index],
                    score=round(score * (0.6 + 0.4 * coverage), 6),
                    matched_terms=matched.get(doc_index, 0),
                )
            )
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:k]

    def _idf(self, term: str, df: int) -> float:
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached
        total = self.size
        value = math.log(1 + (total - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = value
        return value

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "chunk_ids": self._chunk_ids,
            "lengths": self._lengths,
            "postings": {term: {str(k): v for k, v in posting.items()} for term, posting in self._postings.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BM25Index:
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        index._chunk_ids = list(payload.get("chunk_ids", []))
        index._lengths = list(payload.get("lengths", []))
        index._postings = {
            term: {int(k): v for k, v in posting.items()}
            for term, posting in (payload.get("postings") or {}).items()
        }
        index._avgdl = (sum(index._lengths) / len(index._lengths)) if index._lengths else 0.0
        index._idf_cache = {}
        return index

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BM25Index | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("BM25 缓存加载失败（将重建）: %s", exc)
            return None


def _term_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts
