"""混合检索与召回优化。

链路：Query → (稠密 FAISS 召回 ∪ 稀疏 BM25 召回) → RRF 融合 → 去重 → 阈值判定 → 邻居扩展

为什么用 RRF 而不是简单加权分数？
    稠密余弦相似度（0~1）和 BM25（无上界）量纲不同，直接加权需要反复调参且不稳定；
    RRF（Reciprocal Rank Fusion, k=60）只看**排名**：score = Σ w / (k + rank)，
    无需归一化、对异常分数鲁棒，是工程上性价比最高的融合方案。

拒答判定：
    只有"检索置信度"低于阈值（或知识库为空）时直接兜底，
    把"不该答"的问题挡在 LLM 之前 —— 既省 token，也避免模型硬编答案造成幻觉。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .bm25 import BM25Index
from .exceptions import KnowledgeBaseEmptyError
from .logging_utils import get_logger, stopwatch, tokenize_for_sparse
from .store import ChunkStore
from .vector_store import FaissVectorStore, Hit

logger = get_logger(__name__)

_PUNCT_RE = re.compile(r"[\s，。！？；：、“”‘’（）()\[\]【】,.!?;:\"'`~@#$%^&*\-_=+|\\/<>]+")


@dataclass(slots=True)
class RetrievalResult:
    """一次检索的完整结果与诊断信息。"""

    hits: list[Hit] = field(default_factory=list)
    dense_hits: list[Hit] = field(default_factory=list)
    sparse_hits: list[Hit] = field(default_factory=list)
    top_score: float = 0.0
    is_answerable: bool = False
    strategy: str = "dense"
    threshold: float = 0.0
    query_terms: int = 0
    unknown_ratio: float = 0.0
    standout_ratio: float = 0.0
    refusal_reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def top_hit(self) -> Hit | None:
        return self.hits[0] if self.hits else None

    def diagnostics(self) -> dict[str, Any]:
        """暴露给日志/接口的检索诊断信息（方便调参与线上排查）。"""
        return {
            "strategy": self.strategy,
            "top_score": round(self.top_score, 4),
            "threshold": round(self.threshold, 4),
            "is_answerable": self.is_answerable,
            "hits": len(self.hits),
            "dense_hits": len(self.dense_hits),
            "sparse_hits": len(self.sparse_hits),
            "query_terms": self.query_terms,
            "unknown_ratio": round(self.unknown_ratio, 3),
            "standout_ratio": round(self.standout_ratio, 2),
            "sparse_top_score": round(max((hit.sparse_score for hit in self.sparse_hits), default=0.0), 3),
            "refusal_reason": self.refusal_reason,
            "timings_ms": {k: round(v, 2) for k, v in self.timings.items()},
            "sources": sorted({hit.source_file for hit in self.hits}),
        }


class HybridRetriever:
    """稠密 + 稀疏混合检索器。"""

    def __init__(
        self,
        *,
        max_unknown_ratio: float = 0.7,
        sparse_rescue_min_score: float = 8.0,
        sparse_rescue_max_unknown_ratio: float = 0.55,
        sparse_rescue_dense_ratio: float = 0.6,
        sparse_rescue_min_standout: float = 2.0,
        standout_min_samples: int = 5,
        oov_min_chunks: int = 6,
        vector_store: FaissVectorStore,
        chunk_store: ChunkStore,
        bm25: BM25Index,
        top_k: int = 5,
        fetch_k: int = 20,
        hybrid: bool = True,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dedupe_prefix_chars: int = 60,
        expand_neighbors: bool = True,
        max_context_hits: int = 8,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.bm25 = bm25
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)
        self.hybrid = hybrid
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.dedupe_prefix_chars = dedupe_prefix_chars
        self.expand_neighbors = expand_neighbors
        self.max_context_hits = max_context_hits
        self.mmr_lambda = mmr_lambda
        self.max_unknown_ratio = max_unknown_ratio
        self.sparse_rescue_min_score = sparse_rescue_min_score
        self.sparse_rescue_max_unknown_ratio = sparse_rescue_max_unknown_ratio
        self.sparse_rescue_dense_ratio = sparse_rescue_dense_ratio
        self.sparse_rescue_min_standout = sparse_rescue_min_standout
        # 语料过小时，分数分布/词表统计都不可靠，此时关闭对应判据（避免"一份小文档就永远拒答"）
        self.standout_min_samples = standout_min_samples
        self.oov_min_chunks = oov_min_chunks

    # ------------------------------------------------------------ 主入口
    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        mmr: bool = False,
    ) -> RetrievalResult:
        top_k = top_k or self.top_k
        timings: dict[str, float] = {}

        if len(self.chunk_store) == 0:
            raise KnowledgeBaseEmptyError()

        # ① 稠密召回
        with stopwatch() as sw:
            dense_hits = self.vector_store.search(
                question, k=self.fetch_k, fetch_k=self.fetch_k, mmr=mmr, mmr_lambda=self.mmr_lambda
            )
        timings["dense_ms"] = sw["ms"]

        # ② 稀疏召回
        sparse_hits: list[Hit] = []
        if self.hybrid and self.bm25.size:
            with stopwatch() as sw:
                for sparse in self.bm25.search(question, k=self.fetch_k):
                    record = self.chunk_store.get(sparse.chunk_id)
                    if record is None:
                        continue
                    sparse_hits.append(
                        Hit(
                            chunk_id=sparse.chunk_id,
                            text=record.text,
                            metadata=dict(record.metadata),
                            score=0.0,
                            sparse_score=sparse.score,
                            retrieval="sparse",
                        )
                    )
            timings["sparse_ms"] = sw["ms"]
            for rank, hit in enumerate(sparse_hits, start=1):
                hit.sparse_rank = rank

        # ③ 融合 + 去重 + 邻居扩展
        sparse_top_score = max((hit.sparse_score for hit in sparse_hits), default=0.0)
        dense_scores = sorted(hit.score for hit in dense_hits)
        dense_median = dense_scores[len(dense_scores) // 2] if dense_scores else 0.0
        standout_ratio = (
            max(dense_scores) / dense_median
            if (dense_median > 1e-9 and len(dense_scores) >= self.standout_min_samples)
            else -1.0  # -1 表示样本不足以估计分布，判据不生效
        )
        with stopwatch() as sw:
            fused = self._fuse(dense_hits, sparse_hits) if (self.hybrid and sparse_hits) else dense_hits
            threshold = (
                score_threshold
                if score_threshold is not None and score_threshold >= 0
                else float(getattr(self.vector_store.embeddings.info, "recommended_threshold", 0.30))
            )
            selected = [hit for hit in fused if hit.score >= threshold or hit.dense_rank is None]
            # 稠密分数是"这题知识库到底有没有答案"的可靠信号（BM25 字面命中容易误判）
            top_score = max((hit.score for hit in fused), default=0.0)
            unknown_ratio = self._unknown_term_ratio(question)
            refusal_reason = ""

            # 证据 ①：稠密语义相似度达标，且问题概念在知识库词表内
            dense_ok = top_score >= threshold and unknown_ratio <= self.max_unknown_ratio
            # 证据 ②：BM25 强命中且查询概念基本都在库内（字面救援，
            #         对"V3.2""192.168.1.102""36 小时"这类关键词问题尤其有效）
            sparse_ok = (
                bool(sparse_hits)
                and sparse_top_score >= self.sparse_rescue_min_score
                and unknown_ratio <= self.sparse_rescue_max_unknown_ratio
                # 还需稠密分不低于阈值的 60%：避免"关键词恰好出现但语义完全无关"的巧合
                and top_score >= threshold * self.sparse_rescue_dense_ratio
                # 且最优片段必须显著"突出"于本次检索的分数分布（否则只是整体噪声偏高）
                and (standout_ratio < 0 or standout_ratio >= self.sparse_rescue_min_standout)
            )
            answerable = dense_ok or sparse_ok
            if not answerable:
                if unknown_ratio > self.max_unknown_ratio:
                    refusal_reason = f"query_terms_unknown({unknown_ratio:.2f})"
                else:
                    refusal_reason = (
                        f"low_evidence(dense={top_score:.3f}<{threshold:.3f},"
                        f"sparse={sparse_top_score:.2f}<{self.sparse_rescue_min_score:.1f})"
                    )
            results = (selected if answerable else [])[:top_k]
            if answerable and self.expand_neighbors:
                results = self._expand_neighbors(results)
            results = self._dedupe(results)[: self.max_context_hits]
        timings["fuse_ms"] = sw["ms"]

        for rank, hit in enumerate(results, start=1):
            hit.fusion_score = hit.fusion_score or (0.0 if rank > 1 else hit.score)

        strategy = "hybrid_rrf" if (self.hybrid and sparse_hits) else ("dense_mmr" if mmr else "dense")
        result = RetrievalResult(
            hits=results,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            top_score=top_score,
            is_answerable=answerable,
            strategy=strategy,
            threshold=threshold,
            query_terms=len(set(re.split(r"\s+", question))) if question else 0,
            unknown_ratio=unknown_ratio,
            standout_ratio=standout_ratio,
            refusal_reason=refusal_reason,
            timings=timings,
        )
        logger.info(
            "检索完成: %s | top=%.4f (阈值 %.2f) 命中 %d 条 | %s",
            strategy,
            top_score,
            threshold,
            len(results),
            {k: round(v, 1) for k, v in timings.items()},
        )
        return result

    # ------------------------------------------------------------ 拒答判定
    def _unknown_term_ratio(self, question: str) -> float:
        """查询中"知识库里完全没有出现过"的内容词占比。

        直觉：如果一个问题的关键概念（火星、殖民地……）在整个知识库词表中都不存在，
        那么无论相似度算出来多少，都不该编答案 —— 这是拒答判定里最便宜、最有效的一票。
        """
        if len(self.chunk_store) < self.oov_min_chunks:
            return 0.0  # 片段太少，词表统计不可靠
        tokens = {
            token
            for token in tokenize_for_sparse(question)
            if len(token) > 1 or "\u4e00" <= token[0] <= "\u9fff"
        }
        if not tokens:
            return 0.0
        vocabulary = self.bm25_vocabulary()
        if not vocabulary:
            return 0.0
        unknown = sum(1 for token in tokens if token not in vocabulary)
        return unknown / len(tokens)

    def bm25_vocabulary(self) -> set[str]:
        """知识库词表（优先用 BM25 倒排，缓存结果避免重复构建集合）。"""
        cached = getattr(self, "_vocab_cache", None)
        if cached is not None and cached[0] == self.bm25.size:
            return cached[1]
        vocabulary = set(self.bm25._postings) if self.bm25.size else set()
        if not vocabulary and len(self.chunk_store):
            vocabulary = {
                token
                for record in self.chunk_store
                for token in set(tokenize_for_sparse(record.text))
            }
        self._vocab_cache = (self.bm25.size, vocabulary)
        return vocabulary

    # ------------------------------------------------------------ 融合
    def _fuse(self, dense_hits: list[Hit], sparse_hits: list[Hit]) -> list[Hit]:
        """RRF 融合：score = w_dense/(k+rank_dense) + w_sparse/(k+rank_sparse)。"""
        merged: dict[str, Hit] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            hit.dense_rank = rank
            hit.fusion_score = self.dense_weight / (self.rrf_k + rank)
            merged[hit.chunk_id] = hit
        for rank, hit in enumerate(sparse_hits, start=1):
            hit.sparse_rank = rank
            contribution = self.sparse_weight / (self.rrf_k + rank)
            if hit.chunk_id in merged:
                existing = merged[hit.chunk_id]
                existing.fusion_score += contribution
                existing.sparse_score = hit.sparse_score
                existing.retrieval = "hybrid"
            else:
                hit.retrieval = "sparse"
                hit.fusion_score = contribution
                merged[hit.chunk_id] = hit
        ordered = sorted(merged.values(), key=lambda h: h.fusion_score, reverse=True)
        return ordered

    # ------------------------------------------------------------ 后处理
    def _dedupe(self, hits: list[Hit]) -> list[Hit]:
        """近似重复去重：重叠窗口切片会产生高度相似的相邻块，重复喂给 LLM 只会浪费 token。"""
        seen_ids: set[str] = set()
        seen_prefix: set[str] = set()
        output: list[Hit] = []
        for hit in hits:
            if hit.chunk_id in seen_ids:
                continue
            signature = _PUNCT_RE.sub("", hit.text)[: self.dedupe_prefix_chars]
            if signature and signature in seen_prefix:
                continue
            seen_ids.add(hit.chunk_id)
            seen_prefix.add(signature)
            output.append(hit)
        return output

    def _expand_neighbors(self, hits: list[Hit], radius: int = 1) -> list[Hit]:
        """邻居扩展：把命中块的上下文邻居补进来，缓解"答案被切在块边界"的问题。"""
        expanded: list[Hit] = []
        seen: set[str] = set()
        for hit in hits:
            index = int(hit.metadata.get("chunk_index", -1) or -1)
            doc_id = str(hit.metadata.get("doc_id", ""))
            candidates = [hit]
            if index >= 0 and doc_id:
                for offset in range(1, radius + 1):
                    for neighbor_index in (index - offset, index + offset):
                        record = self.chunk_store.get(f"{doc_id}-{neighbor_index:04d}")
                        if record is None:
                            continue
                        candidates.append(
                            Hit(
                                chunk_id=record.chunk_id,
                                text=record.text,
                                metadata=dict(record.metadata),
                                score=hit.score * 0.98,  # 邻居分数略降，排序仍以主命中的顺序为准
                                sparse_score=hit.sparse_score,
                                fusion_score=hit.fusion_score * 0.98,
                                retrieval=f"{hit.retrieval}+neighbor",
                            )
                        )
            for candidate in candidates:
                if candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                expanded.append(candidate)
        return expanded
