"""进程内埋点与统计（无外部依赖，`GET /api/v1/metrics` 直接暴露）。

关注四个指标族（对齐简历里的性能目标）：
    - 可用性：请求数、错误数、拒答数；
    - 延迟：TTFT（首字时间）、检索耗时、端到端耗时的 P50/P90/P95；
    - 检索质量：平均 top 相似度、平均命中片段数、降级次数；
    - 入库：文档数、切片数、解析失败数。

说明：单进程内存统计，够用且零依赖；多副本部署时把它换成 Prometheus
Counter/Histogram 即可（接口形态保持一致），属于明确的演进项。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """滑动窗口延迟统计。"""

    name: str
    max_samples: int = 2000
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=2000), repr=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.max_samples)

    def record(self, milliseconds: float) -> None:
        if milliseconds >= 0:
            self._samples.append(float(milliseconds))

    @property
    def count(self) -> int:
        return len(self._samples)

    def percentile(self, ratio: float) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
        return round(ordered[index], 2)

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {"count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "count": len(self._samples),
            "avg": round(sum(self._samples) / len(self._samples), 2),
            "p50": self.percentile(0.5),
            "p90": self.percentile(0.9),
            "p95": self.percentile(0.95),
            "max": round(max(self._samples), 2),
        }


class Metrics:
    """线程安全的指标收集器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.ttft = LatencyStats("ttft_ms")
        self.total_latency = LatencyStats("total_ms")
        self.retrieval_latency = LatencyStats("retrieval_ms")
        self.embedding_latency = LatencyStats("embedding_ms")

        self.requests = 0
        self.chat_requests = 0
        self.stream_requests = 0
        self.errors = 0
        self.refusals = 0
        self.empty_kb_hits = 0
        self.llm_fallbacks = 0
        self.ingest_jobs = 0
        self.documents_indexed = 0
        self.chunks_indexed = 0
        self.chunks_deleted = 0
        self.upload_bytes = 0
        self.top_scores: deque[float] = deque(maxlen=500)

    # ---------------- 记录 ----------------
    def incr(self, field_name: str, value: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name, 0) + value)

    def observe_query(
        self,
        *,
        ttft_ms: float | None = None,
        total_ms: float | None = None,
        retrieval_ms: float | None = None,
        top_score: float | None = None,
        refused: bool = False,
        fallback: bool = False,
        stream: bool = True,
    ) -> None:
        with self._lock:
            self.chat_requests += 1
            if stream:
                self.stream_requests += 1
            if ttft_ms is not None:
                self.ttft.record(ttft_ms)
            if total_ms is not None:
                self.total_latency.record(total_ms)
            if retrieval_ms is not None:
                self.retrieval_latency.record(retrieval_ms)
            if top_score is not None:
                self.top_scores.append(float(top_score))
            if refused:
                self.refusals += 1
            if fallback:
                self.llm_fallbacks += 1

    # ---------------- 输出 ----------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            scores = list(self.top_scores)
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests": {
                    "total": self.requests,
                    "chat": self.chat_requests,
                    "stream": self.stream_requests,
                    "errors": self.errors,
                    "refusals": self.refusals,
                    "empty_kb": self.empty_kb_hits,
                    "llm_fallbacks": self.llm_fallbacks,
                },
                "ingest": {
                    "jobs": self.ingest_jobs,
                    "documents": self.documents_indexed,
                    "chunks": self.chunks_indexed,
                    "chunks_deleted": self.chunks_deleted,
                    "upload_bytes": self.upload_bytes,
                },
                "latency_ms": {
                    "ttft": self.ttft.summary(),
                    "total": self.total_latency.summary(),
                    "retrieval": self.retrieval_latency.summary(),
                    "embedding": self.embedding_latency.summary(),
                },
                "retrieval_quality": {
                    "avg_top_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                    "samples": len(scores),
                    "refusal_rate": round(self.refusals / self.chat_requests, 3) if self.chat_requests else 0.0,
                },
            }


metrics = Metrics()


def summarize(values: Iterable[float]) -> dict[str, float]:
    """给脚本用的小工具：百分比统计。"""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 2),
        "p50": round(ordered[len(ordered) // 2], 2),
        "p90": round(ordered[int(len(ordered) * 0.9) - 1 if len(ordered) > 2 else -1], 2),
        "max": round(ordered[-1], 2),
    }
