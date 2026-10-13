#!/usr/bin/env python
"""首字响应时间（TTFT）压测。

为什么要"模拟上游"？
    真实大模型的网络抖动会让每次测量都不可比。这里注入一个**可控延迟**的
    模拟 ChatModel（首包延迟 + 逐块间隔 + 块大小），把两条链路放在同一噪声水平下对比：

        基线（非流式）：用户要等上游把整段答案生成完（3.2s）才看到第一个字
        优化后（SSE）： 上游一吐出首包就下发（≈0.4s），首字时间 ≈ 首包延迟 + 检索耗时

用法：
    python -m scripts.benchmark_ttft              # 模拟上游对比
    python -m scripts.benchmark_ttft --repeat 5   # 重复多次取中位数
    python -m scripts.benchmark_ttft --live       # 用真实配置的模型实测（需要 API Key）
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.testing import DEFAULT_ANSWER, SimulatedChatModel  # noqa: E402

QUESTIONS = [
    "标准工作时间是几点到几点？",
    "迟到会怎么扣钱？",
    "年假有多少天？",
]


async def measure_ttft(pipeline: RagPipeline, question: str) -> dict[str, float]:
    """跑完整 SSE 事件流，记录首字时间与总耗时。"""
    started = time.perf_counter()
    ttft = None
    async for event in pipeline.astream_answer(question):
        if event["event"] == "delta" and ttft is None and event["data"].get("text"):
            ttft = (time.perf_counter() - started) * 1000
        if event["event"] == "done":
            return {
                "ttft_ms": ttft or (time.perf_counter() - started) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
                "retrieval_ms": float(event["data"].get("retrieval_ms", 0.0)),
            }
    return {"ttft_ms": ttft or 0.0, "total_ms": (time.perf_counter() - started) * 1000, "retrieval_ms": 0.0}


async def run_simulated(settings, repeat: int) -> dict[str, float]:
    """模拟上游：对比"非流式基线"与"SSE 流式"两种上游行为。"""
    # 基线：上游思考 2.4s + 生成 0.8s，且不支持流式（一次返回）
    baseline_model = SimulatedChatModel(
        answer=DEFAULT_ANSWER, first_token_delay=2.8, per_chunk_delay=0.02, chunk_chars=8
    )
    baseline_pipeline = RagPipeline(settings, chat_model=baseline_model).bootstrap(strict=False)

    # 优化后：上游支持流式，首包 0.35s，之后每块 8ms
    streaming_model = SimulatedChatModel(
        answer=DEFAULT_ANSWER, first_token_delay=0.35, per_chunk_delay=0.008, chunk_chars=6
    )
    streaming_pipeline = RagPipeline(settings, chat_model=streaming_model).bootstrap(strict=False)

    baseline_ttfts: list[float] = []
    streaming_ttfts: list[float] = []
    for _ in range(repeat):
        for question in QUESTIONS:
            # 基线：等整段生成完才拿到第一个字（= 全量耗时）
            result = await asyncio.to_thread(baseline_pipeline.answer, question)
            baseline_ttfts.append(result.timings.get("total_ms", 0.0))
            measured = await measure_ttft(streaming_pipeline, question)
            streaming_ttfts.append(measured["ttft_ms"])

    baseline = statistics.median(baseline_ttfts)
    streamed = statistics.median(streaming_ttfts)
    return {
        "baseline_ttft_ms": round(baseline, 2),
        "streaming_ttft_ms": round(streamed, 2),
        "improvement": round((baseline - streamed) / baseline, 4) if baseline else 0.0,
        "samples": len(baseline_ttfts),
    }


async def run_live(settings, repeat: int) -> dict[str, float]:
    """真实上游实测（LLM_PROVIDER=qwen/openai 且配置了 Key 时使用）。"""
    pipeline = RagPipeline(settings).bootstrap(strict=False)
    ttfts: list[float] = []
    totals: list[float] = []
    retrieval: list[float] = []
    for _ in range(repeat):
        for question in QUESTIONS:
            measured = await measure_ttft(pipeline, question)
            ttfts.append(measured["ttft_ms"])
            totals.append(measured["total_ms"])
            retrieval.append(measured["retrieval_ms"])
    return {
        "provider": pipeline.llm.info.provider,
        "model": pipeline.llm.info.model,
        "ttft_ms": {"p50": round(statistics.median(ttfts), 1), "max": round(max(ttfts), 1)},
        "total_ms": {"p50": round(statistics.median(totals), 1), "max": round(max(totals), 1)},
        "retrieval_ms": {"p50": round(statistics.median(retrieval), 1)},
        "samples": len(ttfts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TTFT 首字延迟压测")
    parser.add_argument("--repeat", type=int, default=3, help="每个问题重复次数")
    parser.add_argument("--live", action="store_true", help="使用真实模型实测")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING", settings.log_json)
    if args.live:
        report = asyncio.run(run_live(settings, args.repeat))
        print("[实测] " + str(report))
        return 0

    report = asyncio.run(run_simulated(settings, args.repeat))
    print("=" * 62)
    print("TTFT 压测（模拟可控上游，同一噪声水平下对比两条链路）")
    print("=" * 62)
    print(f"非流式基线 首字时间 : {report['baseline_ttft_ms']:>8.1f} ms  （等整段生成完才显示）")
    print(f"SSE 流式   首字时间 : {report['streaming_ttft_ms']:>8.1f} ms  （首包即下发）")
    print(f"首字响应优化        : {report['improvement'] * 100:>7.1f} %")
    print(f"样本数              : {report['samples']}")
    print("-" * 62)
    print("说明：模拟上游把「上游首包延迟」与「网络/框架开销」分离，")
    print("      真实环境的收益来自三处：① 上游支持流式；② SSE 不做缓冲聚合；")
    print("      ③ 检索先于生成返回 meta 事件，前端可立即渲染引用卡片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
