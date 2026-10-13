#!/usr/bin/env python
"""离线评测脚本：检索命中率 / 答案要点覆盖率 / 拒答准确率 / 时延。

用法：
    python -m scripts.evaluate                 # 用内置私域测试集评测
    python -m scripts.evaluate --set other.json
    python -m scripts.evaluate --tune           # 网格搜索最优阈值（校准拒答判定）
    python -m scripts.evaluate --stream         # 顺带测 SSE 流式链路的 TTFT
    python -m scripts.evaluate --json report.json

指标定义（面试要能一句话说清）：
    检索命中率 hit@k   ：期望文档是否出现在检索结果里（召回能力）
    答案要点覆盖率     ：答案里命中了多少条关键信息（回答质量，能识别"答非所问"）
    拒答准确率         ：域外问题是否被正确拒答（幻觉防线，宁可拒答不可编造）
    可答准确率         ：域内问题是否被判为"可答"（漏拒率）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

DEFAULT_SET = Path(__file__).resolve().parents[1] / "tests" / "data" / "eval_set.json"


def normalize(text: str) -> str:
    """归一化：去掉空白/标点差异，全角转半角，便于关键信息匹配。"""
    text = text.lower().replace("　", " ")
    table = str.maketrans("０１２３４５６７８９％．（）", "0123456789%.()")
    text = text.translate(table)
    return re.sub(r"[\s,，。；;、:：!！?？\"'`~\-_/\\()\[\]{}]+", "", text)


def evaluate(pipeline: RagPipeline, eval_set: dict[str, Any], *, stream: bool = False) -> dict[str, Any]:
    in_scope = eval_set["in_scope"]
    out_of_scope = eval_set["out_of_scope"]

    retrieval_hits = 0
    answerable = 0
    keyword_ok = 0
    keyword_total = 0
    latencies: list[float] = []
    ttfts: list[float] = []
    details: list[dict[str, Any]] = []

    for item in in_scope:
        question = item["question"]
        result = pipeline.answer(question)
        gold_sources = set(item.get("sources") or [])
        hit_sources = {ref["source"] for ref in result.references}
        retrieval_hit = bool(gold_sources & hit_sources) if gold_sources else bool(result.references)
        retrieval_hits += int(retrieval_hit)
        answerable += int(not result.refused)

        normalized_answer = normalize(result.answer)
        terms = item.get("key_terms") or []
        matched = [term for term in terms if normalize(term) in normalized_answer]
        keyword_total += len(terms)
        keyword_ok += len(matched)
        latencies.append(result.timings.get("total_ms", 0.0))

        if stream:
            ttfts.append(asyncio.run(_measure_ttft(pipeline, question)))

        details.append({
            "question": question,
            "answerable": not result.refused,
            "retrieval_hit": retrieval_hit,
            "expected": terms,
            "matched": matched,
            "top_score": result.retrieval.get("top_score"),
            "hits": result.retrieval.get("hits"),
            "grounded": result.grounded,
            "ttft_ms": round(ttfts[-1], 2) if (stream and ttfts) else None,
            "answer_preview": result.answer[:120].replace("\n", " "),
        })

    refusals = 0
    hard_negative_failures: list[str] = []
    for item in out_of_scope:
        result = pipeline.answer(item["question"])
        refused = result.refused
        refusals += int(refused)
        if not refused:
            hard_negative_failures.append(item["question"])
        details.append({
            "question": item["question"],
            "out_of_scope": True,
            "refused": refused,
            "hard": bool(item.get("hard")),
            "top_score": result.retrieval.get("top_score"),
            "reason": result.retrieval.get("refusal_reason"),
            "answer_preview": result.answer[:120].replace("\n", " "),
        })

    total_in = len(in_scope)
    total_out = len(out_of_scope)
    return {
        "系统": {
            "embedding": pipeline.embeddings.info.as_dict(),
            "llm": pipeline.llm.info.as_dict(),
            "hybrid": pipeline.settings.hybrid_enabled,
            "score_threshold": round(pipeline.settings.effective_score_threshold, 4),
            "chunks": len(pipeline.chunk_store),
        },
        "检索命中率 hit@k": round(retrieval_hits / total_in, 4) if total_in else 0.0,
        "答案要点覆盖率": round(keyword_ok / keyword_total, 4) if keyword_total else 0.0,
        "可答准确率（域内）": round(answerable / total_in, 4) if total_in else 0.0,
        "拒答准确率（域外）": round(refusals / total_out, 4) if total_out else 0.0,
        "综合准确率": round((answerable + refusals) / (total_in + total_out), 4) if (total_in + total_out) else 0.0,
        "整体判定准确率（含困难负例）": round(
            (retrieval_hits + answerable + refusals) / (total_in * 2 + total_out), 4
        ) if (total_in * 2 + total_out) else 0.0,
        "误答的域外问题": hard_negative_failures,
        "时延": {
            "answer_ms": _summary(latencies),
            "ttft_ms": _summary(ttfts) if ttfts else None,
        },
        "明细": details,
    }


async def _measure_ttft(pipeline: RagPipeline, question: str) -> float:
    """走完整 SSE 事件流，测首字时间（TTFT）。"""
    started = time.perf_counter()
    async for event in pipeline.astream_answer(question):
        if event["event"] == "delta" and event["data"]["text"]:
            return (time.perf_counter() - started) * 1000
    return (time.perf_counter() - started) * 1000


def _summary(values: list[float]) -> dict[str, float] | None:
    clean = [v for v in values if v]
    if not clean:
        return None
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "avg": round(statistics.fmean(ordered), 1),
        "p50": round(ordered[len(ordered) // 2], 1),
        "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 1),
        "max": round(ordered[-1], 1),
    }


def tune(pipeline: RagPipeline, eval_set: dict[str, Any]) -> list[dict[str, Any]]:
    """网格搜索拒答判定的两个关键阈值，输出平衡准确率最高的一组参数。

    这是"用数据而不是拍脑袋定阈值"的做法：私域测试集 → 扫阈值 → 选最优 → 写回配置。
    """
    candidates = [0.10, 0.14, 0.18, 0.20, 0.22, 0.26, 0.30, 0.35, 0.40]
    rescue = [0.0, 6.0, 8.0, 10.0, 12.0]
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        for rescue_score in rescue:
            pipeline.retriever.sparse_rescue_min_score = rescue_score
            answered = refused = 0
            for item in eval_set["in_scope"]:
                result = pipeline.retrieve(item["question"], threshold=threshold)
                answered += int(result.is_answerable)
            for item in eval_set["out_of_scope"]:
                result = pipeline.retrieve(item["question"], threshold=threshold)
                refused += int(not result.is_answerable)
            total_in, total_out = len(eval_set["in_scope"]), len(eval_set["out_of_scope"])
            accuracy = (answered + refused) / (total_in + total_out)
            rows.append({
                "阈值": threshold,
                "BM25救援阈值": rescue_score,
                "域内可答": f"{answered}/{total_in}",
                "域外拒答": f"{refused}/{total_out}",
                "准确率": round(accuracy, 4),
            })
    rows.sort(key=lambda row: row["准确率"], reverse=True)
    return rows[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 系统离线评测")
    parser.add_argument("--set", default=str(DEFAULT_SET), help="评测集 JSON 路径")
    parser.add_argument("--json", default=None, help="把报告写入 JSON 文件")
    parser.add_argument("--stream", action="store_true", help="额外测量流式链路的 TTFT")
    parser.add_argument("--tune", action="store_true", help="网格搜索最优拒答阈值")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING", settings.log_json)
    pipeline = RagPipeline(settings)
    pipeline.bootstrap(strict=False)
    if len(pipeline.chunk_store) == 0:
        print("知识库为空，正在索引 docs/ ……")
        pipeline.ingest_directory(settings.docs_dir)

    eval_set = json.loads(Path(args.set).read_text(encoding="utf-8"))
    print(f"\n=== 评测集：{eval_set.get('name', args.set)} ===")
    report = evaluate(pipeline, eval_set, stream=args.stream)

    print(json.dumps({k: v for k, v in report.items() if k != "明细"}, ensure_ascii=False, indent=2))
    print("\n--- 明细（前 12 条）---")
    for row in report["明细"][:12]:
        if row.get("out_of_scope"):
            print(f"[域外] {'拒答✓' if row['refused'] else '误答✗'} {row['question']} (top={row['top_score']})")
        else:
            print(
                f"[域内] {'可答✓' if row['answerable'] else '漏拒✗'} {row['question']} "
                f"命中={row['retrieval_hit']} 要点={row['matched']} top={row['top_score']}"
            )

    if args.tune:
        print("\n=== 阈值网格搜索（Top 12）===")
        for row in tune(pipeline, eval_set):
            print(json.dumps(row, ensure_ascii=False))

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
