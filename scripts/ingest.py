#!/usr/bin/env python
"""命令行入库工具。

用法：
    python -m scripts.ingest                      # 增量索引 docs/ 目录
    python -m scripts.ingest --dir ./my_docs      # 指定目录
    python -m scripts.ingest --reset              # 清空后全量重建（换 embedding 模型后必做）
    python -m scripts.ingest --file a.pdf         # 单个文件
    python -m scripts.ingest --status             # 只看当前知识库状态
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402


def build_pipeline(reset: bool = False, strict: bool = False) -> RagPipeline:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    pipeline = RagPipeline(settings)
    pipeline.bootstrap(strict=strict)
    if reset:
        pipeline.reset()
    return pipeline


def print_status(pipeline: RagPipeline) -> None:
    status = pipeline.status()
    print(json.dumps(
        {
            "知识库": status["knowledge_base"],
            "向量模型": status["embedding"],
            "大模型": status["llm"],
            "检索配置": status["retrieval"],
            "切片配置": status["chunking"],
            "支持格式": status["supported_types"],
        },
        ensure_ascii=False,
        indent=2,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 知识库入库工具")
    parser.add_argument("--dir", default=None, help="要索引的目录（默认 docs/）")
    parser.add_argument("--file", default=None, help="只索引单个文件")
    parser.add_argument("--reset", action="store_true", help="先清空知识库再全量索引")
    parser.add_argument("--status", action="store_true", help="仅打印当前状态")
    parser.add_argument("--force", action="store_true", help="内容未变化时也强制重新入库")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = build_pipeline(reset=args.reset)

    if args.status:
        print_status(pipeline)
        return 0

    started = time.perf_counter()
    if args.file:
        results = [pipeline.ingest_file(args.file, force=args.force)]
    else:
        directory = Path(args.dir) if args.dir else settings.docs_dir
        results = pipeline.ingest_directory(directory, force=args.force)

    print("\n================ 入库结果 ================")
    for result in results:
        flag = "跳过" if result.skipped else "完成"
        detail = result.reason or f"{result.chunks} 片段 / {result.chars} 字符 / {result.parser}"
        print(f"[{flag}] {result.source_file:32s} {detail}  ({result.elapsed_ms:.0f}ms)")
    stats = pipeline.chunking_stats().as_dict()
    print("\n---------------- 汇总 ----------------")
    print(json.dumps(
        {
            "耗时": f"{time.perf_counter() - started:.2f}s",
            "文档数": len(pipeline.registry),
            "片段数": len(pipeline.chunk_store),
            "向量数": pipeline.vector_store.size,
            "切片统计": stats,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
