"""统一命令行入口。

    python main.py serve                 # 启动 API 服务（等价 uvicorn app:app）
    python main.py ingest [--reset]      # 索引 docs/ 目录
    python main.py eval [--tune]         # 跑私域测试集评测
    python main.py bench [--live]        # TTFT 压测
    python main.py status                # 查看知识库与配置
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def _run_module(module: str, extra: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", module, *extra], cwd=str(ROOT))


def cmd_ingest(args: argparse.Namespace) -> int:
    extra: list[str] = []
    if args.reset:
        extra.append("--reset")
    if args.directory:
        extra += ["--dir", args.directory]
    return _run_module("scripts.ingest", extra)


def cmd_eval(args: argparse.Namespace) -> int:
    extra = ["--tune"] if args.tune else []
    if args.stream:
        extra.append("--stream")
    return _run_module("scripts.evaluate", extra)


def cmd_bench(args: argparse.Namespace) -> int:
    extra = ["--live"] if args.live else []
    extra += ["--repeat", str(args.repeat)]
    return _run_module("scripts.benchmark_ttft", extra)


def cmd_status(args: argparse.Namespace) -> int:
    return _run_module("scripts.ingest", ["--status"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 知识库问答系统 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 API 服务")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    ingest = sub.add_parser("ingest", help="索引文档")
    ingest.add_argument("--reset", action="store_true", help="清空后全量重建")
    ingest.add_argument("--directory", default=None, help="要索引的目录（默认 docs/）")
    ingest.set_defaults(func=cmd_ingest)

    evaluate = sub.add_parser("eval", help="私域测试集评测")
    evaluate.add_argument("--tune", action="store_true", help="网格搜索最优阈值")
    evaluate.add_argument("--stream", action="store_true", help="同时测量 TTFT")
    evaluate.set_defaults(func=cmd_eval)

    bench = sub.add_parser("bench", help="TTFT 压测")
    bench.add_argument("--live", action="store_true", help="使用真实模型实测")
    bench.add_argument("--repeat", type=int, default=3)
    bench.set_defaults(func=cmd_bench)

    status = sub.add_parser("status", help="查看知识库状态")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
