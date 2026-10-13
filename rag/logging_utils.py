"""日志与耗时埋点工具。

设计目标：
1. 统一日志格式，生产环境可切 JSON（方便 ELK）；
2. 提供 `stopwatch()` 上下文管理器，用于采集检索 / 首字 / 全量耗时；
3. 提供 estimate_tokens —— 中英文混排的 token 估算，用于上下文预算控制。
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class JsonFormatter(logging.Formatter):
    """极简结构化日志：时间 / 级别 / logger / 消息 / 附加字段。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """初始化根 logger（幂等，可重复调用）。"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)
    # 第三方库降噪
    for noisy in ("httpx", "httpcore", "urllib3", "faiss", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def stopwatch() -> Iterator[dict[str, float]]:
    """with stopwatch() as sw: ...; sw['ms'] 为耗时（毫秒）。

    用法：
        with stopwatch() as sw:
            do_something()
        logger.info("cost=%.1fms", sw["ms"])
    """
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["ms"] = (time.perf_counter() - start) * 1000


def estimate_tokens(text: str) -> int:
    """中英文混排 token 估算：CJK 字符 ≈ 1 token，其余按 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = max(0, len(text) - cjk)
    return int(cjk + rest / 4) + 1


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "…") -> str:
    """按 token 预算截断文本（保头部，尽量在句子边界截断）。"""
    if max_tokens <= 0 or estimate_tokens(text) <= max_tokens:
        return text
    budget_chars = max(16, int(max_tokens * 1.6))
    head = text[:budget_chars]
    for boundary in ("。", "！", "？", "；", "\n", ".", "!", "?"):
        idx = head.rfind(boundary)
        if idx >= budget_chars * 0.5:
            return head[: idx + 1] + suffix
    return head + suffix


def tokenize_for_sparse(text: str) -> list[str]:
    """轻量分词（用于 BM25 / 本地 embedding）：

    - 英文/数字：小写词
    - 中文：单字 + 相邻双字（bigram），既保证召回又不需要 jieba 依赖
    """
    tokens: list[str] = [w.lower() for w in _WORD_RE.findall(text)]
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]+", text)
    for run in cjk_runs:
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens
