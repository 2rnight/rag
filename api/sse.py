"""SSE（Server-Sent Events）工具。

协议格式：
    event: delta
    data: {"text": "年假为"}

    : ping            <- 心跳注释行，客户端自动忽略

工程细节（都是真踩过的坑）：
1. `X-Accel-Buffering: no` + `Cache-Control: no-cache`：
   不然 Nginx / 各类反代会把流"攒"起来，用户看到的是"一次性全出来"，TTFT 白优化；
2. **心跳**：长连接空闲超过网关读超时（常见 60s）会被掐断，
   所以即使没有 token（比如模型在长思考）也要定期发注释行；
3. **首字即刷**：生成器 `yield` 之后由 ASGI 立刻写入，不做任何缓冲聚合；
4. 客户端断开时不抛异常，而是让生成器被回收（GeneratorExit/CancelledError），
   在 finally 里取消后台任务，避免上游连接泄漏。
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from rag.logging_utils import get_logger

logger = get_logger(__name__)

SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证逐字下发
    "Content-Encoding": "identity",
}


def format_event(event: str, data: dict[str, Any] | str) -> str:
    """序列化一个 SSE 事件。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_with_heartbeat(
    source: AsyncIterator[dict[str, Any]],
    *,
    heartbeat: float = 15.0,
) -> AsyncIterator[str]:
    """把业务事件流包装成 SSE 文本流，并在空闲时插入心跳。

    业务事件格式：{"event": "meta|delta|done|error", "data": {...}}
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=64)

    async def produce() -> None:
        try:
            async for event in source:
                await queue.put(event)
        except Exception as exc:  # noqa: BLE001 - 错误也要以事件形式通知前端
            logger.exception("SSE 生产端异常: %s", exc)
            await queue.put({"event": "error", "data": {"message": f"服务端异常: {exc}"}})
        finally:
            await queue.put(None)

    task = asyncio.create_task(produce())
    try:
        yield format_event("open", {"heartbeat": heartbeat})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield ": ping\n\n"  # 心跳：保持连接活跃
                continue
            if event is None:
                break
            yield format_event(str(event.get("event", "message")), event.get("data") or {})
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
