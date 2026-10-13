"""测试与压测用的模拟组件（不参与生产链路）。

为什么需要它？
    想量化"SSE 流式把 TTFT 降了 80%"，就必须有一个**可控上游**：
    真实大模型的网络抖动会让每次测量结果不可比。这里提供一个可配置的
    模拟 ChatModel（首包延迟 + 每块间隔 + 块大小），压测/单测都能复现。
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

DEFAULT_ANSWER = (
    "根据知识库资料：标准工作时间为周一至周五 9:00-12:00、13:30-18:00，"
    "核心工作时间 10:00-16:00，员工可在 7:30-10:00 之间弹性到岗 [1]。"
    "迟到 30 分钟以内每次扣 50 元，30-60 分钟扣 100 元，超过 60 分钟按旷工半天处理 [2]。"
)


class SimulatedChatModel(BaseChatModel):
    """可配置延迟的模拟大模型。

    参数：
        answer:           模拟返回的答案文本
        first_token_delay: 首 token 延迟（秒），模拟"模型思考 + 网络"耗时
        per_chunk_delay:   每块之间的间隔（秒），模拟逐字生成速度
        chunk_chars:       每块字符数
        fail:              置为 True 时直接抛错，用于验证降级链路
    """

    model_name: str = "simulated-llm"
    answer: str = DEFAULT_ANSWER
    first_token_delay: float = 0.0
    per_chunk_delay: float = 0.01
    chunk_chars: int = 6
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "simulated"

    # ---------------- 非流式：一次性返回（模拟"不做流式"的基线） ----------------
    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.fail:
            raise RuntimeError("simulated upstream failure")
        time.sleep(self.first_token_delay + self.per_chunk_delay * self._pieces())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])

    # ---------------- 流式 ----------------
    def _pieces(self) -> int:
        return max(1, len(self.answer) // max(1, self.chunk_chars) + 1)

    def _stream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.fail:
            raise RuntimeError("simulated upstream failure")
        time.sleep(self.first_token_delay)
        for start in range(0, len(self.answer), self.chunk_chars):
            piece = self.answer[start : start + self.chunk_chars]
            time.sleep(self.per_chunk_delay)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    async def _astream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.fail:
            raise RuntimeError("simulated upstream failure")
        await asyncio.sleep(self.first_token_delay)
        for start in range(0, len(self.answer), self.chunk_chars):
            piece = self.answer[start : start + self.chunk_chars]
            await asyncio.sleep(self.per_chunk_delay)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def simulate_legacy_ttft(curriculum_delay: float = 3.2) -> float:
    """非流式基线：用户要等上游把整段答案生成完才看到第一个字。"""
    return curriculum_delay
