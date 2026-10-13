"""大模型统一接入层。

三件事（面试可讲）：

1. **统一接口**：Qwen(DashScope OpenAI 兼容模式) 与 OpenAI 都通过 LangChain 的
   `BaseChatModel` 接入，业务层只认 `ainvoke/astream`，换模型只改配置；
   LangChain 的回调体系让"首 token 时间 / token 用量"这类埋点可以直接挂在链路上。
2. **首字（TTFT）与稳定性**：
   - 流式：`astream` 逐块吐出，配合 API 层 SSE 立刻下发给浏览器；
   - **首 token 超时**：若上游 N 秒内一个字都没吐出来，直接切兜底，避免用户干等；
   - **降级策略**：上游 4xx/5xx/限流/断流时自动切"抽取式兜底模型"，接口永不 5xx。
3. **答案校验（反幻觉第二道闸）**：生成后做引用标注检查 + 答案/上下文
   bigram 覆盖率检查，判断答案是否真的"落地"在检索资料上；不达标时给出
   低置信度提示或直接替换为抽取式答案（strict 模式）。
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .exceptions import LLMError
from .logging_utils import get_logger, tokenize_for_sparse
from .prompts import extract_context_from_messages, extract_question_from_messages

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_HEADER_RE = re.compile(r"^\[(\d{1,2})\]\s*来源[:：]\s*(.+)$")
_SENTENCE_RE = re.compile(r"[^。！？；;!?\n]+[。！？；;!?]?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_PUNCT_RE = re.compile(r"[\s，。！？；：、“”‘’（）()\[\]【】,.!?;:\"'`~@#$%^&*\-_=+|\\/<>]+")


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class LLMInfo:
    provider: str
    model: str
    is_fallback: bool = False
    temperature: float = 0.0
    streaming: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "is_fallback": self.is_fallback,
            "temperature": self.temperature,
            "streaming": self.streaming,
        }


@dataclass(slots=True)
class StreamDelta:
    """流式输出的一个增量块。"""

    text: str
    fallback: bool = False
    index: int = 0


@dataclass(slots=True)
class GroundingCheck:
    """答案落地性校验结果。"""

    ok: bool
    citations: list[int]
    coverage: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "citations": self.citations,
            "coverage": round(self.coverage, 3),
            "reason": self.reason,
        }


def _extract_numbers(text: str) -> list[float]:
    """提取文本中的数值（支持"1.5倍""36 小时""1000-5000元"这类写法）。"""
    values: list[float] = []
    for raw in _NUMBER_RE.findall(text):
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _split_sentences(body: str) -> list[str]:
    """按句子/条目切分，过滤掉过短片段（导航噪声、空行）。"""
    return [piece.strip(" -·•\t") for piece in _SENTENCE_RE.findall(body) if len(piece.strip()) >= 8]


# ---------------------------------------------------------------- 抽取式兜底模型


class ExtractiveChatModel(BaseChatModel):
    """抽取式兜底 ChatModel：不调用任何外部 API。

    行为：从 Prompt 的 <参考资料> 里按"与问题的词元重合度"挑出最相关的若干句子，
    带引用编号原样组织成答案。**零幻觉**（每句话都能在资料里找到原文），
    在断网 / 无 Key / 上游故障 / CI 测试场景下保证系统仍可用、可演示、可评测。
    """

    model_name: str = "extractive-local"
    max_sentences: int = 5
    refusal_message: str = "根据当前知识库中的资料，我未找到能够回答该问题的内容。"
    stream_delay: float = 0.004

    @property
    def _llm_type(self) -> str:
        return "extractive-local"

    # ---------------- 生成 ----------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        answer = self.compose(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        answer = self.compose(messages)
        for position in range(0, len(answer), 3):
            piece = answer[position : position + 3]
            if self.stream_delay:
                time.sleep(self.stream_delay)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        answer = self.compose(messages)
        for position in range(0, len(answer), 3):
            piece = answer[position : position + 3]
            if self.stream_delay:
                await asyncio.sleep(self.stream_delay)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    # ---------------- 抽取逻辑 ----------------
    def compose(self, messages: Sequence[BaseMessage]) -> str:
        """从 <参考资料> 中抽取出最相关的句子组成答案。

        排序信号（按重要性）：
          1. 与问题的词元重合度（bigram 权重高于单字）；
          2. 数值亲和度：问题里有数字时，优先选"数值区间能覆盖该数字"的句子
             （例如问「报销 6000 元谁审批」→ 命中「5000 元以上：VP 审批」而不是「1000-5000 元」）；
          3. 数字/时间类线索：问题在问数量时，优先带数字或"次年/每月/每季度"的句子。
        """
        context = extract_context_from_messages(messages)
        question = extract_question_from_messages(messages)
        if not context.strip():
            return self.refusal_message

        blocks = self._parse_blocks(context)
        if not blocks:
            return self.refusal_message

        question_tokens = {tok for tok in tokenize_for_sparse(question) if tok.strip()}
        question_numbers = _extract_numbers(question)
        numeric_question = bool(
            question_numbers
            or re.search(r"多少|几|多久|多长时间|什么时候|几点|上限|标准|比例|费用|时间", question)
        )

        scored_blocks: list[tuple[float, int, str, list[tuple[float, str]]]] = []
        for number, _source, body in blocks:
            sentences = _split_sentences(body)
            scored: list[tuple[float, str]] = []
            for sentence in sentences:
                score = self._sentence_score(sentence, question_tokens, question_numbers, numeric_question)
                if score > 0:
                    scored.append((score, sentence))
            if not scored:
                continue
            block_score = sum(score for score, _ in sorted(scored, key=lambda item: item[0], reverse=True)[:2])
            scored_blocks.append((block_score, number, _source, scored))

        if not scored_blocks:
            return self.refusal_message

        scored_blocks.sort(key=lambda item: item[0], reverse=True)
        best_block_score = scored_blocks[0][0]
        # 只保留"有分量"的块：低于最优块 40% 的块往往是噪声（语气词偶然命中）
        kept_blocks = [block for block in scored_blocks if block[0] >= 0.4 * best_block_score][:2]

        candidates: list[tuple[float, int, str]] = []
        for _, number, _source, scored in kept_blocks:
            for score, sentence in scored:
                candidates.append((score, number, sentence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        picked: list[tuple[float, int, str]] = []
        seen_signatures: set[str] = set()
        for score, number, sentence in candidates:
            signature = _PUNCT_RE.sub("", sentence)[:24]
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            picked.append((score, number, sentence))
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            return self.refusal_message

        # 同一来源内恢复原文顺序，读起来更连贯
        picked.sort(key=lambda item: (item[1], -item[0]))

        lines = ["根据知识库资料，检索到以下相关内容："]
        for _, number, sentence in picked:
            shortened = sentence if len(sentence) <= 240 else sentence[:240] + "…"
            lines.append(f"- {shortened} [{number}]")
        lines.append("")
        lines.append(
            "（当前为离线抽取式回答：未配置大模型 API Key 或上游不可用，"
            "以上内容摘自知识库原文，未经过模型润色。）"
        )
        return "\n".join(lines)

    @staticmethod
    def _sentence_score(
        sentence: str,
        question_tokens: set[str],
        question_numbers: list[float],
        numeric_question: bool,
    ) -> float:
        tokens = set(tokenize_for_sparse(sentence))
        if not tokens:
            return 0.0
        overlap = question_tokens & tokens
        if not overlap:
            return 0.0
        weighted = sum(1.0 if len(token) > 1 else 0.4 for token in overlap)
        length_penalty = 1.0 / (1.0 + max(0, len(sentence) - 160) / 400)
        score = weighted * length_penalty

        sentence_numbers = _extract_numbers(sentence)
        if question_numbers and sentence_numbers:
            # 数值亲和：句子里的数值 ≤ 问题数值时给奖励（更可能是"命中该档位"的条款）
            lower = [value for value in sentence_numbers if value <= max(question_numbers)]
            if lower:
                ratio = max(lower) / max(question_numbers)
                score += 1.2 * ratio
        elif numeric_question and re.search(r"\d|次年|每月|每季度|每年|每周|小时|分钟", sentence):
            # 问题在问数量/时间，带数字或周期性表述的句子更可能是答案所在
            score += 1.5
        return score

    @staticmethod
    def _parse_blocks(context: str) -> list[tuple[int, str, str]]:
        blocks: list[tuple[int, str, str]] = []
        current: tuple[int, str, list[str]] | None = None
        for line in context.split("\n"):
            match = _HEADER_RE.match(line.strip())
            if match:
                if current:
                    blocks.append((current[0], current[1], "\n".join(current[2])))
                current = (int(match.group(1)), match.group(2).split("|")[0].replace("来源:", "").strip(), [])
            elif current:
                current[2].append(line)
        if current:
            blocks.append((current[0], current[1], "\n".join(current[2])))
        return blocks


# ---------------------------------------------------------------- 客户端

def _chunk_to_text(chunk: Any) -> str:
    """从 LangChain 的 AIMessageChunk 中提取纯文本（兼容 content blocks 形式）。"""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


class LLMClient:
    """大模型客户端：主模型 + 抽取式兜底 + 首字超时 + 答案校验。"""

    def __init__(
        self,
        *,
        primary: BaseChatModel | None,
        provider: str,
        model: str,
        temperature: float = 0.1,
        first_token_timeout: float = 20.0,
        fallback: BaseChatModel | None = None,
        refusal_message: str = "根据当前知识库中的资料，我未找到能够回答该问题的内容。",
    ) -> None:
        self._primary = primary
        self._fallback = fallback or ExtractiveChatModel(refusal_message=refusal_message)
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self.first_token_timeout = first_token_timeout
        self.last_error: str | None = None
        self.used_fallback = primary is None

    # ---------------- 元信息 ----------------
    @property
    def info(self) -> LLMInfo:
        return LLMInfo(
            provider=self._provider,
            model=self._model,
            is_fallback=self._primary is None,
            temperature=self._temperature,
            streaming=True,
        )

    @property
    def fallback_model(self) -> BaseChatModel:
        return self._fallback

    # ---------------- 调用 ----------------
    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        try:
            response = self._primary.invoke(list(messages)) if self._primary else self._fallback.invoke(list(messages))
        except Exception as exc:  # noqa: BLE001 - 兜底到抽取式模型
            self.last_error = str(exc)
            logger.warning("主模型调用失败，降级到抽取式兜底: %s", exc)
            self.used_fallback = True
            response = self._fallback.invoke(list(messages))
        return _chunk_to_text(response)

    async def astream(self, messages: Sequence[BaseMessage]) -> AsyncIterator[StreamDelta]:
        """流式生成，带首字超时与降级。"""
        self.last_error = None
        messages = list(messages)
        if self._primary is None:
            self.used_fallback = True
            async for delta in self._stream_model(self._fallback, messages, fallback=True):
                yield delta
            return

        emitted = ""
        index = 0
        try:
            stream = self._primary.astream(messages)
            iterator = stream.__aiter__()
            first = True
            while True:
                try:
                    if first:
                        chunk = await asyncio.wait_for(iterator.__anext__(), timeout=self.first_token_timeout)
                        first = False
                    else:
                        chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f"首字响应超时（>{self.first_token_timeout}s）") from exc
                text = _chunk_to_text(chunk)
                if not text:
                    continue
                emitted += text
                yield StreamDelta(text=text, fallback=False, index=index)
                index += 1
            if not emitted:  # 上游返回空内容 -> 兜底
                raise LLMError("模型返回内容为空")
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            self.used_fallback = True
            logger.warning("主模型流式失败（已产出 %d 字），切换抽取式兜底: %s", len(emitted), exc)
            if emitted:
                yield StreamDelta(text="\n\n（生成中断，以下为知识库原文摘录）\n", fallback=True, index=index)
                index += 1
            async for delta in self._stream_model(self._fallback, messages, fallback=True, start_index=index):
                yield delta

    async def _stream_model(
        self, model: BaseChatModel, messages: list[BaseMessage], *, fallback: bool, start_index: int = 0
    ) -> AsyncIterator[StreamDelta]:
        index = start_index
        try:
            async for chunk in model.astream(messages):
                text = _chunk_to_text(chunk)
                if not text:
                    continue
                yield StreamDelta(text=text, fallback=fallback, index=index)
                index += 1
        except Exception as exc:  # noqa: BLE001 - 连兜底都失败时给用户明确提示
            logger.error("兜底模型同样失败: %s", exc)
            yield StreamDelta(
                text=f"\n\n[服务异常] 生成失败：{exc}。请稍后重试或检查模型配置。",
                fallback=True,
                index=index,
            )


# ---------------------------------------------------------------- 工厂


def build_llm_client(settings) -> LLMClient:
    """LLM 工厂：按配置构造主模型；缺 Key/装不上依赖时使用抽取式兜底。"""
    provider = settings.llm_provider
    if provider == "auto":
        provider = "qwen" if settings.dashscope_api_key else ("openai" if settings.openai_api_key else "local")

    primary: BaseChatModel | None = None
    model_name = settings.llm_model
    try:
        if provider == "qwen":
            from langchain_openai import ChatOpenAI

            primary = ChatOpenAI(
                model=model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
                streaming=True,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            primary = ChatOpenAI(
                model=model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                streaming=True,
            )
    except Exception as exc:  # noqa: BLE001 - 依赖缺失/参数错误都不该让服务起不来
        logger.error("初始化主模型失败，将使用抽取式兜底: %s", exc)
        primary = None

    if primary is None:
        provider = "local"
        model_name = "extractive-local"
        logger.warning(
            "未配置可用的大模型（LLM_PROVIDER=%s, key=%s），"
            "当前使用抽取式兜底模型：回答来自知识库原文摘录，零幻觉但不做语言润色。",
            settings.llm_provider,
            bool(settings.dashscope_api_key or settings.openai_api_key),
        )

    return LLMClient(
        primary=primary,
        provider=provider,
        model=model_name,
        temperature=settings.llm_temperature,
        first_token_timeout=getattr(settings, "llm_first_token_timeout", 20.0),
        refusal_message=settings.refusal_message,
    )


# ---------------------------------------------------------------- 答案校验


def verify_answer(
    answer: str,
    context: str,
    *,
    min_overlap: float = 0.20,
) -> GroundingCheck:
    """校验答案是否"落地"在检索上下文上。

    指标：
    - coverage：答案中的中文 bigram / 英文词，有多大比例能在上下文中找到；
      模型正常引用/复述资料时覆盖率通常 > 0.4；纯记忆生成（幻觉）会明显偏低（< 0.15）。
    - citations：是否标注了 [编号] 引用。
    """
    if not answer.strip():
        return GroundingCheck(ok=False, citations=[], coverage=0.0, reason="答案为空")

    answer_tokens = _content_tokens(answer)
    context_tokens = _content_tokens(context)
    coverage = 0.0
    if answer_tokens:
        coverage = len(answer_tokens & context_tokens) / len(answer_tokens)

    citations = sorted({int(match) for match in _CITATION_RE.findall(answer)})
    refusal_like = any(keyword in answer for keyword in ("未找到", "没有找到", "无法回答", "资料中未", "知识库中未"))

    if refusal_like:
        return GroundingCheck(ok=True, citations=citations, coverage=coverage, reason="合理的拒答")

    reasons: list[str] = []
    if not citations:
        reasons.append("未标注引用来源")
    if coverage < min_overlap:
        reasons.append(f"答案与资料重合度偏低（{coverage:.2f} < {min_overlap:.2f}）")
    return GroundingCheck(ok=not reasons, citations=citations, coverage=coverage, reason="; ".join(reasons))


def _content_tokens(text: str) -> set[str]:
    """提取用于覆盖率比较的内容 token（中文 bigram + 英文词/数字）。"""
    tokens = {token for token in tokenize_for_sparse(text) if len(token) > 1 or token.isdigit()}
    return tokens
