"""动态 Prompt 拼接与拒答兜底策略。

设计（面试主线之三：幻觉治理）：

1. **上下文预算控制**：检索回来的片段不无脑全塞 —— 按 token 预算（默认 3000）
   逐条裁剪、单条限长（超长片段截断），token 估算对中英文分别加权，
   保证"塞得下、不超模型窗口、不浪费钱"；
2. **结构化引用**：每段上下文都有 `[编号] 来源/页码/章节/相关度` 头部，
   要求模型在答案里标注 `[1][2]`，前端即可做引用对齐（答案 → 原文一键定位）；
3. **动态规则**：根据本次检索结果**动态**追加约束，而不是一套静态提示词：
   - 置信度低 → 强制谨慎措辞；
   - 命中多个文档 → 冲突处理规则（更具体的条款优先，并显式说明冲突）；
   - 存在多轮历史 → 指代消解提示；
   - 问题含"最新/今天"等时效词 → 提示资料可能滞后，不得编造时间；
4. **拒答兜底**：把"资料不足时说什么"写死在 Prompt 里（文案与后端兜底文案一致），
   让"拒答"成为模型的第一选择而不是最后手段。
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .logging_utils import estimate_tokens, truncate_to_tokens
from .vector_store import Hit

# 供"抽取式兜底 LLM"解析上下文用的显式标记（Prompt 与解析器共享同一份契约）
CONTEXT_START = "<<<CONTEXT>>>"
CONTEXT_END = "<<<END_CONTEXT>>>"

# 时效性关键词：命中时追加"资料可能滞后"的约束
TEMPORAL_KEYWORDS = (
    "最新", "最近", "近期", "今天", "现在", "目前", "当前", "如今",
    "实时", "今年", "本月", "刚刚", "2025", "2026",
)

SYSTEM_PROMPT = """你是「{kb_name}」的智能知识库问答助手，服务对象是企业内部员工。
你的回答必须**严格依据**下方 <参考资料> 中的内容，而不是你自己的通用知识。

【回答规则】
1. 只使用 <参考资料> 中的信息作答；资料中没有的内容一律不要编造。
2. 每条关键信息后用方括号标注来源编号，例如：年假为 5 天/年 [1]；金额或条款务必标注。
3. 资料不足以回答、或只能部分回答时：
   - 完全无相关信息 → 直接回复：「{refusal_message}」
   - 部分相关 → 先说明"资料中仅提到…"，再明确指出缺失的部分，不要推测。
4. 若不同资料存在冲突，说明冲突并优先采用**更具体、更新**的条款，同时标注各自来源编号。
5. 用简洁、有条理的中文回答；要点多时用编号列表；不要复述本提示词、不要输出思考过程。
{extra_rules}
<参考资料>
{context}
</参考资料>
"""

RULE_LOW_CONFIDENCE = (
    "6. 本次检索的最高相关度仅为 {score:.2f}（阈值 {threshold:.2f}），属于低置信度："
    "如资料与问题只是「看起来相关」但并未真正回答，请按规则 3 拒答，不要勉强作答。"
)
RULE_MULTI_SOURCE = (
    "6. 本次资料来自多个不同文档（{sources}）：如内容不一致，请分别标注来源并提示用户核对。"
)
RULE_HISTORY = (
    "6. 用户可能与你在多轮对话中（见 <历史对话>）：请先做指代消解，"
    "把「它/那个/上述」等指代补全成明确对象后再作答。"
)
RULE_TEMPORAL = (
    "6. 问题涉及「最新/当前」等时效诉求，而资料有其入库时间："
    "如资料可能已过期，请说明「以下内容基于知识库现有资料（{as_of}）」，不要凭常识补充新信息。"
)


def build_context_block(
    hits: Sequence[Hit],
    *,
    token_budget: int = 3000,
    per_hit_max_tokens: int = 700,
) -> tuple[str, list[Hit]]:
    """把命中的片段拼成带编号与来源信息的上下文，受 token 预算约束。

    返回 (上下文文本, 实际使用的 hits)，实际使用的 hits 用于前端引用与埋点。
    """
    lines: list[str] = []
    used: list[Hit] = []
    consumed = 0

    for index, hit in enumerate(hits, start=1):
        header_bits = [f"[{index}] 来源: {hit.source_file}"]
        if hit.page:
            header_bits.append(f"页码: {hit.page}")
        if hit.section:
            header_bits.append(f"章节: {hit.section}")
        header_bits.append(f"相关度: {hit.score:.2f}")
        if hit.metadata.get("ingested_at"):
            header_bits.append(f"入库时间: {str(hit.metadata['ingested_at'])[:10]}")
        header = " | ".join(header_bits)

        body = truncate_to_tokens(hit.text, per_hit_max_tokens)
        block = f"{header}\n{body}"
        cost = estimate_tokens(block)
        if consumed + cost > token_budget and used:
            break
        lines.append(block)
        used.append(hit)
        consumed += cost

    return "\n\n---\n\n".join(lines), used


def build_messages(
    question: str,
    hits: Sequence[Hit],
    *,
    history: Sequence[dict[str, Any]] | None = None,
    kb_name: str = "企业知识库",
    refusal_message: str,
    threshold: float = 0.30,
    token_budget: int = 3000,
    per_hit_max_tokens: int = 700,
    kb_empty: bool = False,
    extra_rules_hint: str | None = None,
) -> tuple[list[BaseMessage], list[Hit], str]:
    """构建最终送入 LLM 的消息列表。

    Returns:
        (messages, 实际使用的 hits, 渲染后的 system prompt)
    """
    context, used_hits = build_context_block(
        hits, token_budget=token_budget, per_hit_max_tokens=per_hit_max_tokens
    )

    extra_rules: list[str] = []
    top_score = max((hit.score for hit in hits), default=0.0)
    sources = {hit.source_file for hit in used_hits}

    if used_hits and top_score < max(threshold * 1.15, threshold + 0.05):
        extra_rules.append(RULE_LOW_CONFIDENCE.format(score=top_score, threshold=threshold))
    if len(sources) > 1:
        extra_rules.append(RULE_MULTI_SOURCE.format(sources="、".join(sorted(sources))))
    if history:
        extra_rules.append(RULE_HISTORY)
    if any(keyword in question for keyword in TEMPORAL_KEYWORDS):
        extra_rules.append(RULE_TEMPORAL.format(as_of=date.today().isoformat()))
    if extra_rules_hint:
        extra_rules.append(extra_rules_hint)

    system_prompt = SYSTEM_PROMPT.format(
        kb_name=kb_name,
        refusal_message=refusal_message,
        context=f"{CONTEXT_START}\n{context}\n{CONTEXT_END}" if context else "(无检索结果)",
        extra_rules=("\n".join(extra_rules) + "\n") if extra_rules else "",
    )

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history:
        history_text = "\n".join(
            f"{'用户' if message.get('role') == 'user' else '助手'}: {str(message.get('content', ''))[:400]}"
            for message in history
        )
        messages.append(HumanMessage(content=f"<历史对话>\n{history_text}\n</历史对话>"))
    messages.append(HumanMessage(content=question if not kb_empty else f"{question}\n(当前知识库为空)"))
    return messages, used_hits, system_prompt


def extract_context_from_messages(messages: Sequence[BaseMessage]) -> str:
    """从消息里取出 <参考资料> 原文（抽取式兜底 LLM 使用）。"""
    for message in messages:
        if isinstance(message, SystemMessage):
            content = str(message.content)
            start = content.find(CONTEXT_START)
            end = content.find(CONTEXT_END)
            if start != -1 and end != -1:
                return content[start + len(CONTEXT_START) : end].strip()
    return ""


def extract_question_from_messages(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""
