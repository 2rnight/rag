"""Prompt 拼接与生成层单测：预算控制、动态规则、抽取式兜底、答案校验、降级。"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from rag.llm import ExtractiveChatModel, LLMClient, verify_answer
from rag.logging_utils import estimate_tokens
from rag.prompts import (
    CONTEXT_END,
    CONTEXT_START,
    build_context_block,
    build_messages,
    extract_context_from_messages,
)
from rag.testing import SimulatedChatModel
from rag.vector_store import Hit


def make_hit(text: str, score: float = 0.8, source: str = "员工手册.txt", index: int = 0) -> Hit:
    return Hit(
        chunk_id=f"c{index}",
        text=text,
        metadata={"source_file": source, "page_number": 3, "heading_path": "第二章 假期"},
        score=score,
    )


def test_context_block_respects_token_budget():
    hits = [make_hit("内容很长的一段资料。" * 200, score=0.9 - i * 0.01, index=i) for i in range(5)]
    context, used = build_context_block(hits, token_budget=600, per_hit_max_tokens=200)
    assert used, "至少应放进一条片段"
    assert estimate_tokens(context) <= 700  # 单条上限 + 头部信息的余量
    assert CONTEXT_START not in context  # 标记由 build_messages 添加
    assert "[1]" in context and "来源" in context


def test_build_messages_injects_dynamic_rules():
    hits = [make_hit("年假为 5 天。", score=0.31), make_hit("调休说明。", score=0.30, source="FAQ.txt", index=1)]
    messages, used, system_prompt = build_messages(
        "最近年假政策有变化吗？",
        hits,
        history=[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}],
        kb_name="测试知识库",
        refusal_message="资料不足，无法回答。",
        threshold=0.35,
    )
    assert CONTEXT_START in system_prompt and CONTEXT_END in system_prompt
    assert "低置信度" in system_prompt  # 分数接近阈值 -> 追加谨慎规则
    assert "多个不同文档" in system_prompt  # 多来源规则
    assert "历史对话" in system_prompt  # 多轮提示
    assert "时效" in system_prompt  # 命中"最近"
    assert "资料不足，无法回答。" in system_prompt  # 拒答文案注入
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert used and extract_context_from_messages(messages).startswith("[1] 来源")


def test_extractive_model_refuses_without_context():
    model = ExtractiveChatModel(refusal_message="没有资料。")
    assert model.compose([HumanMessage(content="年假多少天？")]) == "没有资料。"


def test_extractive_model_picks_relevant_sentence_with_citation():
    model = ExtractiveChatModel()
    prompt, _, system = build_messages(
        "年假有多少天？",
        [
            make_hit("年假：工作满 1 年不满 10 年 5 天/年；工作满 10 年不满 20 年 10 天/年。", score=0.9),
            make_hit("报销流程：线上提交后由直属上级审批。", score=0.4, source="FAQ.txt", index=1),
        ],
        kb_name="测试",
        refusal_message="资料不足。",
    )
    answer = model.compose(prompt)
    assert "年假" in answer and "[1]" in answer
    assert "报销流程" not in answer  # 低分块不应进来


def test_extractive_model_prefers_matching_numeric_range():
    """数值亲和：问 6000 元应命中『5000 元以上』档，而不是『1000-5000 元』档。"""
    model = ExtractiveChatModel(max_sentences=1)
    messages, _, _ = build_messages(
        "报销6000元需要谁审批？",
        [make_hit("报销金额限制：- 1000元以下：部门经理审批 - 1000-5000元：总监审批 - 5000元以上：VP审批", score=0.9)],
        kb_name="测试",
        refusal_message="资料不足。",
    )
    answer = model.compose(messages)
    assert "VP" in answer


def test_verify_answer_detects_unsupported_claims():
    context = "[1] 来源: 员工手册.txt\n年假天数为 5 天/年。试用期工资为正式工资的 80%。"
    good = verify_answer("年假天数为 5 天/年 [1]。", context, min_overlap=0.2)
    assert good.ok and good.citations == [1]

    bad = verify_answer("公司计划在 2030 年发射自研火箭并登陆火星。", context, min_overlap=0.2)
    assert not bad.ok and bad.coverage < 0.2

    refusal = verify_answer("根据现有资料，未找到相关信息。", context)
    assert refusal.ok and "拒答" in refusal.reason


def test_llm_client_falls_back_when_upstream_fails():
    client = LLMClient(
        primary=SimulatedChatModel(fail=True),
        provider="simulated",
        model="failing",
        fallback=ExtractiveChatModel(refusal_message="资料不足。"),
    )
    messages, _, _ = build_messages(
        "年假有多少天？",
        [make_hit("年假天数为 5 天/年。", score=0.9)],
        kb_name="测试",
        refusal_message="资料不足。",
    )
    answer = client.invoke(messages)
    assert "5 天" in answer
    assert client.used_fallback and client.last_error


@pytest.mark.asyncio
async def test_llm_client_streams_and_marks_fallback():
    client = LLMClient(
        primary=SimulatedChatModel(answer="流式答案 [1]", first_token_delay=0.0, per_chunk_delay=0.0, chunk_chars=999),
        provider="simulated",
        model="ok",
    )
    messages, _, _ = build_messages(
        "问题", [make_hit("答案依据。", score=0.9)], kb_name="测试", refusal_message="资料不足。"
    )
    deltas = [delta async for delta in client.astream(messages)]
    assert "".join(delta.text for delta in deltas) == "流式答案 [1]"
    assert all(not delta.fallback for delta in deltas)


@pytest.mark.asyncio
async def test_llm_client_first_token_timeout_switches_to_fallback():
    client = LLMClient(
        primary=SimulatedChatModel(first_token_delay=5.0),
        provider="simulated",
        model="slow",
        first_token_timeout=0.2,
        fallback=ExtractiveChatModel(refusal_message="资料不足。"),
    )
    messages, _, _ = build_messages(
        "年假有多少天？", [make_hit("年假天数为 5 天/年。", score=0.9)], kb_name="测试", refusal_message="资料不足。"
    )
    text = "".join([delta.text async for delta in client.astream(messages)])
    assert "5 天" in text, "首字超时后应切换到抽取式兜底"
