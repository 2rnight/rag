"""编排层单测：入库幂等、删除一致性、流式事件协议、拒答兜底。"""
from __future__ import annotations

import json

import pytest

from rag.pipeline import RagPipeline


@pytest.fixture()
def local_pipeline(settings):
    """每个用例一个干净的临时知识库。"""
    pipeline = RagPipeline(settings)
    pipeline.bootstrap(strict=False)
    pipeline.reset()
    return pipeline


def test_ingest_and_query_roundtrip(local_pipeline, sample_docs):
    results = local_pipeline.ingest_directory(sample_docs)
    indexed = [r for r in results if not r.skipped]
    assert len(indexed) == 4, f"应索引 4 个文件，实际 {len(indexed)}"
    assert len(local_pipeline.chunk_store) > 0
    assert local_pipeline.vector_store.size == len(local_pipeline.chunk_store)
    assert local_pipeline.bm25.size == len(local_pipeline.chunk_store)

    answer = local_pipeline.answer("年假有多少天？")
    assert not answer.refused
    assert "5 天" in answer.answer.replace("5天", "5 天")
    assert answer.references and answer.references[0]["source"] == "员工手册.txt"
    assert answer.timings["total_ms"] > 0
    assert answer.retrieval["hits"] >= 1


def test_reingest_same_file_is_idempotent(local_pipeline, sample_docs):
    first = local_pipeline.ingest_directory(sample_docs)
    chunks_after_first = len(local_pipeline.chunk_store)
    second = local_pipeline.ingest_directory(sample_docs)
    skipped = [r for r in second if r.skipped]
    assert len(skipped) == len(first), "内容未变化时应跳过重复入库"
    assert len(local_pipeline.chunk_store) == chunks_after_first, "重复入库不应产生重复向量"


def test_delete_document_syncs_all_stores(local_pipeline, sample_docs):
    local_pipeline.ingest_directory(sample_docs)
    before = len(local_pipeline.chunk_store)
    result = local_pipeline.delete_document("员工手册.txt")
    assert result["removed_chunks"] > 0
    assert len(local_pipeline.chunk_store) == before - result["removed_chunks"]
    assert local_pipeline.vector_store.size == len(local_pipeline.chunk_store)
    assert local_pipeline.bm25.size == len(local_pipeline.chunk_store)
    assert all(doc["source_file"] != "员工手册.txt" for doc in local_pipeline.registry.all())
    assert "员工手册.txt" not in local_pipeline.chunk_store.source_counts()


def test_rebuild_all_recreates_index(local_pipeline, settings):
    report = local_pipeline.rebuild_all(settings.docs_dir)
    assert report["documents"] >= 4
    assert report["chunks"] == len(local_pipeline.chunk_store)
    assert local_pipeline.vector_store.size == report["chunks"]


def test_refusal_and_empty_kb_messages(local_pipeline, sample_docs):
    # 空知识库
    empty = local_pipeline.answer("年假有多少天？")
    assert empty.refused and "知识库为空" in empty.answer

    local_pipeline.ingest_directory(sample_docs)
    out_of_scope = local_pipeline.answer("火星殖民地什么时候建成？")
    assert out_of_scope.refused
    assert "未找到" in out_of_scope.answer or "没有" in out_of_scope.answer
    assert out_of_scope.references == []


@pytest.mark.asyncio
async def test_streaming_event_protocol(local_pipeline, sample_docs):
    local_pipeline.ingest_directory(sample_docs)
    events = [event async for event in local_pipeline.astream_answer("年假有多少天？")]

    assert events[0]["event"] == "meta", "必须先返回检索结果，前端才能立刻渲染引用"
    assert events[-1]["event"] == "done"
    assert any(e["event"] == "delta" for e in events)

    meta = events[0]["data"]
    assert meta["answerable"] is True
    assert meta["sources"] and meta["sources"][0]["source"] == "员工手册.txt"

    done = events[-1]["data"]
    assert done["refused"] is False
    assert done["ttft_ms"] > 0 and done["total_ms"] >= done["ttft_ms"]
    assert done["sources"]
    assert json.dumps(done, ensure_ascii=False)  # 事件体必须可 JSON 序列化（SSE 要求）

    text = "".join(e["data"]["text"] for e in events if e["event"] == "delta")
    assert "5 天" in text.replace("5天", "5 天")


@pytest.mark.asyncio
async def test_streaming_refusal_path_skips_llm(local_pipeline, sample_docs):
    local_pipeline.ingest_directory(sample_docs)
    events = [event async for event in local_pipeline.astream_answer("请写一首关于秋天的诗")]
    meta = events[0]["data"]
    assert meta["answerable"] is False
    assert events[-1]["data"]["refused"] is True
    # 拒答不应触发 LLM：只有一条 delta（拒答文案）
    deltas = [e for e in events if e["event"] == "delta"]
    assert len(deltas) == 1


def test_status_and_chunking_stats(local_pipeline, sample_docs):
    local_pipeline.ingest_directory(sample_docs)
    status = local_pipeline.status()
    assert status["ready"] is True
    assert status["knowledge_base"]["chunks"] == len(local_pipeline.chunk_store)
    assert status["chunking"]["chunk_overlap"] / status["chunking"]["chunk_size"] == pytest.approx(0.2)

    stats = local_pipeline.chunking_stats()
    assert stats.chunks == len(local_pipeline.chunk_store)
    assert stats.min_chars > 0 and stats.max_chars <= 750
    assert 0 <= stats.overlap_ratio <= 0.2


def test_keyword_question_answer_contains_exact_value(pipeline):
    """端到端：关键词型问题（专有名词/数字条款）的答案必须给出精确值，而不是"看着相关"的邻居段落。"""
    result = pipeline.answer("交通补贴是多少钱一个月？")
    assert not result.refused
    assert "500元" in result.answer
