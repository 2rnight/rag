"""检索层单测：混合召回、RRF 融合、去重、拒答判定、向量库一致性。"""
from __future__ import annotations

import pytest

from rag.bm25 import BM25Index
from rag.exceptions import KnowledgeBaseEmptyError
from rag.logging_utils import tokenize_for_sparse


def test_bm25_ranks_term_matches_first():
    index = BM25Index().build([
        ("c1", "年假天数为 5 天，工作满 10 年为 10 天。"),
        ("c2", "报销流程需要直属上级与财务部审批。"),
        ("c3", "公司班车路线与发车时间说明。"),
    ])
    hits = index.search("年假多少天", k=3)
    assert hits and hits[0].chunk_id == "c1"
    assert hits[0].matched_terms >= 1


def test_bm25_incremental_add_and_serialization(tmp_path):
    index = BM25Index().build([("c1", "考勤规定与打卡说明")])
    assert index.add([("c2", "年假与调休说明")]) == 1
    hits = index.search("年假", k=5)
    assert hits and hits[0].chunk_id == "c2"

    path = tmp_path / "bm25.json"
    index.save(path)
    loaded = BM25Index.load(path)
    assert loaded is not None and loaded.size == 2
    assert loaded.search("考勤", k=5)[0].chunk_id == "c1"


def test_tokenizer_handles_cjk_and_ascii():
    tokens = tokenize_for_sparse("VPN 地址 vpn.zhiyuan.com 端口 443")
    assert "vpn" in tokens
    assert "端口" in tokens
    assert any(token.isdigit() or token == "443" for token in tokens)


def test_hybrid_retrieval_fuses_dense_and_sparse(pipeline):
    result = pipeline.retrieve("公司WiFi密码是多少？", top_k=5)
    assert result.strategy in {"hybrid_rrf", "dense", "dense_mmr"}
    assert result.is_answerable
    assert result.hits, "应召回片段"
    assert result.top_score >= result.threshold
    assert result.top_hit.source_file == "常见问题FAQ.txt"
    # 融合命中应同时带有稠密与稀疏信息
    fused = [hit for hit in result.hits if hit.retrieval.startswith("hybrid")]
    assert fused, "混合检索应产出融合结果"
    assert fused[0].fusion_score > 0
    diagnostics = result.diagnostics()
    for key in ("strategy", "top_score", "threshold", "is_answerable", "timings_ms"):
        assert key in diagnostics


def test_out_of_scope_question_is_refused(pipeline):
    result = pipeline.retrieve("火星殖民地什么时候建成？")
    assert not result.is_answerable
    assert result.refusal_reason, "拒答时必须给出可解释的原因"
    assert "unknown" in result.refusal_reason or "low" in result.refusal_reason


def test_neighbor_expansion_brings_adjacent_chunks(pipeline):
    result = pipeline.retrieve("年假有多少天？", top_k=3)
    neighbor_hits = [hit for hit in result.hits if "neighbor" in hit.retrieval]
    assert result.hits, "基础召回不应该为空"
    if neighbor_hits:
        assert neighbor_hits[0].metadata["doc_id"] == result.hits[0].metadata["doc_id"]


def test_retrieval_raises_on_empty_knowledge_base(settings, embeddings):
    from rag.pipeline import RagPipeline

    pipeline = RagPipeline(settings, embeddings=embeddings)
    pipeline.bootstrap(strict=False)
    pipeline.reset()
    with pytest.raises(KnowledgeBaseEmptyError):
        pipeline.retrieve("年假有多少天？")
