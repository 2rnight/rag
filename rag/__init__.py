"""RAG 知识库问答系统核心包。

模块划分（从下到上）：
    loaders     多格式文档解析（PDF / Word / Markdown / TXT / CSV / HTML / JSON）
    chunking    重叠窗口切片（Overlap 20%）+ 语义边界对齐
    embeddings  向量化（Qwen text-embedding-v3 / OpenAI / 本地离线兜底模型）
    store       片段仓库（chunks.jsonl）与文档注册表（registry.json）
    bm25        稀疏检索（BM25 Okapi）
    vector_store FAISS 向量索引（持久化 / 增量写入 / 删除 / 混合检索）
    retriever   混合检索 + RRF 融合 + 去重 + 拒答判定
    prompts     动态 Prompt 拼接（上下文预算、历史、置信度提示）
    llm         LangChain 统一大模型接口 + 抽取式兜底 + 答案校验
    pipeline    端到端编排（入库 / 检索 / 流式生成 / 埋点）
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
