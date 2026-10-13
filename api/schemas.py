"""API 请求/响应模型（Pydantic v2）。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- 通用


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    uptime_seconds: float
    knowledge_base_ready: bool
    knowledge_base_chunks: int
    llm_provider: str
    llm_model: str
    llm_fallback: bool
    embedding_model: str


class ChatMessage(BaseModel):
    """一轮对话（前端自带历史，后端无状态，便于横向扩展）。"""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class RetrievalHitModel(BaseModel):
    """命中的知识片段（引用来源卡片）。"""

    index: int
    chunk_id: str
    source: str
    page: int | None = None
    section: str = ""
    score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0
    retrieval: str = "dense"
    chars: int = 0
    preview: str = ""
    text: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------- 问答


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="用户问题")
    history: list[ChatMessage] = Field(default_factory=list, max_length=20, description="多轮历史（最近若干轮）")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索片段数，默认取服务端配置")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="相似度阈值（覆盖默认校准值）")
    mmr: bool | None = Field(default=None, description="是否启用 MMR 多样性重排")

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class RetrievalDiagnostics(BaseModel):
    strategy: str = ""
    top_score: float = 0.0
    threshold: float = 0.0
    is_answerable: bool = False
    hits: int = 0
    dense_hits: int = 0
    sparse_hits: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    question: str
    answer: str
    refused: bool = False
    fallback: bool = False
    grounded: bool = False
    citations: list[int] = Field(default_factory=list)
    coverage: float = 0.0
    notice: str = ""
    references: list[RetrievalHitModel] = Field(default_factory=list)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""


class SearchResponse(BaseModel):
    """纯检索接口（不经过 LLM），用于调试召回效果与画相似度分布。"""

    question: str
    answerable: bool
    strategy: str
    top_score: float
    threshold: float
    timings_ms: dict[str, float] = Field(default_factory=dict)
    results: list[RetrievalHitModel] = Field(default_factory=list)


# ---------------------------------------------------------------- 知识库


class IngestResultModel(BaseModel):
    source_file: str
    doc_id: str = ""
    doc_type: str = ""
    parser: str = ""
    blocks: int = 0
    chunks: int = 0
    chars: int = 0
    quality: float = 0.0
    chunks_total: int = 0
    skipped: bool = False
    reason: str = ""
    elapsed_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    results: list[IngestResultModel]
    documents_total: int
    chunks_total: int
    vectors_total: int
    elapsed_ms: float


class DocumentModel(BaseModel):
    doc_id: str
    source_file: str
    doc_type: str = ""
    parser: str = ""
    chunks: int = 0
    chars: int = 0
    quality: float = 0.0
    ingested_at: str = ""
    file_hash: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentModel]
    chunks_total: int
    vectors_total: int
    sources: dict[str, int] = Field(default_factory=dict)


class DeleteResponse(BaseModel):
    source_file: str
    removed_chunks: int
    chunks_total: int
    vectors: int


class RebuildRequest(BaseModel):
    directory: str | None = Field(default=None, description="服务端目录（默认 docs/，仅允许项目内路径）")


class RebuildResponse(BaseModel):
    directory: str
    documents: int
    chunks: int
    vectors: int
    elapsed_ms: float
    failed: list[dict[str, Any]] = Field(default_factory=list)


class KnowledgeStatusResponse(BaseModel):
    ready: bool
    last_error: str | None = None
    knowledge_base: dict[str, Any]
    documents: list[dict[str, Any]]
    embedding: dict[str, Any]
    llm: dict[str, Any]
    retrieval: dict[str, Any]
    chunking: dict[str, Any]
    supported_types: list[str]
    manifest: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
    chunking: dict[str, Any] | None = None
