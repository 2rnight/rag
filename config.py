"""
全局配置（pydantic-settings）。

所有配置项都可以通过环境变量覆盖，前缀 `RAG_`；
API Key 额外兼容业界约定俗成的变量名（DASHSCOPE_API_KEY / OPENAI_API_KEY），
方便直接复用已有的环境变量或 .env 文件。

约定：
    chunk_overlap = chunk_size * chunk_overlap_ratio   # 默认 20%
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

# 供应商默认模型
DEFAULT_QWEN_LLM = "qwen-plus"
DEFAULT_QWEN_EMBEDDING = "text-embedding-v3"
DEFAULT_OPENAI_LLM = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING = "text-embedding-3-small"

# 中文优先的递归切分分隔符阶梯（从大到小，尽量保住语义结构）
DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "!",
    "?",
    ";",
    ".",
    "，",
    ",",
    " ",
]


class Settings(BaseSettings):
    """项目配置对象（单例，见 `get_settings()`）。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
        case_sensitive=False,
    )

    # ==================== 服务 ====================
    app_name: str = "企业知识库智能问答系统"
    version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = ["*"]
    max_upload_mb: int = 50
    # SSE 心跳间隔（秒）：防止 Nginx/网关在长连接空闲时主动断开
    sse_heartbeat_interval: float = 15.0

    # ==================== 数据目录 ====================
    data_dir: Path = BASE_DIR / "data"
    docs_dir: Path = BASE_DIR / "docs"
    upload_dir: Path | None = None          # None -> data_dir/uploads
    index_dir: Path | None = None           # None -> data_dir/faiss_index

    # ==================== LLM ====================
    llm_provider: Literal["auto", "qwen", "openai", "local"] = "auto"
    llm_model: str = ""                     # 空 -> 按 provider 取默认模型
    llm_temperature: float = 0.1            # 知识库问答要稳，温度压低
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    # 首 token 超时：超过该时间一个字都没吐出来 -> 直接降级到抽取式兜底
    llm_first_token_timeout: float = 20.0

    dashscope_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RAG_DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    )
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("RAG_DASHSCOPE_BASE_URL", "DASHSCOPE_BASE_URL"),
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("RAG_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )

    # ==================== Embedding ====================
    embedding_provider: Literal["auto", "qwen", "openai", "local"] = "auto"
    embedding_model: str = ""               # 空 -> 按 provider 取默认模型
    embedding_dim: int = 1024               # text-embedding-v3 支持 1024/768/512
    embedding_batch_size: int = 10          # DashScope 单次最多 10 条
    embedding_timeout: float = 30.0
    embedding_max_retries: int = 3
    embedding_max_chars: int = 6000         # 单条文本截断长度（token 上限保护）
    local_embedding_dim: int = 1024         # 离线兜底模型的向量维度

    # ==================== 切片 ====================
    chunk_size: int = 500
    chunk_overlap_ratio: float = 0.2        # ★ 重叠窗口比例 20%
    chunk_min_size: int = 60                # 小于该长度的碎片直接合并
    separators: list[str] = DEFAULT_SEPARATORS

    # ==================== 检索 ====================
    retrieval_top_k: int = 5                # 最终喂给 LLM 的片段数
    retrieval_fetch_k: int = 20             # 每路召回候选数（再融合、去重、截断）
    hybrid_enabled: bool = True             # 稠密(FAISS) + 稀疏(BM25) 混合召回
    rrf_k: int = 60                         # RRF 融合平滑常数
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    dedupe_similarity: float = 0.92         # 融合后近似重复片段的去重阈值
    score_threshold: float = -1.0           # <0 表示使用 embedding provider 的推荐阈值
    refusal_use_dense_only: bool = True     # 仅按语义相似度判定"知识库是否有答案"
    max_unknown_ratio: float = 0.60         # 查询概念在知识库词表中的最大"未知占比"，超过则拒答
    # BM25 救援：稠密分不达阈值、但 BM25 证据强且查询概念在库内（未知率低）时，判为可答。
    # 注意 max_unknown_ratio 与 rescue 的未知率上限分工不同：前者识别"问的知识库根本没有的概念"，
    # 后者保证"BM25 字面命中"的可信度。二者都可用 scripts/evaluate.py --tune 重新校准。
    sparse_rescue_min_score: float = 8.0
    sparse_rescue_max_unknown_ratio: float = 0.55
    sparse_rescue_dense_ratio: float = 0.6
    # 最优片段的"突出度"= top 分 / 本次检索分数的中位数：
    # 低于该值说明整批片段分数都差不多（=没有真正命中），救援不成立
    sparse_rescue_min_standout: float = 2.0
    retrieval_mmr: bool = False             # MMR 多样性重排（问题相似但答案分散时开启）
    mmr_lambda: float = 0.7                 # MMR: 相关性 vs 多样性 的权衡系数
    neighbor_expansion: bool = True         # 命中块自动带上左右邻居（缓解"答案被切在边界"）

    # ==================== 上下文 / 多轮 ====================
    context_token_budget: int = 3000        # 拼进 Prompt 的检索上下文 token 预算
    context_snippet_max_tokens: int = 700   # 单个片段最多占用的 token
    history_max_turns: int = 3              # 参与拼接的历史轮数
    max_question_chars: int = 1000

    # ==================== 答案校验 / 兜底 ====================
    strict_grounding: bool = False          # True: 未通过校验的答案直接替换为抽取式答案
    grounding_min_overlap: float = 0.20     # 答案与上下文的最小 bigram 覆盖率

    # ==================== 内置文案 ====================
    refusal_message: str = (
        "根据当前知识库中的资料，我未找到能够回答该问题的内容，"
        "因此不作猜测。建议您补充相关文档，或换个更具体的问法。"
    )
    empty_kb_message: str = "当前知识库为空，请先上传文档（PDF / Word / Markdown / TXT / CSV / HTML）后再提问。"
    low_confidence_hint: str = "（检索置信度较低，以下内容仅供参考）"

    # ==================== 派生属性 ====================
    @model_validator(mode="after")
    def _derive(self) -> Settings:
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        if self.index_dir is None:
            self.index_dir = self.data_dir / "faiss_index"
        if self.llm_provider == "qwen" and not self.llm_model:
            self.llm_model = DEFAULT_QWEN_LLM
        if self.llm_provider == "openai" and not self.llm_model:
            self.llm_model = DEFAULT_OPENAI_LLM
        if not self.llm_model:
            self.llm_model = DEFAULT_QWEN_LLM
        if not self.embedding_model:
            if self.embedding_provider == "openai":
                self.embedding_model = DEFAULT_OPENAI_EMBEDDING
            else:
                self.embedding_model = DEFAULT_QWEN_EMBEDDING
        return self

    @property
    def chunk_overlap(self) -> int:
        """重叠字符数 = chunk_size × 20%（向上取整，至少 1）。"""
        return max(1, round(self.chunk_size * self.chunk_overlap_ratio))

    @property
    def chunk_overlap_percent(self) -> str:
        return f"{self.chunk_overlap_ratio * 100:.0f}%"

    @property
    def chunk_store_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.json"

    @property
    def registry_path(self) -> Path:
        return self.index_dir / "registry.json"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.upload_dir, self.index_dir):
            Path(p).mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """对外暴露的安全配置（不下发任何密钥）。"""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_overlap_ratio": self.chunk_overlap_ratio,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_fetch_k": self.retrieval_fetch_k,
            "hybrid_enabled": self.hybrid_enabled,
            "score_threshold": self.effective_score_threshold,
            "context_token_budget": self.context_token_budget,
        }

    # 由 pipeline 在 embedding 就绪后写入（不同 embedding 的余弦分布差异很大）
    _calibrated_threshold: float | None = None

    @property
    def effective_score_threshold(self) -> float:
        if self.score_threshold >= 0:
            return self.score_threshold
        if self._calibrated_threshold is not None:
            return self._calibrated_threshold
        return 0.30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()


if __name__ == "__main__":  # pragma: no cover - 手动排查用
    import json

    print(json.dumps(settings.public_dict(), ensure_ascii=False, indent=2))
