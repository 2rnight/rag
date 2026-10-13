"""向量化模块：Qwen(text-embedding-v3) / OpenAI / 本地离线兜底。

要点（面试可讲）：
1. **统一抽象**：所有 provider 都实现 LangChain 的 `Embeddings` 接口
   （`embed_documents` / `embed_query`），上层 FAISS 索引完全不关心底层是谁，
   换模型只是换一个工厂参数 —— 这是"LangChain 统一大模型对接接口"在向量侧的价值。
2. **维度/模型指纹**：向量库 manifest 记录 `provider:model:dim` 指纹，
   换 embedding 模型后如果直接复用旧索引，检索会静默变差（维度不同甚至直接报错）。
   指纹不一致时索引会拒绝加载并提示重建 —— 线上很常见的坑。
3. **工程细节**：DashScope 单次最多 10 条、有 token 上限，所以要做
   **批量切分 + 超长文本截断 + 指数退避重试**；查询向量走 LRU 缓存降低重复问题开销。
4. **离线兜底（local）**：没有 API Key / 断网时，用"字符 n-gram 哈希 + 亚线性 TF"
   构造确定性向量（本质是哈希词袋余弦相似度），保证整条链路（入库→检索→流式问答→评测）
   在没有外网的环境下依然能跑通与演示；生产环境切换到 text-embedding-v3 只需改一个环境变量。
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from .exceptions import EmbeddingError
from .logging_utils import get_logger, tokenize_for_sparse

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmbeddingInfo:
    """向量模型元信息（写入索引 manifest，用于一致性校验）。"""

    provider: str
    model: str
    dim: int
    is_fallback: bool = False
    recommended_threshold: float = 0.30  # 该模型的"有答案/无答案"余弦阈值经验值

    def fingerprint(self) -> str:
        return f"{self.provider}:{self.model}:{self.dim}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "is_fallback": self.is_fallback,
            "recommended_threshold": self.recommended_threshold,
            "fingerprint": self.fingerprint(),
        }


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """L2 归一化：让内积 == 余弦相似度（FAISS 用 MAX_INNER_PRODUCT 即可）。"""
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(v) / norm for v in vector]


def truncate_for_embedding(text: str, max_chars: int) -> str:
    """超长文本截断：保留头尾（重要信息常在两头），中间用省略号占位。"""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head - 8
    return f"{text[:head]}……{text[-tail:]}"


class RagEmbeddings(Embeddings):
    """向量模型基类：实现 LangChain 的 `Embeddings` 接口 + 查询缓存 + 埋点。

    约定：**所有实现都必须输出 L2 归一化向量**，
    这样 FAISS 用内积检索就等价于余弦相似度（分数可解释、可直接当阈值用）。
    """

    provider: str = "base"
    model: str = "base"
    dim: int = 0

    def __init__(self, *, model: str, dim: int, batch_size: int = 10, max_chars: int = 6000) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.max_chars = max_chars
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_size = 512
        self.calls = 0
        self.total_texts = 0

    # ---------------- LangChain Embeddings 接口 ----------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            return cached
        vector = self.embed_documents([text])[0]
        self._query_cache[text] = vector
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vector

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        import anyio

        return await anyio.to_thread.run_sync(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        import anyio

        return await anyio.to_thread.run_sync(self.embed_query, text)

    # ---------------- 元信息 ----------------
    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(provider=self.provider, model=self.model, dim=self.dim)

    def stats(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "calls": self.calls, "texts": self.total_texts}

    def _batches(self, texts: list[str]) -> list[list[str]]:
        return [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]


class HttpEmbeddings(RagEmbeddings):
    """OpenAI 兼容协议的向量模型客户端（DashScope / OpenAI / 其他网关通用）。"""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        dim: int | None = None,
        batch_size: int = 10,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_chars: int = 6000,
        recommended_threshold: float = 0.30,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError(f"{provider} 缺少 API Key")
        self.provider = provider
        self.recommended_threshold = recommended_threshold
        self._requested_dim = dim
        self._dimensions_supported = dim is not None
        self._max_retries = max(1, max_retries)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self._aclient: httpx.AsyncClient | None = None
        super().__init__(model=model, dim=dim or 0, batch_size=batch_size, max_chars=max_chars)

    # ---------------- 同步 ----------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            payload_inputs = [truncate_for_embedding(t, self.max_chars) for t in batch]
            embeddings = self._post(payload_inputs)
            vectors.extend(l2_normalize(v) for v in embeddings)
        self.total_texts += len(texts)
        return vectors

    def _post(self, inputs: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            payload: dict[str, Any] = {"model": self.model, "input": inputs, "encoding_format": "float"}
            if self._dimensions_supported and self._requested_dim:
                payload["dimensions"] = self._requested_dim
            try:
                response = self._client.post("/embeddings", json=payload)
            except httpx.HTTPError as exc:  # 网络/超时 -> 退避重试
                last_error = exc
                self._sleep(attempt)
                continue

            if response.status_code == 400 and "dimensions" in response.text and "dimensions" in payload:
                # 部分网关不支持 dimensions 参数：降级重试一次
                logger.warning("embeddings 接口不支持 dimensions 参数，已自动降级")
                self._dimensions_supported = False
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {response.text[:200]}", request=response.request, response=response
                )
                self._sleep(attempt, retry_after=response.headers.get("retry-after"))
                continue
            if response.status_code >= 400:
                raise EmbeddingError(
                    f"向量化失败（HTTP {response.status_code}）",
                    detail=response.text[:300],
                )
            data = response.json().get("data") or []
            if not data:
                raise EmbeddingError("向量化返回为空", detail=response.text[:300])
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in ordered]
            if self.dim == 0:
                self.dim = len(vectors[0])
            return vectors

        raise EmbeddingError(f"向量化重试 {self._max_retries} 次后仍失败: {last_error}")

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        import time

        delay = min(8.0, 0.5 * (2**attempt))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)

    # ---------------- 异步 ----------------
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                base_url=str(self._client.base_url),
                headers=dict(self._client.headers),
                timeout=self._client.timeout,
            )
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            inputs = [truncate_for_embedding(t, self.max_chars) for t in batch]
            embeddings = await self._apost(inputs)
            vectors.extend(l2_normalize(v) for v in embeddings)
        self.total_texts += len(texts)
        return vectors

    async def _apost(self, inputs: list[str]) -> list[list[float]]:
        import asyncio

        assert self._aclient is not None
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            payload: dict[str, Any] = {"model": self.model, "input": inputs, "encoding_format": "float"}
            if self._dimensions_supported and self._requested_dim:
                payload["dimensions"] = self._requested_dim
            try:
                response = await self._aclient.post("/embeddings", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
                continue
            if response.status_code == 400 and "dimensions" in response.text and "dimensions" in payload:
                self._dimensions_supported = False
                continue
            if response.status_code >= 400:
                last_error = EmbeddingError(
                    f"向量化失败（HTTP {response.status_code}）", detail=response.text[:200]
                )
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
                continue
            data = response.json().get("data") or []
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in ordered]
            if self.dim == 0 and vectors:
                self.dim = len(vectors[0])
            return vectors
        raise EmbeddingError(f"向量化重试 {self._max_retries} 次后仍失败: {last_error}")

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider=self.provider,
            model=self.model,
            dim=self.dim,
            recommended_threshold=self.recommended_threshold,
        )


class LocalHashEmbeddings(RagEmbeddings):
    """离线兜底向量模型（无需网络/API Key）。

    算法：字符级 n-gram 分词（中英混排）-> 哈希词袋 -> 亚线性 TF 加权 -> L2 归一化。
    没有任何"语义"能力，但对术语/数字/专有名词的**字面召回**足够稳定，
    因此可以完整跑通"入库 → FAISS 检索 → 流式问答 → 评测"全链路，
    也适合作为 CI 里不依赖外网、结果确定性的默认 provider。
    """

    provider = "local"
    recommended_threshold = 0.20

    def __init__(self, dim: int = 1024, model: str = "local-ngram-hash-v1") -> None:
        super().__init__(model=model, dim=dim, batch_size=64, max_chars=10000)

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider=self.provider,
            model=self.model,
            dim=self.dim,
            is_fallback=True,
            recommended_threshold=self.recommended_threshold,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.total_texts += len(texts)
        return [self._vector(text) for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def _vector(self, text: str) -> list[float]:
        counts = Counter(tokenize_for_sparse(text.lower()))
        vector = [0.0] * self.dim
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dim
            vector[index] += 1.0 + math.log(count)  # 亚线性 TF，抑制高频词
        return l2_normalize(vector)


class NormalizedEmbeddings(RagEmbeddings):
    """归一化包装器：给索引层兜底保证"向量已 L2 归一化"这一不变量。

    这样即使换成第三方/自研向量模型（没做归一化），内积也不至于变成"越大越像"的假象。
    """

    def __init__(self, inner: RagEmbeddings) -> None:
        super().__init__(
            model=inner.model, dim=inner.dim, batch_size=inner.batch_size, max_chars=inner.max_chars
        )
        self._inner = inner
        self.provider = inner.provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [l2_normalize(vector) for vector in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return l2_normalize(self._inner.embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [l2_normalize(vector) for vector in await self._inner.aembed_documents(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return l2_normalize(await self._inner.aembed_query(text))

    @property
    def info(self) -> EmbeddingInfo:
        return self._inner.info

    @property
    def inner(self) -> RagEmbeddings:
        return self._inner


def ensure_normalized(embeddings: RagEmbeddings) -> RagEmbeddings:
    """幂等地包一层归一化（已经是包装器则直接返回）。"""
    return embeddings if isinstance(embeddings, NormalizedEmbeddings) else NormalizedEmbeddings(embeddings)


def resolve_provider(settings) -> str:
    """provider 解析：auto 时按是否有 Key 决定（qwen -> openai -> local）。"""
    provider = settings.embedding_provider
    if provider != "auto":
        return provider
    if settings.dashscope_api_key:
        return "qwen"
    if settings.openai_api_key:
        return "openai"
    return "local"


def build_embeddings(settings) -> RagEmbeddings:
    """向量模型工厂。"""
    provider = resolve_provider(settings)
    if provider == "qwen":
        embeddings = HttpEmbeddings(
            provider="qwen",
            model=settings.embedding_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
            max_chars=settings.embedding_max_chars,
            recommended_threshold=0.35,
        )
        logger.info("向量模型: Qwen %s (dim=%s)", settings.embedding_model, settings.embedding_dim)
        return embeddings
    if provider == "openai":
        embeddings = HttpEmbeddings(
            provider="openai",
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            dim=settings.embedding_dim if settings.embedding_model.startswith("text-embedding-3") else None,
            batch_size=min(settings.embedding_batch_size, 64),
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
            max_chars=settings.embedding_max_chars,
            recommended_threshold=0.30,
        )
        logger.info("向量模型: OpenAI %s", settings.embedding_model)
        return embeddings

    logger.warning(
        "未检测到 DASHSCOPE_API_KEY / OPENAI_API_KEY，已启用离线兜底向量模型（local n-gram hash）。"
        "功能链路完整，但不具备语义泛化能力；配置 Key 后重建索引即为生产模式。"
    )
    return LocalHashEmbeddings(dim=settings.local_embedding_dim)
