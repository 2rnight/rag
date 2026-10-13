"""统一异常体系：便于 API 层把内部错误映射成合适的 HTTP 状态码。"""
from __future__ import annotations


class RagError(Exception):
    """业务异常基类。"""

    status_code = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class UnsupportedFileTypeError(RagError):
    status_code = 415

    def __init__(self, suffix: str, supported: list[str]) -> None:
        super().__init__(
            f"不支持的文件类型: {suffix or '(无扩展名)'}",
            detail=f"当前支持: {', '.join(sorted(supported))}",
        )


class DocumentParseError(RagError):
    """文档解析失败（损坏、加密、扫描件无文字层等）。"""

    status_code = 422


class EmptyDocumentError(DocumentParseError):
    def __init__(self, source: str, hint: str = "") -> None:
        message = f"文档 {source} 未解析出有效文本"
        if hint:
            message += f"：{hint}"
        super().__init__(message)


class IndexNotReadyError(RagError):
    status_code = 409

    def __init__(self, message: str = "向量索引尚未构建，请先上传文档或执行全量入库") -> None:
        super().__init__(message)


class EmbeddingError(RagError):
    """向量化调用失败（额度、网络、鉴权）。"""

    status_code = 502


class LLMError(RagError):
    status_code = 502


class KnowledgeBaseEmptyError(RagError):
    status_code = 409

    def __init__(self, message: str = "知识库为空，请先上传或索引文档") -> None:
        super().__init__(message)
