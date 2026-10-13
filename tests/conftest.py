"""pytest 公共夹具。

原则：**测试必须离线可跑、结果确定性**（CI 不依赖外网/API Key），
所以统一使用 local 兜底向量模型与抽取式兜底 LLM。
数据目录指向临时目录，避免污染仓库里的 data/。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_DATA = Path(tempfile.mkdtemp(prefix="rag-test-data-"))

# 必须在导入 config/app 之前设置环境变量（Settings 是 lru_cache 单例）
os.environ.update(
    {
        "RAG_LLM_PROVIDER": "local",
        "RAG_EMBEDDING_PROVIDER": "local",
        "RAG_DATA_DIR": str(TMP_DATA),
        "RAG_DOCS_DIR": str(REPO_ROOT / "docs"),
        "RAG_LOG_LEVEL": "WARNING",
        "RAG_INDEX_DIR": str(TMP_DATA / "faiss_index"),
    }
)

from config import get_settings  # noqa: E402
from rag.embeddings import build_embeddings  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def embeddings(settings):
    return build_embeddings(settings)


@pytest.fixture(scope="session")
def pipeline(settings) -> RagPipeline:
    """一个已索引 docs/ 的流水线（会话级复用，避免重复向量化）。"""
    instance = RagPipeline(settings)
    instance.bootstrap(strict=False)
    if len(instance.chunk_store) == 0:
        instance.ingest_directory(settings.docs_dir)
    return instance


@pytest.fixture(scope="session")
def sample_docs(settings) -> Path:
    """真实语料的一个独立副本（用于入库/检索/删除等集成用例）。"""
    directory = TMP_DATA / "sample_docs"
    directory.mkdir(parents=True, exist_ok=True)
    for path in sorted((REPO_ROOT / "docs").glob("*")):
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    return directory


@pytest.fixture(scope="session")
def tiny_docs() -> Path:
    """极简语料（多格式各一个文件），用于验证解析与多格式路由。"""
    directory = TMP_DATA / "tiny_docs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "handbook.md").write_text(
        """# 测试员工手册

## 第一章 考勤
标准工作时间为每周一至周五 9:00-18:00，午休 12:00-13:30。
迟到 30 分钟以内每次扣 50 元。

## 第二章 假期
年假天数为 5 天/年，工作满 10 年后为 10 天/年。
""",
        encoding="utf-8",
    )
    (directory / "faq.txt").write_text(
        "Q: 公司WiFi密码是多少？\nA: WiFi 密码是 Test@2026，访客网络无需密码。\n",
        encoding="utf-8",
    )
    (directory / "products.csv").write_text(
        "产品,价格,库存\n智能助手Pro,1999,120\n数据分析平台,3999,45\n", encoding="utf-8"
    )
    (directory / "config.json").write_text(
        '{"service": "rag", "port": 8000, "features": ["hybrid-search", "sse"]}', encoding="utf-8"
    )
    return directory


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    shutil.rmtree(TMP_DATA, ignore_errors=True)
