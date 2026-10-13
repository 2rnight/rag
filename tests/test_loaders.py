"""解析器单测：多格式路由、结构化探测、质量分、异常路径。"""
from __future__ import annotations

import pytest

from rag.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from rag.loaders import (
    CsvParser,
    HybridDocumentLoader,
    JsonParser,
    MarkdownParser,
    TextParser,
    join_wrapped_lines,
    text_quality,
)


def test_loader_routes_by_suffix(tiny_docs):
    loader = HybridDocumentLoader()
    for name in ("handbook.md", "faq.txt", "products.csv", "config.json"):
        docs = loader.parse(tiny_docs / name)
        assert docs, f"{name} 未解析出内容"
        assert all(doc.metadata["source_file"] == name for doc in docs)
        assert all(doc.metadata["file_hash"] for doc in docs)


def test_loader_rejects_unsupported_type(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        HybridDocumentLoader().parse(path)


def test_loader_rejects_empty_document(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        HybridDocumentLoader().parse(path)


def test_markdown_parser_keeps_heading_hierarchy(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(
        "# 总标题\n\n## 第一章 概述\n\n这是第一章的正文内容，用于验证标题层级解析。\n\n### 1.1 细节\n\n细节内容。\n",
        encoding="utf-8",
    )
    docs = MarkdownParser().parse(path)
    paths = [doc.metadata["heading_path"] for doc in docs]
    assert any("总标题 > 第一章 概述" in item for item in paths)
    # 过短小节会与上一节合并，但小标题保留在 heading_paths 中（引用溯源不丢层级）
    all_paths = [item for doc in docs for item in doc.metadata.get("heading_paths", [])]
    assert any("1.1 细节" in item for item in all_paths)


def test_text_parser_detects_heading_structure(tmp_path):
    structured = tmp_path / "manual.txt"
    structured.write_text(
        "# 员工手册\n\n## 第一章\n内容一。\n\n## 第二章\n内容二。\n", encoding="utf-8"
    )
    docs = TextParser().parse(structured)
    assert all(doc.metadata["parser"] == "text+heading" for doc in docs)

    plain = tmp_path / "plain.txt"
    plain.write_text("第一段内容。\n\n第二段内容。\n", encoding="utf-8")
    docs = TextParser().parse(plain)
    assert all(doc.metadata["parser"] == "text" for doc in docs)


def test_csv_parser_converts_rows_to_named_fields(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("产品,价格\n智能助手,1999\n分析平台,3999\n", encoding="utf-8")
    docs = CsvParser(rows_per_block=1).parse(path)
    assert len(docs) == 2
    assert "产品: 智能助手" in docs[0].text
    assert docs[0].metadata["heading_path"].startswith("表格")


def test_json_parser_splits_list_items(tmp_path):
    path = tmp_path / "list.json"
    path.write_text('[{"a": 1}, {"a": 2}, {"a": 3}]', encoding="utf-8")
    docs = JsonParser(items_per_block=2).parse(path)
    assert len(docs) == 2
    assert docs[0].metadata["heading_path"] == "条目 1-2"


def test_text_quality_and_wrapped_line_join():
    assert text_quality("正常的中文内容，包含足够多的有效字符。") > 0.5
    assert text_quality("\ufffd\ufffd\ufffd") < 0.2
    joined = join_wrapped_lines("这是一句被硬换行截断的\n句子内容。")
    assert "截断的句子" in joined.replace("\n", "")
    assert "\n" in join_wrapped_lines("列表项 A\n- 列表项 B")
