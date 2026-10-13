"""切片策略单测：重叠窗口、边界对齐、ID 唯一性、元数据完整性。"""
from __future__ import annotations

from rag.chunking import OverlapChunker, make_doc_id
from rag.loaders import ParsedDocument


def test_overlap_defaults_to_twenty_percent():
    chunker = OverlapChunker(chunk_size=500)
    assert chunker.chunk_overlap == 100
    assert chunker.overlap_ratio == 0.2


def test_long_text_produces_overlapping_chunks():
    text = "。".join(f"第{i}条 这是用于测试重叠窗口的句子内容" for i in range(120))
    chunker = OverlapChunker(chunk_size=200, overlap_ratio=0.2, min_chunk_size=40)
    pieces = chunker.split_text_with_offsets(text)

    assert len(pieces) > 2, "长文本应被切成多块"
    for index, (start, chunk, overlap) in enumerate(pieces):
        assert len(chunk) <= 200 + 8  # 允许句末标点带来的少量溢出
        if index:
            assert overlap > 0, "相邻块之间必须有重叠，保证语义连续"
            assert overlap <= chunker.chunk_overlap + 8
        assert start >= 0


def test_chunks_do_not_break_sentences_in_the_middle():
    text = "".join(f"句子{i}：这是一段完整的中文内容。；" for i in range(80))
    chunker = OverlapChunker(chunk_size=120, overlap_ratio=0.2)
    for chunk in chunker.split_text(text):
        assert not chunk.rstrip().endswith("：")


def test_short_text_returns_single_chunk():
    chunker = OverlapChunker(chunk_size=500)
    assert chunker.split_text("很短的一句话。") == ["很短的一句话。"]


def test_duplicate_and_contained_chunks_are_removed():
    # 构造大量重复内容，去重后不应出现完全相同的切片
    text = ("重复内容。这段文字会被反复出现。" * 40) + "结尾补充说明。"
    chunker = OverlapChunker(chunk_size=100, overlap_ratio=0.2)
    chunks = chunker.split_text(text)
    assert len(chunks) == len(set(chunks))


def test_chunk_ids_are_unique_across_blocks_of_same_file():
    """同一文件被解析成多个逻辑块时，chunk_id 必须全局唯一（否则会踩 FAISS 重复 id）。"""
    chunker = OverlapChunker(chunk_size=120, overlap_ratio=0.2)
    docs = [
        ParsedDocument(
            text="第一章内容。" * 20,
            metadata={"source_file": "a.md", "file_hash": "h1", "heading_path": "第一章"},
        ),
        ParsedDocument(
            text="第二章内容。" * 20,
            metadata={"source_file": "a.md", "file_hash": "h1", "heading_path": "第二章"},
        ),
    ]
    chunks, stats = chunker.chunk_documents(docs)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk.metadata["source_file"] == "a.md" for chunk in chunks)
    # 序号在整个文件范围内连续
    indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))
    assert all(chunk.metadata["total_chunks"] == len(chunks) for chunk in chunks)
    assert stats.chunks == len(chunks)
    assert stats.avg_chars > 0


def test_group_blocks_merges_small_sections_and_keeps_paths():
    chunker = OverlapChunker(chunk_size=500, overlap_ratio=0.2)
    docs = [
        ParsedDocument(text="小节 A 的内容。", metadata={"source_file": "b.md", "file_hash": "h", "heading_path": "A"}),
        ParsedDocument(text="小节 B 的内容。", metadata={"source_file": "b.md", "file_hash": "h", "heading_path": "B"}),
        ParsedDocument(text="另一份文件。", metadata={"source_file": "c.md", "file_hash": "h2", "heading_path": "C"}),
    ]
    groups = chunker.group_blocks(docs)
    assert len(groups) == 2, "同文件小段落应被聚合，不同文件不应混合"
    assert groups[0].metadata["heading_paths"] == ["A", "B"]
    assert groups[0].metadata["blocks_merged"] == 2


def test_make_doc_id_is_stable_and_content_sensitive():
    assert make_doc_id("a.pdf", "hash1") == make_doc_id("a.pdf", "hash1")
    assert make_doc_id("a.pdf", "hash1") != make_doc_id("a.pdf", "hash2")
