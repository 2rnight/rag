"""API 层集成测试（FastAPI TestClient，含 SSE 事件流解析）。"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(pipeline):
    """复用已索引 docs/ 的流水线，并挂到 app.state 上。"""
    from app import app

    with TestClient(app) as test_client:
        test_client.app.state.pipeline = pipeline
        yield test_client


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        name, payload = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload += line[5:].strip()
        if payload:
            events.append((name, json.loads(payload)))
    return events


def test_health_and_config_endpoints(client):
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["chunks"] >= 0

    config = client.get("/api/v1/config").json()
    assert config["chunk_overlap_ratio"] == 0.2
    assert "dashscope_api_key" not in config, "公开配置绝不能泄露密钥"


def test_knowledge_status_and_documents(client):
    status = client.get("/api/v1/knowledge/status")
    assert status.status_code == 200
    body = status.json()
    assert body["knowledge_base"]["chunks"] > 0
    assert "embedding" in body and "llm" in body

    documents = client.get("/api/v1/knowledge/documents").json()
    assert documents["chunks_total"] > 0
    assert documents["documents"]


def test_upload_and_delete_document(client, tmp_path):
    markdown = "# 测试制度\n\n## 第一章 门禁\n门禁卡补办需要 3 个工作日，费用 20 元。\n"
    file_path = tmp_path / "门禁制度.md"
    file_path.write_text(markdown, encoding="utf-8")

    with file_path.open("rb") as handle:
        response = client.post(
            "/api/v1/knowledge/upload",
            files={"files": (file_path.name, handle, "text/markdown")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["chunks"] >= 1
    assert payload["vectors_total"] > 0

    deleted = client.delete("/api/v1/knowledge/documents/门禁制度.md")
    assert deleted.status_code == 200
    assert deleted.json()["removed_chunks"] >= 1

    missing = client.delete("/api/v1/knowledge/documents/不存在的文件.md")
    assert missing.status_code == 404


def test_unsupported_upload_returns_reason(client, tmp_path):
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\x00\x01\x02")
    with path.open("rb") as handle:
        response = client.post(
            "/api/v1/knowledge/upload", files={"files": (path.name, handle, "application/octet-stream")}
        )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["skipped"] and "不支持的文件类型" in result["reason"]


def test_chat_and_search_endpoints(client):
    chat = client.post("/api/v1/chat", json={"question": "年假有多少天？"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["answer"]
    assert body["references"] and body["references"][0]["index"] == 1
    assert body["provider"]

    search = client.post("/api/v1/search", json={"question": "公司WiFi密码是多少？", "top_k": 3})
    assert search.status_code == 200
    hits = search.json()["results"]
    assert hits and hits[0]["source"] == "常见问题FAQ.txt"


def test_chat_stream_sse_protocol(client):
    response = client.post("/api/v1/chat/stream", json={"question": "公积金缴纳比例是多少？"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "open"
    assert "meta" in names and "done" in names
    assert names.index("meta") < names.index("done")

    meta = next(data for name, data in events if name == "meta")
    assert meta["sources"] and meta["top_score"] > 0
    done = next(data for name, data in events if name == "done")
    assert done["answer"] and done["ttft_ms"] > 0


def test_chat_stream_get_endpoint(client):
    """EventSource 兼容接口（GET + query 参数）。"""
    response = client.get("/api/v1/chat/stream", params={"question": "迟到会怎么扣钱？"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert any(name == "delta" for name, _ in events)


def test_out_of_scope_question_is_refused_via_api(client):
    response = client.post("/api/v1/chat", json={"question": "帮我写一首关于宇宙的诗"})
    body = response.json()
    assert body["refused"] is True
    assert body["references"] == []


def test_metrics_endpoint(client):
    client.post("/api/v1/chat", json={"question": "年假有多少天？"})
    metrics = client.get("/api/v1/metrics").json()["metrics"]
    assert metrics["requests"]["chat"] >= 1
    assert "ttft" in metrics["latency_ms"]
    assert metrics["retrieval_quality"]["samples"] >= 1
