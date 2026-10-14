.PHONY: help install sync dev serve serve-uv ingest reset test eval bench cheatsheet lint clean docker

PY ?= python
UV ?= uv
PORT ?= 8000

help:
	@echo "RAG 知识库问答系统 - 常用命令"
	@echo "  make install   安装依赖（pip）"
	@echo "  make sync      安装依赖（uv sync，按 uv.lock 可复现）"
	@echo "  make ingest    索引 docs/ 目录（增量、幂等）"
	@echo "  make reset     清空并全量重建索引"
	@echo "  make serve     启动 API 服务（http://localhost:$(PORT)）"
	@echo "  make dev       安装 + 索引 + 启动"
	@echo "  make test      运行 pytest（离线、确定性）"
	@echo "  make eval      私域测试集评测"
	@echo "  make bench     TTFT 压测"
	@echo "  make cheatsheet 生成 A4 单页速查表（PDF，需 reportlab + 中文字体）"
	@echo "  make docker    构建并启动容器"
	@echo "  make clean     清理运行期数据（data/）"

install:
	$(PY) -m pip install -r requirements.txt

sync:
	$(UV) sync

ingest:
	$(PY) -m scripts.ingest

reset:
	$(PY) -m scripts.ingest --reset

serve:
	$(PY) -m uvicorn app:app --host 0.0.0.0 --port $(PORT)

serve-uv:
	$(UV) run uvicorn app:app --host 0.0.0.0 --port $(PORT)

dev: install ingest serve

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m scripts.evaluate

bench:
	$(PY) -m scripts.benchmark_ttft --repeat 3

cheatsheet:
	$(PY) -m scripts.build_cheatsheet

clean:
	rm -rf data .pytest_cache __pycache__
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

docker:
	docker compose up -d --build
