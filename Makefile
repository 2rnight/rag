.PHONY: help install dev serve ingest reset test eval bench lint clean docker

PY ?= python
PORT ?= 8000

help:
	@echo "RAG 知识库问答系统 - 常用命令"
	@echo "  make install   安装依赖"
	@echo "  make ingest    索引 docs/ 目录（增量、幂等）"
	@echo "  make reset     清空并全量重建索引"
	@echo "  make serve     启动 API 服务（http://localhost:$(PORT)）"
	@echo "  make dev       安装 + 索引 + 启动"
	@echo "  make test      运行 pytest（离线、确定性）"
	@echo "  make eval      私域测试集评测"
	@echo "  make bench     TTFT 压测"
	@echo "  make docker    构建并启动容器"
	@echo "  make clean     清理运行期数据（data/）"

install:
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) -m scripts.ingest

reset:
	$(PY) -m scripts.ingest --reset

serve:
	$(PY) -m uvicorn app:app --host 0.0.0.0 --port $(PORT)

dev: install ingest serve

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m scripts.evaluate

bench:
	$(PY) -m scripts.benchmark_ttft --repeat 3

clean:
	rm -rf data .pytest_cache __pycache__
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

docker:
	docker compose up -d --build
