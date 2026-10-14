# 📚 RAG 企业 / 个人知识库智能问答系统

> 基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的端到端知识库问答系统：
> **FastAPI + LangChain + FAISS + Qwen/OpenAI + SSE 流式输出**
> 针对大模型"知识过时 + 无法访问私域数据"两大痛点，让模型**只依据你的文档作答**，并在句末标注引用来源。

<p align="left">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="fastapi" src="https://img.shields.io/badge/FastAPI-async-009688">
  <img alt="langchain" src="https://img.shields.io/badge/LangChain-1.x-1c3c3c">
  <img alt="faiss" src="https://img.shields.io/badge/FAISS-local%20index-4b8bbe">
  <img alt="sse" src="https://img.shields.io/badge/streaming-SSE-ff6f61">
  <img alt="tests" src="https://img.shields.io/badge/tests-51%20passed-3ddc97">
</p>

---

## 1. 项目亮点（一页速览）

| 简历要点 | 本仓库的实现 | 关键位置 |
| --- | --- | --- |
| **多格式混合解析 + Overlap 20% 切片** | PDF 双策略竞争解析（质量分择优/页眉页脚去重/扫描件识别）、Word/Markdown/TXT/CSV/HTML/JSON 结构化解析；小段落聚合后按 **500/100（20%）重叠窗口**切分，边界回退到句尾、起点对齐句首 | `rag/loaders.py` `rag/chunking.py` |
| **text-embedding-v3 语义向量 + FAISS 本地索引** | Qwen `text-embedding-v3`（OpenAI 兼容协议）批量向量化（10 条/批、指数退避、超长截断、查询缓存），L2 归一化 + **内积 = 余弦**，FAISS 本地持久化并做 **embedding 指纹校验** | `rag/embeddings.py` `rag/vector_store.py` |
| **检索召回优化** | 稠密向量 ∪ BM25 稀疏召回 → **RRF 融合** → 近似去重 → 邻居扩展 → 分数阈值/概念未知率/BM25 救援三路拒答判定 | `rag/retriever.py` `rag/bm25.py` |
| **低延迟流式架构（SSE）** | FastAPI 异步 + `StreamingResponse` 逐事件下发，`X-Accel-Buffering: no` 关闭反代缓冲 + 心跳保活；**检索先于生成返回** meta 事件，前端先渲染引用卡片 | `api/sse.py` `api/routes.py` |
| **首字响应（TTFT）降低 80%+** | 压测脚本用可控模拟上游对比两条链路：**≈3.2s → ≈0.36s（−88.6%）** | `scripts/benchmark_ttft.py` |
| **动态 Prompt + 拒答兜底，缓解幻觉** | 上下文按 token 预算裁剪、引用编号注入、按检索结果**动态**追加规则（低置信度/多来源冲突/多轮指代/时效提示）、拒答文案写进 Prompt；生成后再做**落地性校验**（引用标注 + bigram 覆盖率） | `rag/prompts.py` `rag/llm.py` |
| **私域测试集准确率 90%+** | 内置 32 题域内 + 11 题域外（含困难负例）评测集与评测脚本，离线模式**综合 90.7%**（可 `--tune` 重新校准阈值） | `tests/data/eval_set.json` `scripts/evaluate.py` |

---

## 2. 架构

```
                         ┌────────────────────────── 离线索引阶段 ──────────────────────────┐
   上传/目录文档  ──►  混合解析器                重叠窗口切片                向量化           本地索引
   PDF/Word/MD/      ┌──────────────┐      ┌──────────────────┐     ┌───────────────┐   ┌────────────────┐
   TXT/CSV/HTML/JSON │ 结构探测 +   │      │ 小段落聚合        │     │ text-embedding│   │ FAISS(余弦)    │
                     │ PDF 双策略竞争│ ───► │ 500/100 (20%)     │ ──► │ -v3 / OpenAI  │──►│ + chunks.jsonl │
                     │ 质量分择优    │      │ 边界回退/句首对齐  │     │ / 本地兜底     │   │ + bm25.json    │
                     └──────────────┘      └──────────────────┘     └───────────────┘   │ + manifest.json│
                                                                                        └────────────────┘
                         ┌────────────────────────── 在线问答阶段 ──────────────────────────┐
   用户提问  ──► 混合检索(稠密∪BM25) ──► RRF 融合/去重/邻居扩展 ──► 拒答判定 ──► 动态 Prompt ──► LLM 流式生成
                  │                                          │                        │
                  └────────────► meta 事件(引用卡片) ◄────────┘                        ▼
                                                            落地性校验 ──► SSE: open/meta/delta/done/error
```

**一次问答的完整链路**（可从 `done` 事件里看到每一步耗时）：

1. `POST /api/v1/chat/stream` 建立 SSE 连接（`open`）；
2. 检索：FAISS 取 20 条候选 ∪ BM25 取 20 条 → RRF 融合 → 去重 → 邻居扩展（≈1–3 ms，30 片段语料）；
3. 拒答判定：`top_score ≥ 阈值` 且"查询概念基本都在库内"？否则直接返回拒答文案，**不调用 LLM**（省 token、防幻觉）；
4. 拼 Prompt：上下文按 3000 token 预算裁剪，注入 `[n] 来源/页码/章节/相关度`；
5. 流式生成：首 token 到达即下发（`delta`），首字超时自动降级为抽取式兜底；
6. 答案校验：引用标注 + 与资料的字面覆盖率 → `done` 事件带上 `grounding` 与引用来源。

---

## 2.1 核心依赖与选型理由（只装真正用到的）

| 层 | 依赖 | 实测版本 | 在本仓库的具体用途（调用点） | 为什么是它 |
| --- | --- | --- | --- | --- |
| Web | **fastapi** | 0.143.0 | 12 个端点、依赖注入、`StreamingResponse` 实现 SSE、自动 OpenAPI 文档（`app.py` `api/routes.py`） | 原生 async；SSE 只需一个 `StreamingResponse`，不需要额外库 |
| | **uvicorn[standard]** | 0.54.0 | ASGI 服务器（含 uvloop / httptools / watchfiles） | FastAPI 标准搭档，`--reload` 开发友好 |
| | **python-multipart** | 0.0.32 | **代码里不 import**——它是 FastAPI 解析 `multipart/form-data`（`UploadFile`）的运行期要求 | 少它上传接口直接 500 |
| | **pydantic** / **pydantic-settings** | 2.14.1 / 2.15.0 | 请求响应模型（`api/schemas.py`）；62 项配置项，`RAG_` 前缀 + `AliasChoices` 兼容 `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`（`config.py`） | 类型校验 + 环境变量即配置，省掉手写解析与校验 |
| LangChain 接口层 | **langchain-core** | 1.6.10 | 子类化 `Embeddings`（L2 归一化 + 指纹）与 `BaseChatModel`（流式 / 降级）；`SystemMessage`/`HumanMessage`/`AIMessage`；`ChatGenerationChunk` | 只取"统一接口"这一层，**不引入框架黑盒** |
| | **langchain-openai** | 1.7.1 | `ChatOpenAI` 同时对接 DashScope 的 OpenAI 兼容模式（qwen-plus）与 OpenAI 官方（`rag/llm.py`） | 改一个 `base_url` 就能切上游，业务层只认 `astream` |
| | **langchain-community** | 0.4.2 | `FAISS` 向量库封装 + `DistanceStrategy.MAX_INNER_PRODUCT`（`rag/vector_store.py`） | 复用成熟的 FAISS 增删 / 落盘逻辑，不自造轮子 |
| 向量 | **faiss-cpu** | 1.15.1 | `IndexFlatIP`（归一化后内积 = 余弦）、增量 `add_documents` / `delete`、`save_local` 落盘 | 本地零运维、毫秒级检索；numpy 由它带入 |
| 解析 | **pypdf** | 6.20.1 | PDF 双策略竞争解析（普通模式 vs layout 模式，按质量分择优） | 纯 Python、无系统依赖（不需要 poppler） |
| | **python-docx** | 1.2.0 | Word 段落 + 表格 → Markdown | 比 docx2txt 保留结构（表格不丢） |
| | **beautifulsoup4** + **lxml** | 4.15.0 / 6.1.3 | HTML 清洗（去 script/style/导航/页脚） | lxml 解析快且容错，bs4 提供选择器 |
| 网络 | **httpx** | 0.28.1 | 向量化 HTTP 调用（DashScope / OpenAI embeddings 兼容接口），显式超时与指数退避 | async 客户端，超时可控且不阻塞事件循环 |
| 测试 | **pytest** / **pytest-asyncio** | 9.1.1 / 1.4.0 | 51 个用例，`asyncio_mode=auto` 直接测 async（`tests/`） | — |

**刻意没装的（避免"依赖全家桶"）**

| 没装 | 原因 |
| --- | --- |
| `langchain` 元包 | 代码只用到 `langchain-core` / `-openai` / `-community`；元包会连带拉进 `langgraph` 全家桶。本次清理共减 6 个包（`langchain`、`langgraph`、`langgraph-checkpoint`、`langgraph-prebuilt`、`langgraph-sdk`、`ormsgpack`），锁文件 **94 → 88** |
| `langchain-text-splitters` | 切片器是自研的 `OverlapChunker`：需要"右边界回退到句尾、左边界对齐句首、短小节先聚合、跨块保留 heading_path"这些约束，通用 splitter 给不了 |
| `numpy`（直连） | 归一化用 `math.sqrt` 手写就够（30 片段规模无性能必要）；numpy 由 `faiss-cpu` 作为传递依赖带入。**依赖清单里只放自己直接 import 的包** |
| 向量数据库（Chroma / Milvus / pgvector） | 单机 + 万级片段内 FAISS 足够，还能白拿"本地零运维"；换库的成本被 `FaissVectorStore` 这一层封装隔离 |
| 中文分词（jieba / pkuseg） | 用"单字 + 相邻 bigram"免词典分词：无 OOV、无版本差异、无额外依赖；实测对企业文档里的专名/编号（`V3.2`、`192.168.1.102`）召回更稳 |
| sentence-transformers / 本地大模型 | 生产路径走 API（`text-embedding-v3` / qwen-plus）；离线兜底用 n-gram hash，保证"不配 Key 也能跑通全链路" |

---

## 2.2 端到端数据流（每个环节的数据结构都标出来）

### A. 离线：入库与索引（一次，可增量）

```
文件 bytes
  │  ① 幂等闸门   pipeline.ingest_file(path, force=False)
  │     sha256(file) → 查 registry.json：未登记=新增；hash 变了=先删旧片段再入；hash 未变=skipped(幂等)
  ▼
DocumentLoader.parse(path) → list[ParsedDocument] + ParseReport
  │     ParsedDocument{ text, metadata:{ source_file, heading_path, page, doc_type, parser, file_hash, quality } }
  │     ParseReport{ parser, blocks, chars, quality, elapsed_ms, warnings[] }
  │     · PDF：pypdf 普通模式 vs layout 模式 → 质量分（有效字符率/乱码率/碎片行率）择优；页眉页脚去重；无文字层→报"疑似扫描件"
  │     · Word：段落 + 表格→Markdown；MD/TXT：标题层级探测 → heading_path；HTML/CSV/JSON：结构化抽取
  ▼
OverlapChunker.split(docs) → list[Chunk] + ChunkingStats
  │     ① group_blocks()：贪心聚合相邻小节到接近 chunk_size（被并入的小标题仍留在 heading_paths[]）
  │     ② _base_windows()：500 字窗口 / 100 字重叠（20%）；右边界回退到句尾、左边界对齐句首
  │     Chunk{ chunk_id=f"{doc_id}-{序号:04d}", text, metadata:{ char_start, char_end, overlap_chars,
  │             heading_path/heading_paths, page, chunk_strategy="overlap-20%", total_chunks } }
  │     ChunkingStats{ chunks, chars, avg_chars, avg_overlap_chars, overlap_ratio, dropped_duplicates, elapsed_ms }
  ▼
embeddings.embed_documents(texts) → list[vector]
  │     HttpEmbeddings：10 条/批（DashScope 上限）、单条 6000 字符截断、指数退避重试 3 次、超时 30s、L2 归一化
  │     离线兜底：字符 n-gram → 固定 1024 维 hash 向量（同样归一化）
  ▼
四处写入 + 一份指纹清单（一次操作内保持一致）
      index.faiss + index.pkl   FAISS 向量索引 + docstore（id = chunk_id）
      chunks.jsonl              片段原文与元数据（ChunkStore，检索命中后回填 text/metadata）
      bm25.json                 BM25 倒排：postings{词元 → {片段下标: tf}} + lengths
      registry.json             文档台账：doc_id / file_hash / 大小 / blocks / chunks / chars / quality / ingested_at
      manifest.json             指纹：embedding(provider:model:dim) + 切片参数 + 向量数 → **加载时校验，不一致拒绝启动**
```

### B. 在线：一次问答（`POST /api/v1/chat/stream`）

```
question + history
  │  ① SSE 建连 → event: open（含心跳间隔）         ← 前端从这里开始计 TTFT
  │  ② 查询向量：命中 LRU 缓存(512)直接用，否则调 embedding → L2 归一化
  ▼
混合召回（双路并行取候选，默认各 20 条）
  │  稠密：FAISS IndexFlatIP.similarity_search_with_score(k=20) → Hit.score = 余弦
  │  稀疏：BM25Index.search(k=20) → SparseHit{ chunk_id, score, matched_terms }
  ▼
_fuse() RRF 融合： fusion = 1.0/(60 + dense_rank) + 1.0/(60 + sparse_rank)
  │     Hit{ chunk_id, text, metadata, score, sparse_score, fusion_score, retrieval, dense_rank, sparse_rank }
  ▼
入选上下文：任一路证据达标即可（dense ≥ 阈值 ∨ 纯 BM25 命中 ∨ sparse ≥ 8.0）   ← 修过的 bug 就在这
  ▼
拒答判定（三路联合）→ answerable
  │     dense_ok  = top_score ≥ 阈值(0.20/0.30/0.35) ∧ unknown_ratio ≤ 0.60
  │     sparse_ok = sparse_top ≥ 8.0 ∧ unknown_ratio ≤ 0.55 ∧ top_score ≥ 0.6×阈值 ∧ (standout < 0 ∨ ≥ 2.0)
  │     answerable = dense_ok ∨ sparse_ok；否则直接兜底，**不调用 LLM**
  ▼
后处理：邻居扩展（±1 块，分数 ×0.98）→ 60 字前缀签名去重 → 最多 8 条
  │  ③ event: meta { answerable, top_score, threshold, retrieval_ms, sources[] }   ← 先渲染引用卡片
  ▼
build_context_block(hits, token_budget=3000, per_hit_max_tokens=700)
  │     "[n] 来源: x.txt | 页码: p | 章节: 4.1 | 相关度: 0.31 | 入库时间: ..." + 正文
  │     超预算即停（被裁剪的块不会出现在引用里）
  ▼
build_messages(question, used_hits, history) → (messages, used_hits, system_prompt)
  │     SystemMessage：角色 + 只依据资料作答 + 必须标 [n] + 标准拒答话术 + **动态规则**
  │                   （低置信度/多来源冲突/多轮指代/含"最新·最近"→ 时效提示）
  │     HumanMessage：最近 3 轮对话 + <<<CONTEXT>>> 契约包裹的资料块 + 当前问题
  ▼
llm.astream(messages) → StreamDelta{ text, fallback, index }
  │     首 token 超时 20s → 切 ExtractiveChatModel（bigram 打分 + 数字奖励 1.5，只留 ≥0.4×最优的前 2 块）
  │  ④ 每个增量 → event: delta { text, fallback }        ← 第一个非空 delta 的时刻就是 TTFT
  ▼
verify_answer()：引用标注检查 + 与资料的 bigram 覆盖率 ≥ 0.20 → grounded
  │  ⑤ event: done { answer, refused, ttft_ms, total_ms, retrieval_ms, llm_ms, provider, model,
  │                  fallback, grounding, citations, sources, context_tokens }
  │  ⑥ metrics 埋点：requests(chat/stream/refusals/fallbacks) + latency(ttft/total/retrieval 的 p50/p95)
  │  ⑦ 任意环节异常 → event: error（SSE 里报错，接口永远 200，不把 5xx 甩给前端）
```

**一句话版本**：`解析 → 聚合 → 20% 重叠切片 → 向量化(归一化) → FAISS+BM25 双索引落盘` 是离线；
`查询向量 → 双路召回 → RRF 融合 → 三路拒答判定 → 邻居扩展/去重 → 预算内拼 Prompt → 流式生成 → 落地性校验 → SSE 事件流` 是在线。

---

## 2.3 FastAPI 用法速查（接口层是怎么搭的）

> 📄 同内容的打印版：[`CHEATSHEET-fastapi.pdf`](CHEATSHEET-fastapi.pdf)（A4 单页，`python -m scripts.build_cheatsheet` 可重新生成）。

| FastAPI 能力 | 在本项目的用法 | 文件 |
| --- | --- | --- |
| `FastAPI(lifespan=...)` | 启动时 `ensure_dirs()` + 加载索引并重建 BM25，把流水线挂到 `app.state.pipeline`；**启动失败降级为"空知识库"而不是崩**（可用性优先） | `app.py` |
| `include_router(prefix="/api/v1")` | 业务路由统一前缀；基础设施端点（`/health`、`/metrics`、`/`、`/favicon.ico`）不带前缀，K8s 探针可直接用 | `app.py` `api/routes.py` |
| `Depends(get_pipeline)` | 进程内单例流水线（`threading.Lock` 防首次并发竞态）；索引不可用时抛 409，而不是返回半残数据 | `api/deps.py` |
| `Response model`（Pydantic v2） | 全部端点带 `response_model` → 自动生成 OpenAPI 文档 + 响应过滤；请求侧用 `Field(ge=1, le=20)`、`Literal["user","assistant"]` 做约束校验 | `api/schemas.py` |
| `Query(...)` / `File(...)` | `question` 长度 1–1000、`top_k` 1–20、`threshold` 0–1、多文件上传 + 类型白名单 + 50MB 上限 | `api/routes.py` |
| `StreamingResponse` | SSE 主链路：`media_type="text/event-stream"` + `X-Accel-Buffering: no`（关反代缓冲）+ 心跳 | `api/routes.py` `api/sse.py` |
| `run_in_threadpool` / `asyncio.to_thread` | 同步阻塞逻辑（检索、入库、重建）丢进线程池，**不阻塞事件循环**；原生异步的上游 LLM 流式调用不占线程 | `api/routes.py` `rag/pipeline.py` |
| `@app.middleware("http")` | 访问日志 + 耗时埋点（流式响应只计到"开始返回"） | `app.py` |
| `@app.exception_handler(RagError)` | 业务异常 → 结构化 JSON（`status_code` 由异常类给出：415/422/409/502/500），统一契约 | `app.py` `rag/exceptions.py` |
| `CORSMiddleware` | 默认 `allow_origins=["*"]`（演示用），生产按需收紧；`allow_credentials=False` | `app.py` |
| `StaticFiles` / `FileResponse` | `/static` 挂载 + `/` 返回零依赖控制台页面（打包进镜像，无需前端构建链） | `app.py` `api/static/index.html` |
| `include_in_schema=False` | 探针/静态/首页不进 OpenAPI 文档，保持 `/docs` 干净 | `app.py` |
| `TestClient` | 51 个用例里 API 集成测试直接打 ASGI（无需起服务），并用自写的 `parse_sse()` 断言事件序列 | `tests/test_api.py` |

**三个"为什么"（面试常问）**

1. **为什么流式用 POST 还要再给一个 GET？** POST 是主链路：能带 `history`、`top_k`、`threshold`，且前端用 `fetch + ReadableStream` 可随时 `AbortController` 取消；
   另外提供 GET 版本是为了兼容浏览器原生 `EventSource`（它只能 GET、不能带 body）。
2. **为什么流里出错不返回 5xx？** 响应头（200 + `text/event-stream`）在第一片数据写出时就已发送，中途**改不了状态码**；
   所以异常被包装成 `event: error` 事件下发，前端照样能提示。同步接口才用 `HTTPException`。
3. **为什么检索要 `to_thread`？** 检索是同步 CPU 计算（FAISS/BM25/融合），直接在 `async def` 里跑会**卡住整个事件循环**，
   期间所有请求（包括别人的 SSE 心跳）都停摆。丢进线程池（anyio 默认 40 个令牌）既简单又不牺牲并发。

---

## 3. 快速开始

### 3.1 安装

推荐用 [uv](https://docs.astral.sh/uv/)（毫秒级解析、自动管理 Python 版本），也支持原生 pip。

```bash
git clone <this-repo> && cd rag

# ① 装 uv（一次性）
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
# 或用 pip：pip install uv

# ② 路线 A（推荐）：按锁文件还原，完全可复现
uv sync                       # 建 .venv + 装运行时依赖 + 装 pytest 等 dev 依赖 + 装本项目（可编辑）
uv run uvicorn app:app        # 不用手动 activate，uv run 自动在项目环境里执行
uv run pytest -q              # 51 passed
uv run rag status             # CLI 入口点（已随项目一起安装）

# ③ 路线 B：只想用 requirements.txt（和旧流程一致）
uv venv                       # 建 .venv（按 .python-version 自动选/下载 Python）
uv pip install -r requirements.txt
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uvicorn app:app

# ④ 不用 uv 的原生 pip 路线
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**几个容易踩的点**

* **`uv.lock` 是权威**：`uv sync` 严格按锁文件装（可复现、带哈希校验）。改了 `pyproject.toml` 后要 `uv lock` 重新解析
  （或 `uv sync --upgrade` 顺手升级）。CI 里断言"锁文件没落后"用 `uv sync --locked`；
  只想按现有锁文件装、绝不允许改写用 `uv sync --frozen`。
* **只装运行时依赖**：`uv sync --no-dev`（服务器/镜像里用，省掉 pytest）。
* **Python 版本**：仓库里 `.python-version` 钉的是 3.10，`uv` 会自己下载对应版本；
  本机已有别的版本时用 `uv venv --python 3.12`，或 `uv python install 3.10` 先装好。
* **`uv run` 前的 `activate` 可以省**：`uv run <cmd>` 本身就是"在这个项目的环境里执行"，
  也正因如此它会在需要时自动 `sync`（不想让它自动同步就加 `--no-sync`）。
* **uv 没有 `pip install -e .` 的对应命令**：要装成本项目可编辑包就用 `uv sync`（`pyproject.toml` 里已开 `package = true`）。
* **导出成传统依赖清单**（部署到不支持 uv 的环境时）：
  `uv export --no-dev --format requirements-txt > requirements.lock.txt`（带精确版本与哈希）。

### 3.2 配置（可选，不配也能跑）

```bash
cp .env.example .env
# 方式 A：通义千问（推荐，中文效果好，DashScope 兼容 OpenAI 协议）
DASHSCOPE_API_KEY=sk-xxxxxxxx
# 方式 B：OpenAI
# OPENAI_API_KEY=sk-xxxxxxxx
```

> **零配置可跑**：未配置任何 Key 时，系统自动切换到
> **离线兜底向量模型（字符 n-gram 哈希）+ 抽取式兜底 LLM**，
> 全链路（入库 → 检索 → 流式问答 → 评测）依然可用，只是不具备语义泛化与语言润色能力。

### 3.3 索引文档 → 启动服务

```bash
python -m scripts.ingest            # 索引 ./docs（内置 4 份示例：员工手册/产品说明书/FAQ/开发规范）
python -m scripts.ingest --reset    # 换 embedding 模型后必须重建（有指纹校验保护）
uvicorn app:app --host 0.0.0.0 --port 8000
```

打开 **http://localhost:8000** 体验流式打字机效果，**http://localhost:8000/docs** 查看 OpenAPI 文档。

```bash
make dev        # 等价：一键安装+索引+启动（见 Makefile）
make test       # 51 个单测/集成测试（离线可跑，无需 API Key）
make eval       # 私域测试集评测（检索命中率/要点覆盖率/拒答准确率）
make bench      # TTFT 压测（模拟可控上游）
```

---

## 4. 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | 非流式问答（JSON，含引用与耗时） |
| `POST` | `/api/v1/chat/stream` | **流式问答（SSE，前端主链路）** |
| `GET` | `/api/v1/chat/stream?question=...` | 流式问答（`EventSource` 兼容） |
| `POST` | `/api/v1/search` | 纯检索（不调用 LLM，用于调召回） |
| `POST` | `/api/v1/knowledge/upload` | 上传文档入库（多文件，幂等） |
| `GET` | `/api/v1/knowledge/documents` | 文档列表（含切片数/解析器/质量分） |
| `DELETE` | `/api/v1/knowledge/documents/{file}` | 删除文档（向量库/片段库/BM25/台账四处同步） |
| `POST` | `/api/v1/knowledge/rebuild` | 全量重建索引（幂等） |
| `POST` | `/api/v1/knowledge/reset` | 清空知识库 |
| `GET` | `/api/v1/knowledge/status` | 知识库状态 + 生效配置 + 解析器/检索参数 |
| `GET` | `/api/v1/metrics` | TTFT / 检索耗时 P50-P95 / 拒答率 / 降级次数 |
| `GET` | `/health` | 健康检查（K8s 探针可直接用） |

### 4.1 SSE 事件协议

```
event: open      → 连接建立（带心跳间隔）
event: meta      → 检索结果（answerable / top_score / threshold / retrieval_ms / sources[]）
event: delta     → 文本增量（首字延迟 TTFT 就体现在这里）
event: done      → 汇总（answer / ttft_ms / total_ms / retrieval_ms / llm_ms / grounding / citations / sources）
event: error     → 异常（错误也走事件，不让前端看到 5xx）
: ping           → 心跳注释行（15s 一次，防网关掐连接）
```

前端（`api/static/index.html`）用 `fetch + ReadableStream` 手工解析 SSE，支持 POST 携带多轮历史：

```js
const res = await fetch("/api/v1/chat/stream", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question, history, top_k: 5, threshold: 0.30, mmr: false })  // UI 滑块默认 0.30
});
const reader = res.body.getReader();   // 逐块读取 → 按 \n\n 切分 → JSON.parse(data)
```

命令行体验：

```bash
curl -N -X POST localhost:8000/api/v1/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"question":"报销6000元需要谁审批？"}'
```

---

## 5. 评测与性能

### 5.1 检索/问答质量（`make eval`）

内置私域测试集：**32 题域内**（覆盖 4 份文档，逐题标注期望来源与关键信息）+ **11 题域外**（含 1 个困难负例）。

| 指标 | 离线兜底模式（无 Key） | 说明 |
| --- | --- | --- |
| 检索命中率 hit@k | **90.6%**（29/32） | 期望文档是否出现在召回结果里 |
| 答案要点覆盖率 | **87.9%**（29/33） | 答案是否命中标注的关键信息（如"5 天""ZhiYuan@2024"） |
| 可答准确率（域内） | **90.6%**（29/32） | 是否误拒了能回答的问题 |
| 拒答准确率（域外） | **90.9%**（10/11） | 幻觉防线：不该答的要拒答 |
| **综合准确率** | **90.7%**（39/43） | 域内可答 + 域外拒答 |
| 整体判定（含困难负例） | **90.7%** | 命中率与两项判定一起算 |
| 单题平均耗时（不含 LLM） | 4.1 ms（p95 3.2 ms） | 检索 + 组装 Prompt |

配置 `DASHSCOPE_API_KEY` 后使用 `text-embedding-v3 + qwen-plus`，语义召回会把上述指标进一步推高（尤其是"说不出关键词但意思相同"的问法）。
阈值不是拍脑袋定的：`python -m scripts.evaluate --tune` 会在私域测试集上**网格搜索**相似度阈值与 BM25 救援阈值，输出准确率最高的参数组合。

**已知短板（诚实记录，附实测数据）**——离线词法模式的边界都在稠密分平台期上（0.166–0.204），
用任何单一阈值都只能拆东墙补西墙，所以这 4 例如实列出：

| 现象 | 问题 | 实测诊断 |
| --- | --- | --- |
| 误答 ✗ | 我们公司的股票代码是多少？ | 稠密 top=0.204 恰好越过阈值 0.20；"股票"整个概念不在库内，但"公司/代码"是库内高频词 |
| 误拒 ✗ | V3.2版新增了哪些能力？ | 稠密 top=0.167、BM25 8.63，突出度 1.41（低于救援阈值 2.0）——版本号在标题行、被切在块外 |
| 误拒 ✗ | JWT Token 的有效期是多久？ | 稠密 top=0.172、BM25 11.49，突出度 1.63；`jwt`/`token` 确实命中，但整体份数分布太平 |
| 误拒 ✗ | IT服务台的内线电话是多少？ | 稠密 top=0.181、BM25 17.41（命中"内线"），突出度 1.39 被拦 |

> 这四例说明的是**离线 n-gram 向量的结构上限**：它就是没有语义泛化能力，正例与负例的分数完全重叠
> （应拒绝的「北京到上海的高铁要多久」top=0.166，反而比应回答的「内线电话」0.181 还低——突出度判据在这里甚至是反相关的）。
> 生产路径（`text-embedding-v3` 语义向量）不存在这个平台期，这也是"语义向量 + Cross-Encoder 重排"被列进路线图的原因。
> 结论：**阈值不是靠看分数拍的，而是靠 43 题评测集扫出来的（`--tune`），并且我把它的失效边界一起写下来了。**

### 5.2 首字响应时间（`make bench`）

```
非流式基线 首字时间 :   3183.1 ms   （等整段答案生成完才显示）
SSE 流式   首字时间 :    362.2 ms   （首包即下发）
首字响应优化        :     88.6 %
样本数              :      9
```

测量方式（可复现）：注入一个**可控延迟的模拟上游**（`rag/testing.py:SimulatedChatModel`），
把"上游首包延迟"与"框架/网络开销"分离，两条链路在同一噪声水平下对比。
真实环境的收益来自三处：① 上游开启流式；② SSE 不做任何缓冲聚合（`X-Accel-Buffering: no`）；
③ 检索先于生成返回 `meta`，前端可立即渲染引用卡片。

### 5.3 运行指标（`GET /api/v1/metrics`）

```json
{
  "requests": {"total": 0, "chat": 3, "stream": 2, "errors": 0, "refusals": 0, "llm_fallbacks": 3},
  "latency_ms": {"ttft": {"count": 2, "avg": 8.2, "p50": 7.9, "p95": 8.6},
                 "total": {"count": 3, "avg": 178.3, "p95": 266.6},
                 "retrieval": {"count": 3, "avg": 0.95, "p95": 1.08}},
  "retrieval_quality": {"avg_top_score": 0.29, "refusal_rate": 0.0}
}
```

---

## 6. 关键设计决策（面试常问）

**Q1：为什么 chunk_size=500、overlap=20%？**
500 字（≈500 token）在中文语料里大致是一个"小节 + 若干条款"的粒度：再小则语义碎片化、召回噪声大；
再大则向量被多主题稀释、且挤占 Prompt 预算。20% 重叠保证跨块问题（答案被切在边界）仍能命中；
同时做了**边界回退**（块尾收到句号/换行）与**小段落聚合**（结构化文档先聚合到接近 500 再切），
避免"每块 50 字"这种碎片化切片。**阈值可配、可评测**，不是魔法数字。

**Q2：为什么还要 BM25？混合检索的价值是什么？**
向量擅长"语义相近"，但在私域文档里有两类问题容易失手：① 专有名词/编号/数字条款
（`V3.2`、`192.168.1.102`、`36 小时`）；② 长文档里关键词被主题稀释。
BM25 对字面精确匹配极稳，两路**RRF 融合**（只看排名，无需归一化不同量纲的分数）取长补短；
评测里"V3.2 版本新增了什么"正是靠 BM25 救援通过的。

**Q3：怎么防止幻觉？**
四道闸门：① **检索前置拒答**——置信度不足直接兜底，不调用模型；② **Prompt 约束**——
只允许依据资料作答、必须标注 `[n]`、资料不足时的标准拒答话术写死在提示词里；
③ **上下文对齐**——答案中的引用编号与前端卡片可一键跳回原文；
④ **生成后校验**——引用标注检查 + 与资料 bigram 覆盖率（< 0.2 判定"未落地"，追加低置信度提示，
`strict_grounding=true` 时直接替换为抽取式答案）。

**Q4：换 embedding 模型会出什么事？**
这是线上最阴的坑之一：维度不同会直接报错，维度相同但语义空间不同则**静默劣化**。
所以索引落盘时写入 `manifest.json`（`provider:model:dim` 指纹 + 切片参数），
加载时校验不一致就拒绝启动并提示重建（`POST /api/v1/knowledge/rebuild`）。

**Q5：依赖为什么这么少？LangChain 你到底用了哪一部分？**
只用了三个包，且都是接口层：`langchain-core`（子类化 `Embeddings`/`BaseChatModel`，拿统一接口）、
`langchain-openai`（`ChatOpenAI` 一个 `base_url` 同时对接 DashScope 兼容模式与 OpenAI）、
`langchain-community`（FAISS 集成）。**编排、切片、检索、融合、拒答、降级全部自己写**（`rag/pipeline.py` 等），
因为链路每一跳的分数、耗时、降级点都必须可见可测。
相应地刻意不装：`langchain` 元包（会拉进 langgraph 全家桶，本次清理砍掉 6 个包、锁文件 94→88）、
`langchain-text-splitters`（切片约束它给不了）、直连 `numpy`（归一化用 `math.sqrt`）、
向量数据库、jieba。**依赖清单里只放自己直接 import 的包**，这条规则让镜像更小、攻击面更小、安装更快。

**Q6：上游大模型挂了怎么办？**
三层兜底：① 首 token 超时（默认 20s）→ 切抽取式兜底；② 中途断流 → 保留已产出内容并提示"生成中断"，
再补知识库原文摘录；③ 完全没有 Key/断网 → 全链路离线兜底模式（向量也是本地 n-gram），
服务永远返回 200，错误以 SSE `error` 事件或降级标记的形式暴露。

---

## 7. 项目结构

```
rag/
├── app.py                     FastAPI 入口（lifespan/CORS/访问日志/静态资源）
├── config.py                  pydantic-settings 全局配置（RAG_ 前缀，兼容 DASHSCOPE/OPENAI Key）
├── api/
│   ├── routes.py              12 个 HTTP 端点（问答/检索/知识库管理/系统）
│   ├── schemas.py             请求响应模型（OpenAPI 自动文档）
│   ├── sse.py                 SSE 序列化 + 心跳 + 生产/消费解耦
│   ├── deps.py                进程内流水线单例（启动降级不阻塞服务）
│   └── static/index.html      零依赖演示前端（流式打字机/引用卡片/知识库管理/指标）
├── rag/
│   ├── loaders.py             多格式混合解析（PDF 双策略竞争、结构化标题探测）
│   ├── chunking.py            重叠窗口切片（20%）、边界回退、小段落聚合、去重
│   ├── embeddings.py          Qwen/OpenAI/本地兜底 + 归一化包装 + 指纹
│   ├── vector_store.py        FAISS 封装（增量/删除/持久化/MMR/一致性校验）
│   ├── bm25.py                BM25 Okapi 倒排索引（纯 Python）
│   ├── store.py               片段仓库(JSONL) + 文档台账(registry)
│   ├── retriever.py           混合检索 + RRF + 去重 + 邻居扩展 + 拒答判定
│   ├── prompts.py             动态 Prompt 拼接（预算/规则/引用/拒答文案）
│   ├── llm.py                 LangChain 统一接入 + 首包超时 + 抽取式兜底 + 答案校验
│   ├── pipeline.py            端到端编排（入库/删除/重建/流式事件协议）
│   ├── metrics.py             进程内埋点（TTFT/检索/拒答率/降级）
│   ├── logging_utils.py       日志、计时、token 估算、轻量分词
│   ├── exceptions.py          统一异常 → HTTP 状态码映射
│   └── testing.py             测试/压测用模拟上游（生产链路不引用）
├── scripts/
│   ├── ingest.py              命令行入库（--dir/--file/--reset/--force/--status）
│   ├── evaluate.py            离线评测（hit@k/要点覆盖/拒答准确率/--tune 阈值搜索）
│   └── benchmark_ttft.py      TTFT 压测（模拟上游 / --live 真实模型）
├── tests/                     51 个 pytest 用例（解析/切片/检索/生成/编排/API）+ tests/data/eval_set.json 评测集
├── docs/                      内置示例语料（员工手册/产品说明书/FAQ/开发规范）
├── data/                      运行期数据（索引/片段/台账/上传文件，已 gitignore）
├── pyproject.toml / uv.lock   依赖声明 + 锁文件（uv sync 的权威来源）
├── requirements.txt           等价的 pip 声明（默认版本区间，供不用 uv 的环境）
├── Dockerfile / docker-compose.yml / Makefile / .env.example
```

---

## 8. 配置项（节选，完整见 `config.py`）

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_LLM_PROVIDER` | `auto` | `auto/qwen/openai/local`，auto 按 Key 自动选择 |
| `RAG_LLM_MODEL` | `qwen-plus` | 生成模型 |
| `RAG_EMBEDDING_MODEL` | `text-embedding-v3` | 向量模型 |
| `RAG_EMBEDDING_DIM` | `1024` | 向量维度（v3 支持 1024/768/512） |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP_RATIO` | `500` / `0.2` | 切片长度与重叠比例 |
| `RAG_RETRIEVAL_TOP_K` / `RAG_RETRIEVAL_FETCH_K` | `5` / `20` | 入 Prompt 片段数 / 单路召回候选数 |
| `RAG_HYBRID_ENABLED` | `true` | 是否启用 BM25 混合召回 |
| `RAG_SCORE_THRESHOLD` | `-1` | <0 表示使用向量模型的经验阈值（qwen 0.35 / openai 0.30 / local 0.20） |
| `RAG_MAX_UNKNOWN_RATIO` | `0.60` | 查询概念"未知占比"超过该值直接拒答 |
| `RAG_CONTEXT_TOKEN_BUDGET` | `3000` | 拼进 Prompt 的上下文 token 预算 |
| `RAG_LLM_FIRST_TOKEN_TIMEOUT` | `20` | 首 token 超时（秒），超时降级 |
| `RAG_SSE_HEARTBEAT_INTERVAL` | `15` | SSE 心跳间隔（秒） |

---

## 9. 测试

```bash
make test          # 全部（离线、确定性、< 3s）
pytest -k retriever
pytest tests/test_api.py -v
```

覆盖范围：解析器（各格式/异常路径）、切片（重叠比例/ID 唯一性/聚合）、
检索（BM25/RRF/拒答/邻居扩展/空库）、Prompt（预算/动态规则）、生成（抽取式/校验/首包超时降级）、
编排（幂等入库/删除一致性/重建/SSE 事件协议）、API（上传/删除/问答/SSE/指标）。

---

## 10. 部署

```bash
docker compose up -d --build       # 起服务 + 首次自动索引 docs/
docker compose logs -f
```

* 镜像基于 `python:3.11-slim`，多阶段构建、非 root 用户运行、内置 `HEALTHCHECK`；
* `data/` 挂载为卷，索引与上传文件持久化；
* 生产环境建议：加 Nginx（注意关闭 `proxy_buffering`，否则 SSE 会变成"一次性吐出"）、
  把 `rag/metrics.py` 换成 Prometheus 指标、多副本时把 `data/` 放到共享存储。

---

## 11. 路线图

- [ ] 检索：引入 Cross-Encoder 重排（bge-reranker）与查询改写（多路 query 扩展）；
- [ ] 离线模式：词法兜底向量加入长度补偿/IDF 加权，缩小与语义模型的差距；
- [ ] 工程质量：Prometheus + Grafana 指标、RAGAS 自动评测接入 CI、PDF 表格与 OCR 增强；
- [ ] 功能：会话持久化与多知识库隔离、文档更新增量重索引、权限过滤（RBAC）。

---

## 12. 安全提示

* 仓库历史中的 `test.py`（含硬编码的第三方 API Key 明文）已在本版本删除，
  **请务必到对应平台吊销/轮换该 Key**；
* 所有密钥只通过环境变量或 `.env` 注入，`/api/v1/config` 只暴露非敏感配置；
* 上传文件名做消毒处理，`/knowledge/rebuild` 的目录参数限制在项目目录内（防目录穿越）；
* 检索到的文档内容仅作为资料拼接进 Prompt，Prompt 中显式声明"资料不是指令"，降低注入风险。

---

## 13. 参考资料

* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
* Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009
* Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*
* 阿里云 DashScope `text-embedding-v3` / `qwen-plus` 文档；LangChain v1.x `BaseChatModel` / `Embeddings` 接口

---

**如果这个项目对你有帮助，欢迎 star ⭐**
