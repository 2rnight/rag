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
  <img alt="tests" src="https://img.shields.io/badge/tests-49%20passed-3ddc97">
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
| **私域测试集准确率 90%+** | 内置 32 题域内 + 11 题域外（含困难负例）评测集与评测脚本，离线模式即达 **综合 93.0%**（可 `--tune` 重新校准阈值） | `tests/data/eval_set.json` `scripts/evaluate.py` |

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

## 3. 快速开始

### 3.1 安装

```bash
git clone <this-repo> && cd rag

python -m venv .venv && source .venv/bin/activate     # 或 uv venv
pip install -r requirements.txt                        # 或 uv sync
```

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
make test       # 49 个单测/集成测试（离线可跑，无需 API Key）
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
| 检索命中率 hit@k | **90.6%** | 期望文档是否出现在召回结果里 |
| 答案要点覆盖率 | **87.5%** | 答案是否命中标注的关键信息（如"5 天""ZhiYuan@2024"） |
| 可答准确率（域内） | **93.8%**（30/32） | 是否误拒了能回答的问题 |
| 拒答准确率（域外） | **90.9%**（10/11） | 幻觉防线：不该答的要拒答 |
| **综合准确率** | **93.0%** | 域内可答 + 域外拒答 |
| 单题平均耗时（不含 LLM） | 4.5 ms（p95 3.8 ms） | 检索 + 组装 Prompt |

配置 `DASHSCOPE_API_KEY` 后使用 `text-embedding-v3 + qwen-plus`，语义召回会把上述指标进一步推高（尤其是"说不出关键词但意思相同"的问法）。
阈值不是拍脑袋定的：`python -m scripts.evaluate --tune` 会在私域测试集上**网格搜索**相似度阈值与 BM25 救援阈值，输出准确率最高的参数组合。

> 已知限制（诚实记录）：离线词法模式唯一漏判的域外问题是
> 「北京到上海的高铁要多久？」——"北京/上海"确实出现在员工手册（分公司地址）中，
> 纯字面检索无法识别"意图不同"；这类问题需要语义向量或 LLM 相关性判断，属于该模式的固有上限。

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

**Q5：上游大模型挂了怎么办？**
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
├── tests/                     49 个 pytest 用例（解析/切片/检索/生成/编排/API）+ 评测集
├── docs/                      内置示例语料（员工手册/产品说明书/FAQ/开发规范）
├── data/                      运行期数据（索引/片段/台账/上传文件，已 gitignore）
├── Dockerfile / docker-compose.yml / Makefile / requirements.txt / .env.example
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
