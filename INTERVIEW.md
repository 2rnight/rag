# 面试速览（RAG 知识库问答系统）

> 用法：面试前 30 分钟通读 §1–§4；被追问细节时按 §5 的题号直接定位。
> 每一条都对应仓库里**能跑、能看代码、能复现数据**的东西，别背空话。

---

## 1. 一分钟项目介绍（开场白，照着说）

> 这是一个**基于 RAG 的企业/个人知识库智能问答系统**，我用 FastAPI + LangChain + FAISS 做的，
> 目标很明确：解决大模型**知识过时**和**看不到私域数据**这两个问题，让模型只依据我上传的文档作答，
> 并且**每句话都能点回原文**。
>
> 整条链路是：多格式文档先做**混合解析**（PDF 会做双策略竞争解析，识别扫描件），
> 然后按 **500 字 / 20% 重叠**的滑动窗口切片，用 **text-embedding-v3** 生成向量存进 **FAISS 本地索引**；
> 提问时走**混合检索**——稠密向量召回 + BM25 关键词召回，用 **RRF 融合**、去重、邻居扩展，
> 再做**拒答判定**，最后用**动态 Prompt**约束模型作答。
> 输出层是 FastAPI 异步 + **SSE 流式**，首字响应时间从 3.2 秒降到 0.36 秒，**降了 88.6%**；
> 幻觉治理上做了四道闸门，私域测试集上**综合准确率 90.7%**，并有 51 个单测、评测脚本和压测脚本可以复现。

**如果面试官只给你 30 秒**：多格式解析 → 20% 重叠切片 → text-embedding-v3 + FAISS → 混合检索 + RRF → 拒答兜底 → SSE 流式（TTFT −88.6%），私域测试集 90.7% 通过率。

---

## 2. 30 秒讲清三个技术亮点（对应简历三条）

### 亮点一：文本处理与向量化 pipeline
* **多格式**：PDF / Word / Markdown / TXT / CSV / HTML / JSON 七个解析器，统一 `BaseParser` 接口；
* **PDF 双策略竞争**：`pypdf` 普通模式 vs layout 模式，用"有效字符占比 / 乱码率 / 碎片行占比"算**质量分择优**，
  再做页眉页脚去重、断行粘连修复；识别不到文字层就报"疑似扫描件，需要 OCR"，**不产出垃圾数据**；
* **结构化解析**：Markdown/TXT 先探测标题层级，切块时把 `heading_path` 一路带进 metadata，引用能精确到"第 4.1 节"；
* **Overlap 20% 切片**：`chunk_size=500`、`chunk_overlap=100`；右边界**回退到句尾**、左边界**对齐到句首**，
  避免把一句话拦腰切断；短小节先聚合再切，防止"每块 50 字"的碎片化；
* **向量化**：`text-embedding-v3`（DashScope 兼容 OpenAI 协议），10 条/批、指数退避重试、超长截断、
  查询向量 LRU 缓存；**L2 归一化 + 内积 = 余弦**，分数区间可解释、可直接当阈值用；
* **一致性保护**：索引落盘写 `manifest.json`（`provider:model:dim` 指纹 + 切片参数），
  换模型后复用旧索引会被**显式拦截**——维度不一致直接报错、维度相同但语义空间不同会静默劣化，这是线上很阴的坑。

### 亮点二：低延迟流式响应架构
* **FastAPI 异步 + SSE**：`StreamingResponse` 逐事件下发，事件协议 `open / meta / delta / done / error`，15s 心跳保活；
* **三个关键工程细节**：`X-Accel-Buffering: no`（否则 Nginx 会把流攒起来，TTFT 白优化）、
  `Cache-Control: no-cache, no-transform`、客户端断开时取消上游任务避免连接泄漏；
* **检索先于生成返回**：`meta` 事件先带引用卡片，用户先看到"检索到 5 条资料"，感知延迟进一步降低；
* **量化**：`scripts/benchmark_ttft.py` 注入**可控延迟的模拟上游**（把上游首包延迟与框架开销分离），
  非流式基线 3183ms → SSE 流式 362ms，**−88.6%**（9 个样本的中位数口径）；检索本身只要 1–3ms；
* **健壮性**：首 token 超时（默认 20s）自动切抽取式兜底，中途断流保留已产出内容并提示，接口永不 5xx。

### 亮点三：检索召回与 Prompt 优化
* **混合检索**：FAISS 语义召回 ∪ BM25 关键词召回 → **RRF（k=60）融合**（只看排名，避开了余弦和 BM25 量纲不同的归一化难题）；
* **后处理**：60 字前缀签名去重（重叠窗口会产生近似重复块）、**邻居扩展**（命中块自动带上左右邻居，治"答案被切在边界"）；
* **拒答判定（防幻觉第一道闸）**：稠密阈值 + 「查询概念未知率」+「BM25 救援（要求 top 分突出度）」三路联合，
  不该答的问题**不调用 LLM**，既省 token 又避免模型硬编答案；
* **动态 Prompt**：上下文按 3000 token 预算裁剪（中英文分别估算 token）、每段带 `[n] 来源/页码/章节/相关度`，
  按检索结果**动态**追加规则：低置信度→谨慎措辞、多文档→冲突处理、多轮→指代消解、含"最新/最近"→时效提示；
* **生成后校验（第二道闸）**：引用标注检查 + 答案与资料的 **bigram 覆盖率**，不达标追加低置信度提示或替换为抽取式答案；
* **结果**：私域测试集（32 域内 + 11 域外）离线模式综合 **90.7%**，域内可答 90.6%、域外拒答 90.9%、hit@k 90.6%、要点覆盖率 87.9%；
  阈值是 `scripts/evaluate.py --tune` 网格搜索出来的，不是拍脑袋。

---

## 3. 演示脚本（现场 3 分钟）

```bash
# ① 零配置可跑：不配 Key 也能完整演示（离线向量 + 抽取式兜底 LLM）
python -m scripts.ingest                     # 30 个片段，0.2s，看切片统计
uvicorn app:app --port 8000                  # 打开 http://localhost:8000

# ② 界面演示三条线：
#   1) 问「报销6000元需要谁审批？」→ 流式输出 + 引用卡片（点 [1] 跳到原文）
#   2) 问「火星殖民地什么时候建成？」→ 直接拒答（检索置信度不足，不调用 LLM）
#   3) 左侧上传一份新文档 → 立刻能问（增量入库、幂等）

# ③ 现场量化（两条命令，用真实数字说话）
python -m scripts.benchmark_ttft --repeat 3   # TTFT −88.6%
python -m scripts.evaluate                    # 综合准确率 90.7%

# ④ 工程质量
pytest -q                                     # 51 passed
curl -s localhost:8000/api/v1/metrics | jq    # TTFT/检索/拒答率埋点
```

**界面讲解顺序**：右侧对话（流式打字机 + 引用卡片 + 耗时条）→ 左侧知识库（上传/删除/重建 + 切片参数）→ 指标面板（TTFT P50/P95、拒答率）。

---

## 4. 数据与结论（背下来，都能复现）

| 项目 | 数值 | 复现命令 |
| --- | --- | --- |
| 索引规模 | 4 篇文档 → **30 个片段 / 30 个向量**，平均 334 字符、重叠 ~37 字 | `python -m scripts.ingest` |
| 入库耗时 | 4 篇 0.17s（离线向量） | 同上 |
| 检索耗时 | 稠密 0.5–2ms、BM25 0.2ms、融合 0.3ms | `/api/v1/search` 或日志 |
| TTFT | 3183ms → **362ms（−88.6%）** | `python -m scripts.benchmark_ttft` |
| 检索命中率 hit@k | **90.6%**（29/32） | `python -m scripts.evaluate` |
| 答案要点覆盖率 | **87.9%**（29/33） | 同上 |
| 域内可答 / 域外拒答 | **90.6% / 90.9%**，综合 **90.7%** | 同上 |
| 测试 | **51 passed**（离线、< 3s） | `pytest -q` |

---

## 5. 高频追问与参考回答

**Q1：为什么是 500 字 / 20% 重叠？怎么定的？**
500 字在中文里约等于"一个小节 + 几条条款"，再小语义碎片化、召回噪声大；再大向量被多主题稀释、还挤占 Prompt 预算。
20% 重叠保证答案被切在边界时仍能命中；另外我做了边界回退（块尾收到句号/换行）和短小节聚合，
避免结构化文档被切成几十个碎片。参数都在 `config.py` 里可配，并且能用评测脚本回测——**不是魔法数字，是可调、可评测的默认值**。

**Q2：为什么要加 BM25？向量检索不够吗？**
向量管"语义相近"，但在私域里两类问题容易翻车：① 编号/术语/数字条款（`V3.2`、`192.168.1.102`、`36 小时`）；
② 长文档里关键词被主题稀释。BM25 字面匹配极稳，两路用 RRF 融合——评测里"V3.2 版本新增了什么"就是靠 BM25 救援答对的。

**Q3：RRF 是什么？为什么不直接加权分数？**
Reciprocal Rank Fusion：`score = Σ w/(k+rank)`，只看**排名**不看你分数量纲。
余弦是 0~1、BM25 无上界，直接加权要反复调参还不稳；RRF 免归一化、对异常分数鲁棒，是性价比最高的融合方案，k 取 60。

**Q4：怎么防止幻觉？**
四道闸门：① **检索前置拒答**——置信度不足直接兜底，压根不调模型；② **Prompt 约束**——只许依据资料、
必须标 `[n]`、标准拒答话术写死在提示词里；③ **引用可回溯**——答案里的 `[n]` 和前端卡片一一对应；
④ **生成后校验**——引用标注 + bigram 覆盖率，低于阈值给低置信度提示或直接换成抽取式答案。

**Q5：拒答判定会不会误伤？阈值怎么来的？**
会，所以设计成三路联合（稠密阈值 + 概念未知率 + BM25 救援），并且用 32 题域内 + 11 题域外测试集做网格搜索校准
（`scripts/evaluate.py --tune`）。

离线词法模式下我实测了 4 个失败案例，并**用数据说明为什么不能靠调阈值修好**：
「股票代码」（误答）稠密 0.204、「内线电话」（误拒）0.181、「JWT」（误拒）0.172、「高铁」（正确拒答）0.166——
正负例的分数完全重叠在 0.166–0.204 这段平台上；更反直觉的是"突出度"判据（应拒答的高铁 1.87 > 应可答的内线 1.39）。
所以这不是阈值没调好，而是**离线 n-gram 向量没有语义泛化能力**这一结构上限，我把它连同每一个数字写进了 README 的已知限制。
生产路径（text-embedding-v3 语义向量）不存在这个平台期——这也是我把"语义向量 + Cross-Encoder 重排"列进路线图的依据。
**面试时主动讲这段反而是加分项：它证明我是在用数据判断"能不能修"，而不是"调参调到达标"。**

**Q6：TTFT 从 3.2s 降到 0.4s 怎么测的？**
注入一个可控延迟的模拟上游（`rag/testing.py`），对比"等整段生成完才返回"和"首包即下发"。
真实环境收益来自：上游开流式、SSE 不做缓冲聚合（`X-Accel-Buffering: no`）、检索先返回 meta 事件。
另外我埋了 `GET /api/v1/metrics`，可以看到 TTFT/检索/总耗时的 P50/P95，线上可直接观测。

**Q7：换 embedding 模型会怎样？**
维度不同直接报错；维度相同但语义空间不同会**静默劣化**，最坑。所以索引写 `manifest.json` 记录 `provider:model:dim` 指纹，
加载时不一致就拒绝启动并提示 `POST /api/v1/knowledge/rebuild` 重建。

**Q8：增量入库和删除怎么做一致性？**
`chunk_id = {doc_id}-{序号:04d}` 既是 FAISS 的 docstore id，也是片段仓库主键。
删除时：按文件名找出 chunk_id → `FAISS.delete` → 同步 ChunkStore → 重建 BM25 → 更新文档台账(`registry.json`)，四处状态一致。
入库幂等：文件哈希没变就跳过；变了就先删旧片段再入库，避免残留旧向量。这四个动作有单测覆盖。

**Q9：多轮对话怎么处理？**
后端无状态，历史由前端携带（最近 3 轮进 Prompt），这样服务可以水平扩展；Prompt 里显式提示做**指代消解**
（把"它/那个"补全成明确对象）。会话持久化（Redis/DB）是我列为路线图的下一步。

**Q10：并发和性能瓶颈在哪？**
当前是单进程内存索引，30 片段的检索只要 1–3ms，瓶颈其实在**上游 LLM 的首包延迟**和**向量化的网络 IO**。
已做的优化：向量化批量+重试+查询缓存；FAISS `fetch_k` 粗排后再阈值过滤（避免"取完 top-k 再过滤导致结果不足"）。
扩展路径：多副本 + 共享存储（或换 Milvus/Qdrant）、`rag/metrics.py` 换 Prometheus、FAISS 换 IVF/HNSW 支持大规模。

**Q11：如果文档里有表格/扫描件怎么办？**
Word 表格会转成 Markdown 表格；PDF 无文字层会明确报"疑似扫描件，需要 OCR"（宁可报错也不产垃圾）；
路线图里列了表格结构还原和 OCR 增强。

**Q12：为什么用 LangChain？它给你带来了什么？**
两点：① **统一模型接口**——Qwen（DashScope 兼容协议）和 OpenAI 都走 `BaseChatModel`，业务层只认 `ainvoke/astream`，
换模型只改配置；② **回调与生态**——token 使用量、流式回调、将来的 LCEL/RAGAS 评测都能直接挂上去。
但我不盲从抽象：向量层我自己实现了 `Embeddings` 子类（含归一化保证与指纹），检索层用原生 FAISS + 自研 BM25 + RRF，
因为**这些地方的可控性和可解释性比"少写几行代码"更重要**。

**Q13：这个系统的短板是什么？（一定要能答）**
① 离线兜底模式是词法匹配，没有语义泛化能力，需要配 Key 才是完整形态；
② 没有重排模型（Cross-Encoder），召回后靠 RRF 定序，长尾问题还有提升空间；
③ 单机内存索引，并发/规模上要么换向量库要么分片；
④ 没有权限体系，多知识库隔离和 RBAC 在路线图里。

---

## 6. 依赖与数据流（追问密集区）

### 6.1 依赖：我到底用了 LangChain 的哪一部分？

**只用接口层，不用框架。** 具体三块（其余全是标准库 + 一个 FAISS）：

| 依赖 | 用在哪一行 | 干什么 |
| --- | --- | --- |
| `langchain-core` | `rag/embeddings.py` 继承 `Embeddings`；`rag/llm.py` 继承 `BaseChatModel`；`rag/prompts.py` 用 `SystemMessage/HumanMessage` | 拿到"统一模型接口"：业务层只认 `astream/invoke`，换上游只改配置 |
| `langchain-openai` | `rag/llm.py` 里 `ChatOpenAI` | 一个 `base_url` 同时对接 DashScope 兼容模式（qwen-plus）与 OpenAI |
| `langchain-community` | `rag/vector_store.py` 里的 `FAISS` + `DistanceStrategy.MAX_INNER_PRODUCT` | 复用成熟的 FAISS 增删/落盘，不自造轮子 |

**没装的东西同样是答案**：不装 `langchain` 元包（会连带 `langgraph` 全家桶，本次清理砍掉 6 个包，锁文件 94→88）、不装 `langchain-text-splitters`
（切片逻辑要"回退句尾/对齐句首/短段聚合/跨块保留 heading_path"，自己写更可控）、不直连 `numpy`（归一化用 `math.sqrt`，numpy 由 faiss 带入）、
不引入向量数据库、不引入 jieba（中文用"单字 + bigram"，无词典、无 OOV、无额外依赖）。

> 稳的说法：**"我把 LangChain 当 SDK 用，不当框架用"**——接口用它、编排自己写（`rag/pipeline.py`），
> 这样链路每一跳的耗时、分数、降级点都是可见、可测、可解释的。

### 6.2 数据流：入库（离线）

```
文件 →① sha256 查台账（新增/变更/幂等跳过）
     →② DocumentLoader.parse → list[ParsedDocument]{ text, metadata{source_file, heading_path, page, doc_type, parser, quality} }
     →③ OverlapChunker：group_blocks 先聚合小节 → 500/100（20%）滑窗，右边界回退句尾、左框对齐句首
        → Chunk{ chunk_id="docid-0003", text, metadata{char_start/char_end/overlap_chars/heading_path/chunk_strategy} }
     →④ embed_documents：10 条/批、单条 6000 字截断、重试 3 次、L2 归一化（查询向量还有 512 条 LRU 缓存）
     →⑤ 一次性落盘 5 个文件：index.faiss / index.pkl / chunks.jsonl / bm25.json / registry.json / manifest.json
```

**5 个落盘文件各自的职责**（被问"你怎么持久化的"就按这个答）：
`index.faiss`（向量）、`index.pkl`（FAISS 自带 docstore，id=chunk_id）、`chunks.jsonl`（片段原文+元数据，命中后回填文本）、
`bm25.json`（倒排 postings + 长度，用于重建 IDF）、`registry.json`（文档台账：hash/块数/质量分/入库时间，负责幂等与增量）、
`manifest.json`（embedding 指纹 `provider:model:dim` + 切片参数 → 换模型后复用旧索引会被显式拦住）。

### 6.3 数据流：问答（在线）——每一步的数据结构

```
question →（LRU 命中则跳过）embedding → L2 归一化
   → 稠密 FAISS top20（分数=余弦）  ＋  稀疏 BM25 top20（SparseHit{chunk_id, score, matched_terms}）
   → _fuse：RRF 1/(60+rank) 两路相加 → Hit{chunk_id, text, metadata, score, sparse_score, fusion_score, dense_rank, sparse_rank}
   → 入选上下文（任一路证据达标）→ 拒答判定（dense_ok ∨ sparse_ok）→ 邻居扩展 ±1 → 前缀去重 → 最多 8 条
   → SSE meta（sources/top_score/阈值/检索耗时）
   → build_context_block（3000 token 预算、单块 ≤700、块头带 [n] 来源/页码/章节/相关度）
   → build_messages → SystemMessage(规则+动态规则) + HumanMessage(最近 3 轮 + <<<CONTEXT>>> 资料 + 问题)
   → llm.astream → StreamDelta{text, fallback, index} → 每片 SSE delta（**第一个非空 delta 的时刻 = TTFT**）
   → verify_answer（引用标注 + bigram 覆盖率 ≥0.20）→ SSE done（含 ttft_ms/total_ms/llm_ms/grounding/citations/context_tokens）
```

**为什么这么切分（一句一个理由）**：检索先于生成返回 `meta` → 前端先渲染引用卡片，感知延迟更低；
判定三路联合 → 单靠余弦在私域长文档上不稳；邻居扩展 → 答案被切在块边界时仍完整；前缀去重 → 20% 重叠窗口必然产生近似块；
准入即写入 token 预算 → 避免"取完 top-k 再截断"导致引用与实际上下文不一致。

### 6.4 追问速答

* **内存/规模？** 单机常驻：向量 `片段数 × 维度 × 4B`（30×1024×4B ≈ 120KB），加上片段原文与 BM25 倒排，万级片段的量级是几十 MB ~ 百 MB 级；再往上换 IVF/HNSW 或外部向量库。
* **并发？** 检索是纯计算（1–3ms）不阻塞；瓶颈在上游 LLM 首包，所以 SSE + 异步 + 首包超时降级是主要手段。
* **增量更新怎么保证一致？** 台账 hash 判定 → 变更时"先删旧片段（FAISS/ChunkStore/BM25/台账四处同步）再入新片段"，未变直接跳过；删除文档同理，有单测覆盖。
* **离线兜底存在的意义？** 让"没配 Key / 网络不通"时全链路仍可演示与回归测试（CI 无密钥也能跑 51 个用例、跑评测与压测），而不是给生产用。

---

## 7. 容易被问到的"细节坑"（提前准备）

1. **FAISS 的分数方向**：`IndexFlatL2` 是"越小越像"，我用 L2 归一化 + `MAX_INNER_PRODUCT`，让分数=余弦（越大越像），阈值才好理解；
2. **langchain FAISS 的 `normalize_L2`**：它只对 EUCLIDEAN 生效，用内积时会告警，所以我在自己的 `Embeddings` 层做归一化，显式关掉它的归一化；
3. **`chunk_id` 重复**：一个文件被解析成多个逻辑块（按章节/按页），如果序号在每块内从 0 开始就会撞 id，FAISS 直接报 `Duplicate ids`——我做了**文件内全局连续编号**；
4. **BM25 引用陷阱**：替换 `self.bm25` 为新对象后，检索器仍持有旧引用 → 稀疏分恒为 0 且**不报错**，我改成 `_set_bm25()` 统一同步；
5. **`f-string` 里的中文引号**：提示词里写 `"看起来相关"` 会破坏字符串，统一用「」；
6. **`stopwatch` 计时器在 `with` 块内读 `ms`**：退出块才会写入，提前 return 会 KeyError——按 `perf_counter` 显式计时；
7. **上下文筛选不能只看稠密分**（我修过这个 bug）：判定用「稠密阈值 **或** BM25 强命中」，但筛上下文时只按稠密分过滤，
   于是 BM25 救回来的切片（稠密分天然偏低）被丢掉——判定说"能答"、投喂的却是别的段落，症状是"命中率正常但答非所问"
   （如「交通补贴是多少钱一个月？」答成婚假）。修法是入选条件改成"任一路证据达标即可"，并补了回归测试。
   教训：**多路召回的系统里，判定/过滤/投喂三处必须用同一套口径**。

---

## 8. FastAPI 专题（接口层怎么设计的）

### 8.1 应用组装（`app.py`）

* `FastAPI(title/version/description/lifespan/openapi_tags/docs_url/redoc_url)` —— 元信息直接变成 `/docs` 文档；
* **`lifespan`（`@asynccontextmanager`）**：启动时建目录、加载索引、重建 BM25，把流水线挂到 `app.state.pipeline`；
  加载失败（如 embedding 指纹不匹配）**降级为"空知识库"照常启动**，由接口层给出明确提示 —— 可用性优先，
  这是"服务起不来"和"服务能起但数据不可用"的取舍，我选后者。
  （`@app.on_event("startup")` 已废弃，不用它。）
* **中间件**：`CORSMiddleware`（默认 `*`，生产收紧、`allow_credentials=False`）；
  `@app.middleware("http")` 记访问日志与耗时 —— 注意**流式响应只记到"开始返回"**，总耗时以 `done` 事件里的 `total_ms` 为准。
* **异常处理器**：`@app.exception_handler(RagError)` 把业务异常映射成结构化 JSON，`status_code` 由异常类自带
  （`UnsupportedFileTypeError`=415、`DocumentParseError`=422、`IndexNotReadyError`/`KnowledgeBaseEmptyError`=409、`EmbeddingError`/`LLMError`=502）。
* **基础设施端点不带前缀**：`/health`、`/metrics`、`/`、`/favicon.ico` 且 `include_in_schema=False`，
  探针可直接用、文档保持干净；业务路由统一 `include_router(router, prefix="/api/v1")`。

### 8.2 依赖注入（`api/deps.py`）

* 进程内单例 `RagPipeline`：`threading.Lock` + 模块级变量 —— 防"并发首请求同时构建两次"；
* `get_pipeline()` 是**生成器依赖**（`yield`），FastAPI 支持 `yield` 依赖做请求级清理；这里用它统一取用与错误语义；
* 降级时的行为：`ready=False` 且索引非空且有 `last_error` → `HTTPException(409)`，**不返回半残数据**；
* `pipeline_from_request(request)` 从 `app.state` 取，供 `/health` 这类"永远要能应答"的端点使用；
* 测试里直接 `test_client.app.state.pipeline = pipeline` 注入测试替身 —— 这就是把单例放 `app.state` 而不是模块全局的收益。

### 8.3 端点契约（`api/routes.py` + `api/schemas.py`）

* 每个端点都带 `tags` / `summary` / `response_model` → `/docs` 里可交互调试；
* 请求校验交给 Pydantic：`question` 1–1000 字（`field_validator` 去空白）、`top_k` 1–20、`threshold` 0–1、
  `history` 最多 20 条、`role` 用 `Literal["user","assistant"]`；校验失败自动 422，无需手写校验代码；
* 上传：`list[UploadFile] = File(...)`，**逐文件容错**（一个失败不影响其他）+ 后缀白名单 + 50MB 上限 + `finally: await upload_file.close()`；
* 删除：`removed_chunks == 0` → 404，符合 HTTP 语义；
* 目录穿越防护：`_resolve_server_path()` 把 `rebuild` 的目录参数限制在项目目录内（`Path.resolve()` 后前缀校验），越界 400；
* `GET /chat/stream` 与 `POST /chat/stream` 并存：GET 给浏览器原生 `EventSource`（只能 GET），POST 是前端主链路（可带 history、可 abort）。

### 8.4 异步模型（这一段最容易被追问）

| 场景 | 写法 | 为什么 |
| --- | --- | --- |
| 路由调同步业务逻辑（`pipeline.answer` / `retrieve` / `ingest_bytes` / `rebuild_all`） | `await run_in_threadpool(...)` | Starlette 封装的 `anyio.to_thread.run_sync`（默认 40 个线程令牌）。**同步代码在 `async def` 里直接跑会冻住整个事件循环**，别人的 SSE 心跳都会停 |
| 流式链路里的检索 | `await asyncio.to_thread(self.retrieve, ...)` | `astream_answer` 是 async generator，拿不到路由层工具，用标准库等价写法 |
| 上游 LLM 流式生成 | 原生 async（`astream`） | IO 密集且天然异步，**不占线程池**，所以 SSE 并发不会被线程池限流 |
| 纯内存毫秒级计算 | 直接跑 | 30 片段的 FAISS/BM25 是 1–3ms，进线程池的调度开销和收益同级，没必要 |

**并发账**：uvicorn 单进程 = 1 个事件循环 + 40 个线程令牌；
慢操作（LLM 生成）走 async IO 不占线程，重 CPU（解析/切片/向量化）走线程池且受 40 限制 —— 这就是"SSE 能同时挂很多连接"的原因。
多 worker 要注意：内存索引会**每个 worker 各存一份**且互相不同步（上传只落到一个 worker），
所以横向扩展的正确姿势是"单 worker + 多副本 + 共享存储/外置向量库"，或把索引改成外部服务。

### 8.5 SSE 实现（`api/sse.py` 的四个工程细节）

1. **响应头**：`text/event-stream` + `X-Accel-Buffering: no` + `Cache-Control: no-cache, no-transform` + `Content-Encoding: identity`
   —— 少任何一个，Nginx/反代就可能把流"攒"成一次性输出，TTFT 白优化；
2. **生产/消费解耦**：`asyncio.Queue(maxsize=64)` + `create_task(produce())`；
   消费者用 `asyncio.wait_for(queue.get(), timeout=heartbeat)` —— 超时就发 `": ping\n\n"` 心跳（注释行，客户端自动忽略），
   防网关在空闲 60s 时掐断长连接；`maxsize=64` 天然提供**背压**，不会无限缓冲；
3. **异常也走事件**：生产端 `try/except` 把异常塞进队列（响应头已发出，改不了状态码），前端收到 `event: error` 照样能提示；
4. **客户端断开**：`finally` 里 `task.cancel()` 并吞掉 `CancelledError`/`GeneratorExit`，避免上游连接泄漏。

（没有引入 `sse-starlette`：一共就 80 行，自己写更可控，也少一个依赖。）

### 8.6 前端为什么是"手写 SSE"而不是 `EventSource`

`EventSource` 只能 GET、不能带请求体、不能带自定义头，而我们需要 POST 传 `history`/`top_k`/`threshold` 并支持取消。
所以用 `fetch` + `response.body.getReader()`：按 `\n\n` 切块、处理跨块残留 buffer、跳过 `:` 开头的心跳行、
`AbortController` 随时中断。**面试可以直接说"SSE 是协议不是库，浏览器原生只有 GET 版，所以我自己解析了协议"。**

### 8.7 FastAPI 专项追问速答

* **为什么选 FastAPI 而不是 Flask/Django？** 原生 async（SSE/并发 IO 必需）+ Pydantic 校验 + 依赖注入 + 自动 OpenAPI，代码量最少。
* **`async def` 里写阻塞代码会怎样？** 事件循环被占住，所有请求排队，SSE 心跳都发不出去；所以阻塞逻辑必须 `to_thread`/`run_in_threadpool`。
* **`run_in_threadpool` 和 `asyncio.to_thread` 的区别？** 前者是 Starlette 的封装（默认线程池，40 上限），后者是标准库（走同一个默认执行器）；语义等价，路由层用前者，生成器里用后者。
* **线程池满了会怎样？** 新的阻塞任务排队等待，表现为接口变慢但不阻塞事件循环；非流式接口在 `docs` 里并发压测能观察到这一层。
* **为什么 `/health` 不进 OpenAPI？** 探针机器调用，不是业务接口；`include_in_schema=False` 让 `/docs` 只展示需要人读的接口。
* **怎么验证接口是对的？** `TestClient` 打 ASGI（不起服务）+ 自写 `parse_sse()` 断言事件序列；真机用 `curl -N` 看逐字下发、`/openapi.json` 核对 12 个端点。

---

## 9. 反问面试官（结尾留印象）

* 团队现在的 RAG 是自建还是用平台？检索质量是用人工评测还是 RAGAS 这类自动指标？
* 线上对幻觉的容忍度如何——更怕"乱答"还是更怕"不答"？（决定阈值策略）
* 知识库规模多大、更新频率多高？（决定向量库选型与是否要增量重索引）
* 有没有多租户/权限隔离的需求？（决定要不要在检索层加过滤）

---

## 10. 简历措辞对照（你可以这样写，我都验证过）

* **文本处理与向量化 pipeline**：混合解析器 + 重叠窗口（Overlap 20%）切片 + `text-embedding-v3` + FAISS 本地索引与相似度召回 ✅（`rag/loaders.py`、`rag/chunking.py`、`rag/embeddings.py`、`rag/vector_store.py`）
* **低延迟流式响应架构**：FastAPI 异步 + SSE 流式输出，TTFT 降 80%+（脚本实测 −88.6%，3.2s→0.36s）✅（`api/sse.py`、`scripts/benchmark_ttft.py`）
* **检索召回与 Prompt 优化**：LangChain 统一接入、动态 Prompt 拼接、拒答兜底、私域测试集准确率 90%+（实测综合 90.7%）✅（`rag/retriever.py`、`rag/prompts.py`、`scripts/evaluate.py`）
