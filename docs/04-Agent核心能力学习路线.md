# Agent 核心能力学习路线

这篇只讲你需要重点混熟的几块：

- Score 来源评分
- MCP 工具接入
- RAG 证据检索和调优
- Context 上下文控制
- 防幻觉与引用校验

它们不是五个孤立功能，而是一条链：

```text
问题
-> LangGraph 工作流
-> tools.py 搜索外部来源
-> security.py 过滤危险来源
-> rag.py 切 evidence chunk 并检索证据
-> agents.py 对来源评分、生成 claim、验证 citation
-> context_manager.py 控制当前 Agent 记忆
-> ReportAuditor 检查最终报告有没有没证据支撑的话
```

## 1. Score 是怎么创建的

代码位置：

- `deepresearch/langgraph_loop.py` 的 `_score`
- `deepresearch/agents.py` 的 `SourceEvaluatorAgent.score`

工作流里这一句负责调用评分器：

```python
research_state.scores = runtime.evaluator.score(research_state.sources, question)
```

真正打分逻辑在 `SourceEvaluatorAgent`：

```python
final = authority * 0.35 + freshness * 0.2 + relevance * 0.35 + (1 - risk) * 0.1
```

四个维度：

```text
authority  来源权威性：官方文档、论文、GitHub、普通网页分数不同
freshness  新鲜度：发布时间越新越高
relevance  相关度：问题关键词是否出现在 title/snippet/metadata 里
risk       风险：如果来源被安全模块标记，分数降低
```

面试说法：

> 我没有直接相信搜索结果，而是先把来源结构化成 Source，再用 Evaluator 从权威性、新鲜度、相关度和风险四个维度打分。后续 CitationVerifier 会结合这个分数判断 claim 是 supported、weak 还是 unsupported。

## 2. MCP 是怎么创建和接入的

代码位置：

- `deepresearch/tools.py` 的 `MCPSearchConnector`
- `deepresearch/tools.py` 的 `default_research_connectors`
- `deepresearch/tools.py` 的 `ToolRegistry`

MCP 在项目里不是单独替代 tools.py，而是 tools.py 里面的一个 Connector。

```text
ToolRegistry
-> MCPSearchConnector
-> MCP_SEARCH_ENDPOINT
-> 外部 MCP Search Gateway
```

当前接法：

```python
FallbackConnector(MCPSearchConnector(), OfficialDocsConnector())
```

意思是：

```text
先调 MCP
MCP 没配置或失败
就回退到官方文档来源
```

面试说法：

> MCP 的作用是把外部工具标准化。我的 workflow 不直接写死搜索 API，而是通过 ToolRegistry 调用 SourceConnector。MCP 只是其中一种 Connector，后续可以把搜索、GitHub、文件、数据库都挂到 MCP Gateway 后面，主流程不用改。

## 3. RAG 在去重后干什么

代码位置：

- `deepresearch/langgraph_loop.py` 的 `_dedupe`
- `deepresearch/rag.py` 的 `EvidenceRetriever`
- `deepresearch/vector_store.py` 的 `LocalVectorStore`

在 `_dedupe` 里：

```python
research_state.sources = runtime.dedupe_sources(state["collected"])
research_state.evidence_chunks = runtime.rag.index_sources(research_state.sources)
runtime.run_store.write_vector_store(task_state, runtime.rag.vector_store)
```

RAG 做三件事：

```text
1. 把 Source 里的 title、snippet、metadata 拼成 text
2. 把 text 变成 EvidenceChunk
3. 写入本地向量库，用于后面 verifier 找证据
```

当前是轻量版本：

```text
不是 Transformer embedding
不是真正向量数据库
是本地哈希向量 + cosine similarity
```

但边界是对的，后续可以替换成：

```text
OpenAI / bge / jina embedding
FAISS / Chroma / Milvus / pgvector
```

面试说法：

> RAG 不是在一开始回答问题，而是在来源去重后把 Source 切成 evidence chunk，并写入本地 vector store。CitationVerifier 生成 claim 时，会从 retriever 里取与问题最相关的证据片段，避免模型直接凭空生成结论。

## 4. Context 是怎么控制的

代码位置：

- `deepresearch/context_manager.py`
- `deepresearch/runtime.py` 的 `persist`
- `deepresearch/langgraph_loop.py` 每个节点里的 `context.add_*`

ContextManager 不是简单聊天历史，它分成：

```text
task_goal        当前研究目标
working_notes    工作记忆
evidence_pool    来源证据池
verified_claims  已验证 claim
discarded_notes  被压缩/丢弃的旧 note
```

它有一个简单压缩策略：

```python
if len(self.working_notes) > 20:
    self.discarded_notes.append(self.working_notes.pop(0))
```

构建上下文时只取：

```text
最近 8 条 working notes
前 8 个 important sources
```

面试说法：

> 我没有把所有历史都塞给模型，而是把上下文拆成目标、工作记忆、证据池和已验证 claim。ContextManager 会控制保留最近关键 notes 和重要 sources，后续可以扩展成 token budget、证据 pinning 和摘要压缩。

## 5. 怎么防止幻觉

代码位置：

- `deepresearch/security.py`
- `deepresearch/agents.py` 的 `CitationVerifierAgent`
- `deepresearch/agents.py` 的 `ConflictDetectorAgent`
- `deepresearch/agents.py` 的 `ReportAuditorAgent`
- `deepresearch/eval_harness.py`

项目里有五层防幻觉：

```text
第一层：PromptInjectionGuard
搜索结果里有恶意指令，先过滤或标记

第二层：SourceEvaluator
来源先打分，低质量来源不能直接变成强结论

第三层：RAG EvidenceRetriever
claim 必须绑定 evidence excerpt

第四层：CitationVerifier
每个 claim 标记 supported / weak / unsupported

第五层：ReportAuditor
最终报告逐句检查有没有没有证据支撑的句子
```

面试说法：

> 我防幻觉不是只靠 prompt，而是在工作流里做约束。搜索结果先过安全过滤，再对来源打分；RAG 提供 evidence chunk；CitationVerifier 要求 claim 绑定 evidence excerpt，并标记 supported、weak、unsupported；最后 ReportAuditor 对最终报告做逐句 grounding audit，找出 uncited sentence。

## 6. 你现在最该背的主流程

```text
1. cli.py 接收 question
2. runtime.py 装配 Planner、ToolRegistry、RAG、Evaluator、Verifier、Reporter
3. langgraph_loop.py 创建 GraphState
4. plan 节点拆子问题
5. search 节点通过 ToolRegistry 调 GitHub/arXiv/Web/MCP
6. security.py 过滤 prompt injection 风险
7. dedupe 节点去重，并让 RAG 建 evidence chunks
8. score 节点给 Source 打分
9. verify 节点生成 claim，并绑定 evidence excerpt
10. reflect 节点检查是否缺 RAG/MCP/context/safety/eval 等信息
11. 如果有 gap，通过条件边回到 search
12. report 节点生成 Markdown 报告
13. ReportAuditor 检查报告句子是否被证据支撑
14. run_store.py 保存 report、trace、checkpoint、research_state
```

一句总回答：

> 这个项目重点不是做普通后端，而是做一个可追踪的 DeepResearch Agent 工作流。它通过 LangGraph 管理 plan-search-score-verify-reflect-report 的状态流，通过 ToolRegistry 接外部检索工具，通过 RAG 建 evidence chunks，通过 ContextManager 控制工作记忆，通过 CitationVerifier 和 ReportAuditor 降低幻觉，并用 EvalHarness 量化当前效果。
