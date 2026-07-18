# DeepResearch Agent

面向技术调研场景的 Python-first DeepResearch Agent。它不直接根据模型记忆输出答案，而是把研究问题拆成子任务，检索外部资料，构建证据链并验证引用，最终生成可追溯的 Markdown 研究报告。

```text
用户问题
  -> Plan：拆解研究子问题
  -> Query Rewrite：为每个子问题生成多条检索 query
  -> Search：统一调用 GitHub / arXiv / Web / MCP Connector
  -> Dedupe：规范化 URL，合并重复来源并保留 provenance
  -> RAG：Source -> EvidenceChunk -> Hybrid Retrieval
  -> Score：权威性 / 新鲜度 / 相关性 / 风险评分
  -> Verify：claim 与 evidence 校验，标记 supported / weak / unsupported
  -> Reflect：有证据缺口则回到 Query Rewrite 补搜
  -> Report：输出带引用、冲突检查和报告审计结果
```

## 为什么不是普通 RAG 问答

- **动态工作流**：LangGraph 条件边控制补搜，而不是固定的一次检索后直接回答。
- **多源工具调用**：Search 通过 `ToolRegistry` 统一调度 GitHub、arXiv、Web、MCP 与离线 fallback，主流程不依赖某个搜索供应商。
- **证据可追溯**：每条来源记录检索 query、Connector 和规范化 URL；每次 run 都落盘 report、trace、task state、checkpoint 和 vector store。
- **受控运行**：ContextManager 按来源评分选择有限证据并控制 prompt 预算；ToolRegistry 对 Connector 调用做允许名单、输入长度/结果数限制和审计。
- **引用可信性控制**：来源评分、Prompt Injection 过滤、Hybrid RAG、CitationVerifier 和报告逐句审计共同约束模型生成。

## 快速体验

```bash
python3 -m pip install -r requirements.txt
python3 -m deepresearch --offline-tools "比较 LangGraph、CrewAI、AutoGen 的 Agent 工作流能力"
```

开启 `--live-tools` 可检索真实来源；配置 LLM API Key 后使用 `--llm` 启用模型辅助规划、评分、验证和报告生成。未配置 Key 时会回退到规则版，保证流程可复现。

需要在 RAG 前抓取并清洗网页正文时，加上 `--fetch-content`。该开关只处理 Web/official 的 HTTP(S) 来源，正文长度受限后写入 `Source.metadata["content"]`，再进入 EvidenceChunk 切分，避免只依赖搜索摘要。

可选 API 服务：

```bash
uvicorn deepresearch.api:create_app --factory --reload
```

`POST /research` 创建研究任务，`GET /research/{task_id}/events` 通过 SSE 推送 plan/search/verify/report 等事件，`GET /research/{task_id}` 获取已完成报告。

早期 Java Demo 已移动到 `_legacy/java_mvp/`，只作为历史参考，不再是主线。

## 学习入口

如果你是从零开始看，先不要直接读代码。按这个顺序看：

1. [00-全局总览](docs/00-全局总览.md)
2. [01-怎么学](docs/01-怎么学.md)
3. [02-运行时主流程](docs/02-运行时主流程.md)
4. [03-工具接入与调用设计](docs/03-工具接入与调用设计.md)
5. [04-上下文管理设计](docs/04-上下文管理设计.md)
6. [05-来源评分与引用校验设计](docs/05-来源评分与引用校验设计.md)
7. [06-安全与 Prompt 注入防护](docs/06-安全与Prompt注入防护.md)
8. [07-运行工件与状态落盘](docs/07-运行工件与状态落盘.md)
9. [08-Eval 评估框架设计](docs/08-Eval评估框架设计.md)
10. [90-面试话术](docs/90-面试话术.md)

每看一个模块，只回答三件事：

- 这个模块解决什么问题
- 它是怎么做的
- 为什么要这样设计

## 快速运行

```bash
make run
```

等价于：

```bash
python3 -m deepresearch
```

不要直接运行 `deepresearch/__init__.py`。它只是 Python 包初始化文件，不是 Agent 主入口。

使用 LangGraph 引擎：

```bash
python3 -m pip install -r requirements.txt
python3 -m deepresearch --engine langgraph
```

当前默认使用 LangGraph 引擎，会把 DeepResearch 流程映射成 LangGraph 的 `StateGraph` 节点。旧的手写 loop 已删除，当前只保留 LangGraph 主线，避免学习时出现两套流程。

运行评估 Harness：

```bash
make eval
```

普通运行默认启用真实 GitHub / arXiv / Web / MCP 检索，并在失败时自动回退：

```bash
make run
```

或者：

```bash
python3 -m deepresearch --engine langgraph "比较 LangGraph、CrewAI、AutoGen 的 Agent 工作流能力"
```

默认 live tools 会启用真实 HTTP Connector：

- `GitHubSearchConnector`：调用 GitHub repository search API，可选读取 `GITHUB_TOKEN`
- `ArxivSearchConnector`：调用 arXiv Atom API
- `MCPSearchConnector`：读取 `MCP_SEARCH_ENDPOINT` 调 MCP 搜索网关
- `WebSearchConnector`：尝试 DuckDuckGo HTML 搜索，无需 API key
- 如果网络失败或 API 不可用，会自动回退到离线 Connector，保证学习演示不断掉

只想跑离线固定资料时：

```bash
make run-offline
```

开启真实 LLM：

```bash
make run-llm
```

或者同时启用真实检索和真实 LLM：

```bash
make run-full
```

LLM Provider 使用 OpenAI-compatible 接口，不配置 key 时会自动回退到规则版本：

- `DEEPSEEK_API_KEY`：优先使用 DeepSeek，默认模型 `deepseek-v4-flash`
- `OPENAI_API_KEY`：其次使用 OpenAI-compatible endpoint，默认模型 `gpt-4o-mini`
- 可选覆盖：`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`OPENAI_BASE_URL`、`OPENAI_MODEL`
- 稳定搜索可选配置：`BRAVE_SEARCH_API_KEY`、`TAVILY_API_KEY`
- 真实 embedding 向量检索可选配置：`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_DIMENSIONS`

从 checkpoint 恢复：

```bash
python3 -m deepresearch --engine langgraph --resume run_20260714-043610-73dfab
```

或者：

```bash
make resume RUN_ID=run_20260714-043610-73dfab
```

当前 resume 是 MVP 版本：读取旧 run 的 `checkpoint.json` 和 `task_state.json`，用原始问题和上一阶段信息启动一个新的继续运行。生产级版本后续会恢复完整 GraphState。

语法检查：

```bash
make check
```

## 项目结构

```text
deepresearch/
├── cli.py             # 命令行入口，类似 Pico 的 cli.py
├── runtime.py         # 运行时装配，类似 Pico 的 runtime.py
├── langgraph_loop.py  # LangGraph 节点和条件边工作流
├── task_state.py      # 单次 run 的状态快照
├── run_store.py       # task_state / trace / report / checkpoint 落盘
├── checkpoint.py      # checkpoint 生成与下一步推断
├── context_manager.py # 上下文管理、工作记忆、证据池
├── tools.py           # SourceConnector 与 ToolRegistry
├── agents.py          # Planner / Evaluator / Verifier / Reflection / Reporter
├── security.py        # Prompt Injection Guard
├── eval_harness.py    # Agent 评估 Harness
├── models.py          # 领域数据结构
├── rag.py             # EvidenceRetriever 与证据检索
├── vector_store.py    # 本地向量库
└── llm_provider.py    # DeepSeek / OpenAI-compatible LLM 接入

run_agent.py           # 本地运行入口
_legacy/java_mvp/      # 早期 Java 原型，仅归档参考
docs/                  # 学习文档和面试话术
```

## 当前能力

- Python Agent Core，结构参考 Pico 的 runtime / loop / store / state 分层
- 自动拆解研究问题
- LangGraph 将 `query_rewrite` 作为独立节点；SearchQueryPlanner 对每个子问题或反思缺口做 query fan-out，覆盖官方文档、GitHub/论文和趋势检索
- SourceConnector 工具抽象，统一封装 GitHub / arXiv / Web / MCP 和离线 fallback 来源
- 普通运行默认启用外部检索 Connector，失败时自动回退到离线来源
- ToolRegistry 统一调度多个 Connector，记录 retrieved_query/retrieved_by/connectors，并按规范化 URL 合并来源
- ContextManager 管理工作记忆和证据池；评分后优先保留高分证据，在 6000 字符预算内把上下文提供给 Reflect 和 Report 的 LLM prompt
- PromptInjectionGuard 扫描 title/snippet/metadata，按 high/medium 风险过滤或打标
- SourceEvaluator 按权威性、新鲜度、相关性和风险进行来源评分；相关性采用 LLM/关键词 + RAG 向量相似度的混合判断
- CitationVerifier 结合来源分数、EvidenceChunk 和可选 LLM judge，标记 supported / weak / unsupported，并保存 evidence excerpt 与 verification reason
- `--llm` 开启后，CitationVerifier 会调用 LLM 对 claim 与 evidence 做严格 supported / weak / unsupported 判断
- ConflictDetector 检查同一主题下 supported / weak / unsupported 信号是否冲突，并在报告中输出 Conflict Checks
- 轻量 Hybrid RAG Retriever：把来源切成 evidence chunks，写入本地 `vector_store.json`，用 embedding/cosine similarity + 关键词重合检索支撑 claim
- ReportAuditor 对最终报告逐句做 citation audit；`--llm` 开启后可用 LLM 做 grounded / weak / uncited 审计
- ReflectionAgent 查找信息缺口，LangGraph 可通过条件边回到 `query_rewrite`，补搜前先重新规划 query
- ReportAgent 生成带 Evidence、Citation Checks、Remaining Gaps 的 Markdown 报告，并同步写出 `research_report.v1` 结构化 JSON
- 可选 LLM Planner / Evaluator / Verifier / Reflection / Reporter / Auditor：有 API key 时增强语义判断，无 key 时保持规则版可复现
- EvalHarness 评估任务完成、来源数、引用数、弱引用、引用支持率和来源多样性
- ToolRegistry 为每次 Connector 调用记录成功/失败、结果数、耗时和原因，并写入 `research_state.json` 与 `report.json`
- `--fetch-content` 可抓取并清洗 Web/official 正文，正文进入 EvidenceChunk 后参与 Hybrid RAG，而不只使用搜索摘要
- 提供可选 FastAPI + SSE 接口，支持创建研究任务、实时订阅执行事件和查询最终报告
- 每个关键节点原子刷新 `task_state.json`、`trace.jsonl`、`research_state.json`、`checkpoint.json`；当前 resume 使用原问题 replay，不伪称完整 GraphState 原地续跑

## 后续路线

1. 增加更稳定的商业 Web Search Provider 或自建搜索代理。
2. 继续增强 LLM citation grounding，让逐句报告级引用校验由模型辅助判断。
3. 将本地轻量向量库替换为 FAISS、Chroma、Milvus 或 pgvector。
4. 完整 GraphState checkpoint resume，实现中断节点原地续跑。
5. 接真实 MCP Client / MCP Gateway；当前已预留 `MCP_SEARCH_ENDPOINT` 搜索入口。
6. 增加域名策略和高风险工具人工确认。
7. 扩展 Eval Harness，统计引用准确率、报告逐句 grounded rate、来源多样性、冲突检测和失败原因。
