# DeepResearch Agent

这是一个 **Python-first Agent 项目**，目标不是普通后端，而是实现 DeepResearch 方向的 Agent 能力：任务规划、工具调用、上下文管理、来源评估、引用验证、反思和 Eval Harness。

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
- SourceConnector 工具抽象，统一封装 GitHub / arXiv / Web / MCP 和离线 fallback 来源
- 普通运行默认启用外部检索 Connector，失败时自动回退到离线来源
- ToolRegistry 统一调度多个 Connector，并按 URL 合并来源
- ContextManager 管理工作记忆和证据池
- PromptInjectionGuard 基础安全检查
- SourceEvaluator 按权威性、新鲜度、相关性和风险进行来源评分
- CitationVerifier 结合来源分数和关键词证据重合度，标记 supported / weak / unsupported，并保存 evidence excerpt 与 verification reason
- `--llm` 开启后，CitationVerifier 会调用 LLM 对 claim 与 evidence 做严格 supported / weak / unsupported 判断
- ConflictDetector 检查同一主题下 supported / weak / unsupported 信号是否冲突，并在报告中输出 Conflict Checks
- 轻量 RAG Retriever：把来源切成 evidence chunks，写入本地 `vector_store.json`，用哈希向量和 cosine similarity 检索支撑 claim
- ReportAuditor 对最终报告逐句做 citation audit，统计 grounded / weak / uncited 句子
- ReflectionAgent 查找信息缺口，LangGraph 版本可通过条件边回到 search 做补充检索
- ReportAgent 生成带 Evidence、Citation Checks、Remaining Gaps 的 Markdown 报告
- 可选 LLM Planner / Reporter：有 API key 时增强计划和摘要，无 key 时保持规则版可复现
- EvalHarness 评估任务完成、来源数、引用数、弱引用、引用支持率和来源多样性
- 每次 run 生成 `task_state.json`、`trace.jsonl`、`research_state.json`、`report.md`、`checkpoint.json`

## 后续路线

1. 增加更稳定的商业 Web Search Provider 或自建搜索代理。
2. 继续增强 LLM citation grounding，让逐句报告级引用校验由模型辅助判断。
3. 将本地哈希向量库替换为 FAISS、Chroma、Milvus 或 pgvector。
4. 做上下文压缩、证据 pinning、完整 GraphState checkpoint resume。
5. 接真实 MCP Client / MCP Gateway；当前已预留 `MCP_SEARCH_ENDPOINT` 搜索入口。
6. 加 Tool Policy 和更严格的 Prompt Injection 防护。
7. 扩展 Eval Harness，统计引用准确率、报告逐句 grounded rate、来源多样性、冲突检测和失败原因。
