"""Runtime facade for DeepResearch Agent.

这个文件对齐 Pico 的 `runtime.py`：负责把 Planner、工具、上下文、安全、
RunStore 和 LangGraph 工作流装配成一个可以执行 `ask()` 的运行时对象。
"""

import urllib.parse

from .agents import ConflictDetectorAgent, CitationVerifierAgent, PlannerAgent, ReflectionAgent, ReportAgent, ReportAuditorAgent, SearchQueryPlannerAgent, SourceEvaluatorAgent
from .checkpoint import create_checkpoint
from .content_fetcher import build_content_fetcher
from .graph_rag import EvidenceGraphBuilder
from .harness import AgentHarness, RunPolicy
from .models import ResearchEvent, Source, now_iso
from .llm_provider import build_llm_provider
from .rag import EvidenceRetriever
from .run_store import RunStore
from .security import PromptInjectionGuard
from .task_state import TaskState
from .tools import ToolRegistry, default_research_connectors
from .vector_store import build_vector_store
from typing import Optional


class DeepResearchRuntime:
    """DeepResearch Agent 的运行时总装配器。

    Runtime 负责把 Planner、工具、RAG、评分器、验证器、报告器、存储器和
    LangGraph 工作流组装到一起。
    """

    def __init__(
        self,
        run_store: Optional[RunStore] = None,
        tools: Optional[ToolRegistry] = None,
        engine: str = "langgraph",
        use_live_tools: bool = False,
        use_llm: bool = False,
        fetch_content: bool = False,
        run_policy: Optional[RunPolicy] = None,
    ) -> None:
        """根据配置创建一次可执行的 Agent runtime。"""
        self.run_store = run_store or RunStore()
        self.engine = engine
        self.use_live_tools = use_live_tools
        self.use_llm = use_llm
        self.content_fetcher = build_content_fetcher(fetch_content)
        self.llm = build_llm_provider(use_llm)
        self.planner = PlannerAgent(self.llm)
        self.query_planner = SearchQueryPlannerAgent(self.llm)
        self.tools = tools or ToolRegistry(default_research_connectors(use_live=use_live_tools))
        self.harness = AgentHarness(run_policy)
        self.guard = PromptInjectionGuard()
        self.rag = EvidenceRetriever(build_vector_store())
        self.graph_builder = EvidenceGraphBuilder()
        self.evaluator = SourceEvaluatorAgent(self.llm)
        self.verifier = CitationVerifierAgent(self.llm)
        self.conflict_detector = ConflictDetectorAgent()
        self.reflector = ReflectionAgent(self.llm)
        self.reporter = ReportAgent(self.llm)
        self.report_auditor = ReportAuditorAgent(self.llm)
        self.loop = self._build_loop(engine)
        self.current_task_state = None
        self.current_research_state = None

    def _build_loop(self, engine: str):
        """根据 engine 名称创建工作流实现。当前只保留 LangGraph。"""
        if engine == "langgraph":
            from .langgraph_loop import LangGraphAgentLoop

            return LangGraphAgentLoop(self)
        raise ValueError("unknown engine: {}".format(engine))

    def ask(self, question: str, emit=lambda event: None):
        """执行一次新的研究任务。"""
        return self.loop.run(question, emit=emit)

    def resume(self, run_id: str, emit=lambda event: None):
        """根据 checkpoint 恢复某个 run。当前是轻量恢复，会重新推进研究。"""
        checkpoint = self.run_store.read_checkpoint(run_id)
        task_state = self.run_store.read_task_state(run_id)
        question = checkpoint.get("current_goal") or task_state.get("user_request")
        if not question:
            raise ValueError("cannot resume {}; checkpoint or task_state is missing".format(run_id))
        return self.ask(question, emit=emit)

    def create_task_state(self, question: str) -> TaskState:
        """为用户问题创建 TaskState。"""
        return TaskState.create(user_request=question)

    def emit(self, task_state, research_state, emit, event_type: str, message: str) -> None:
        """记录并发出一个运行事件，同时写入 trace。"""
        event = ResearchEvent(task_id=task_state.task_id, event_type=event_type, message=message)
        research_state.events.append(event)
        research_state.updated_at = now_iso()
        self.run_store.append_trace(
            task_state,
            event_type,
            {
                "task_id": task_state.task_id,
                "run_id": task_state.run_id,
                "message": message,
                "created_at": event.created_at,
                "phase": task_state.phase,
            },
        )
        emit(event)

    def persist(self, task_state, research_state, context, trigger: str) -> None:
        """把状态、受预算控制的上下文和 checkpoint 原子写入磁盘。"""
        context_text, context_metadata = context.build()
        task_state.note("{}: context_chars={}".format(trigger, context_metadata["context_chars"]))
        self.run_store.write_task_state(task_state)
        self.run_store.write_research_state(task_state, research_state)
        self.run_store.append_trace(
            task_state,
            "context_built",
            {
                "trigger": trigger,
                "context_metadata": context_metadata,
                "context_preview": context_text[:500],
            },
        )
        checkpoint = create_checkpoint(task_state, research_state, trigger)
        self.run_store.write_checkpoint(task_state, checkpoint)
        self.run_store.write_task_state(task_state)

    def dedupe_sources(self, sources):
        """按规范化 URL 去重，并合并重复来源的摘要和 metadata。"""
        by_url = {}
        for source in sources:
            key = _canonical_url(source.url)
            if key in by_url:
                by_url[key] = _merge_source(by_url[key], source)
            else:
                source.metadata.setdefault("canonical_url", key)
                by_url[key] = source
        return list(by_url.values())


def _canonical_url(url: str) -> str:
    """规范化 URL，过滤跟踪参数和 fragment。"""
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        return url.rstrip("/")
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered_query = [
        (key, value)
        for key, value in query
        if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source", "fbclid", "gclid"}
    ]
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or parsed.path,
        query=urllib.parse.urlencode(filtered_query),
        fragment="",
    )
    return urllib.parse.urlunparse(normalized)


def _merge_source(left: Source, right: Source) -> Source:
    """合并同一 URL 的多次搜索结果。"""
    metadata = dict(left.metadata)
    for key, value in right.metadata.items():
        if not value:
            continue
        if key in {"retrieved_query", "retrieved_by", "connectors"} and metadata.get(key):
            metadata[key] = _append_unique_csv(metadata[key], value)
        else:
            metadata.setdefault(key, value)
    return Source(
        title=left.title or right.title,
        url=left.url,
        kind=left.kind if left.kind != "web" else right.kind,
        snippet=left.snippet if len(left.snippet) >= len(right.snippet) else right.snippet,
        published_at=max(left.published_at or "", right.published_at or ""),
        provider=_append_unique_csv(left.provider, right.provider),
        metadata=metadata,
    )


def _append_unique_csv(left: str, right: str) -> str:
    """合并逗号分隔字段并去重。"""
    values = []
    for raw in [left, right]:
        for item in str(raw or "").split(","):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return ",".join(values)
