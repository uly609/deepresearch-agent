"""Runtime facade for DeepResearch Agent.

这个文件对齐 Pico 的 `runtime.py`：负责把 Planner、工具、上下文、安全、
RunStore 和 LangGraph 工作流装配成一个可以执行 `ask()` 的运行时对象。
"""

from .agents import ConflictDetectorAgent, CitationVerifierAgent, PlannerAgent, ReflectionAgent, ReportAgent, ReportAuditorAgent, SourceEvaluatorAgent
from .llm_provider import build_llm_provider
from .models import ResearchEvent, now_iso
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
    ) -> None:
        """根据配置创建一次可执行的 Agent runtime。"""
        self.run_store = run_store or RunStore()
        self.engine = engine
        self.use_live_tools = use_live_tools
        self.use_llm = use_llm
        self.llm = build_llm_provider(use_llm)
        self.planner = PlannerAgent(self.llm)
        self.tools = tools or ToolRegistry(default_research_connectors(use_live=use_live_tools))
        self.guard = PromptInjectionGuard()
        self.rag = EvidenceRetriever(build_vector_store())
        self.evaluator = SourceEvaluatorAgent()
        self.verifier = CitationVerifierAgent(self.llm)
        self.conflict_detector = ConflictDetectorAgent()
        self.reflector = ReflectionAgent()
        self.reporter = ReportAgent(self.llm)
        self.report_auditor = ReportAuditorAgent()
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
        """把当前 task、research、context 摘要写入磁盘。"""
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

    def dedupe_sources(self, sources):
        """按 URL 对来源去重。"""
        by_url = {}
        for source in sources:
            by_url.setdefault(source.url, source)
        return list(by_url.values())
