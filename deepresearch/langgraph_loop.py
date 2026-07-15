"""LangGraph implementation of the DeepResearch workflow.

This file is the main DeepResearch workflow.
It maps DeepResearch stages to LangGraph nodes:

plan -> search -> dedupe -> score -> verify -> reflect -> report
"""

from typing import Any, List, TypedDict

from .checkpoint import create_checkpoint
from .context_manager import ContextManager
from .models import ResearchState, now_iso
from .task_state import TaskState


class GraphState(TypedDict):
    """LangGraph 节点之间传递的状态结构。"""

    question: str
    task_state: TaskState
    research_state: ResearchState
    context: ContextManager
    collected: List[Any]
    gaps: List[str]
    search_round: int
    searched_queries: List[str]
    emit: Any


class LangGraphAgentLoop:
    """DeepResearch 的 LangGraph 工作流实现。

    它把研究任务拆成 plan/search/dedupe/score/verify/reflect/report 等节点，
    并通过条件边支持反思后回到 search 补搜。
    """

    def __init__(self, runtime) -> None:
        """保存 runtime，并编译 LangGraph 状态图。"""
        self.runtime = runtime
        self.graph = self._compile_graph()

    def run(self, question: str, emit=lambda event: None) -> ResearchState:
        """创建初始状态并启动 LangGraph 执行。"""
        task_state = self.runtime.create_task_state(question)
        research_state = ResearchState(question=question, task_id=task_state.task_id, status="running")
        context = ContextManager(task_goal=question)
        self.runtime.current_task_state = task_state
        self.runtime.current_research_state = research_state
        self.runtime.run_store.start_run(task_state)

        result = self.graph.invoke(
            {
                "question": question,
                "task_state": task_state,
                "research_state": research_state,
                "context": context,
                "collected": [],
                "gaps": [],
                "search_round": 0,
                "searched_queries": [],
                "emit": emit,
            }
        )
        return result["research_state"]

    def _compile_graph(self):
        """声明 LangGraph 节点、普通边和 reflect 后的条件边。"""
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph is not installed. Install it with: python3 -m pip install langgraph"
            ) from exc

        graph = StateGraph(GraphState)
        graph.add_node("start", self._start)
        graph.add_node("plan", self._plan)
        graph.add_node("search", self._search)
        graph.add_node("dedupe", self._dedupe)
        graph.add_node("score", self._score)
        graph.add_node("verify", self._verify)
        graph.add_node("reflect", self._reflect)
        graph.add_node("report", self._report)

        graph.add_edge(START, "start")
        graph.add_edge("start", "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "dedupe")
        graph.add_edge("dedupe", "score")
        graph.add_edge("score", "verify")
        graph.add_edge("verify", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "search": "search",
                "report": "report",
            },
        )
        graph.add_edge("report", END)
        return graph.compile()

    def _start(self, state: GraphState) -> GraphState:
        """开始节点：发出 run_started 事件。"""
        runtime = self.runtime
        task_state = state["task_state"]
        research_state = state["research_state"]
        emit = state["emit"]
        runtime.emit(task_state, research_state, emit, "run_started", "研究任务已开始")
        return state

    def _plan(self, state: GraphState) -> GraphState:
        """规划节点：把用户问题拆成多个可搜索的子问题。"""
        runtime = self.runtime
        question = state["question"]
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]

        task_state.record_attempt("planning")
        plan = runtime.planner.plan(question)
        research_state.plan = plan
        context.add_note("已创建 {} 个子问题".format(len(plan.sub_questions)))
        runtime.emit(task_state, research_state, emit, "plan_created", "已创建 {} 个子问题".format(len(plan.sub_questions)))
        task_state.phase = "planned"
        runtime.persist(task_state, research_state, context, "planned")
        return state

    def _search(self, state: GraphState) -> GraphState:
        """搜索节点：调用 ToolRegistry 搜索来源，并做安全过滤。"""
        runtime = self.runtime
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]
        collected = state["collected"]
        searched_queries = state["searched_queries"]
        plan = research_state.plan
        if plan is None:
            raise RuntimeError("cannot search before plan is created")

        queries = state["gaps"] if state["gaps"] else plan.sub_questions
        state["gaps"] = []
        state["search_round"] = state["search_round"] + 1
        for sub_question in queries:
            if sub_question in searched_queries:
                continue
            searched_queries.append(sub_question)
            task_state.record_tool("search_all")
            runtime.emit(task_state, research_state, emit, "search_started", sub_question)
            results = runtime.tools.search_all(sub_question)
            results = runtime.guard.filter_sources(results)
            collected.extend(results)
            context.add_sources(results)
            runtime.emit(task_state, research_state, emit, "search_finished", "已收集 {} 个候选来源".format(len(results)))
            runtime.persist(task_state, research_state, context, "search_step")
        state["collected"] = collected
        state["searched_queries"] = searched_queries
        return state

    def _dedupe(self, state: GraphState) -> GraphState:
        """去重/RAG 节点：按 URL 去重来源，并建立 evidence chunks 向量索引。"""
        runtime = self.runtime
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]
        research_state.sources = runtime.dedupe_sources(state["collected"])
        research_state.evidence_chunks = runtime.rag.index_sources(research_state.sources)
        runtime.run_store.write_vector_store(task_state, runtime.rag.vector_store)
        runtime.emit(task_state, research_state, emit, "sources_deduped", "保留了 {} 个去重后的来源".format(len(research_state.sources)))
        task_state.phase = "searched"
        runtime.persist(task_state, research_state, context, "searched")
        return state

    def _score(self, state: GraphState) -> GraphState:
        """评分节点：对每个来源计算权威性、新鲜度、相关性和风险分数。"""
        runtime = self.runtime
        question = state["question"]
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]
        research_state.scores = runtime.evaluator.score(research_state.sources, question)
        runtime.emit(task_state, research_state, emit, "sources_scored", "已对来源的权威性、新鲜度、相关性和风险完成评分")
        task_state.phase = "scored"
        runtime.persist(task_state, research_state, context, "scored")
        return state

    def _verify(self, state: GraphState) -> GraphState:
        """引用验证节点：生成 claim，绑定证据片段，并检测来源冲突。"""
        runtime = self.runtime
        question = state["question"]
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]
        research_state.claims = runtime.verifier.verify(question, research_state.sources, research_state.scores, runtime.rag)
        research_state.conflicts = runtime.conflict_detector.detect(research_state.claims)
        context.add_claims(research_state.claims)
        runtime.emit(
            task_state,
            research_state,
            emit,
            "citations_verified",
            "已验证 {} 条证据支撑的结论，发现 {} 个冲突".format(
                len(research_state.claims),
                len(research_state.conflicts),
            ),
        )
        task_state.phase = "verified"
        runtime.persist(task_state, research_state, context, "verified")
        return state

    def _reflect(self, state: GraphState) -> GraphState:
        """反思节点：检查是否缺少关键证据，必要时触发补搜。"""
        runtime = self.runtime
        question = state["question"]
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]
        gaps = runtime.reflector.find_gaps(question, research_state.claims, research_state.sources)
        if gaps:
            context.add_note("反思发现缺口：" + "；".join(gaps))
            runtime.emit(task_state, research_state, emit, "reflection_finished", "发现 {} 个后续缺口".format(len(gaps)))
        else:
            runtime.emit(task_state, research_state, emit, "reflection_finished", "未发现明显的后续缺口")
        task_state.phase = "reflected"
        runtime.persist(task_state, research_state, context, "reflected")
        state["gaps"] = gaps
        return state

    def _route_after_reflect(self, state: GraphState) -> str:
        """条件路由：有缺口且未超过轮次时回到 search，否则进入 report。"""
        if state["gaps"] and state["search_round"] < 2:
            return "search"
        return "report"

    def _report(self, state: GraphState) -> GraphState:
        """报告节点：生成中文报告、做报告级审计，并写入最终工件。"""
        runtime = self.runtime
        question = state["question"]
        task_state = state["task_state"]
        research_state = state["research_state"]
        context = state["context"]
        emit = state["emit"]
        plan = research_state.plan
        if plan is None:
            raise RuntimeError("cannot report before plan is created")

        base_report = runtime.reporter.write(
            question,
            plan,
            research_state.sources,
            research_state.scores,
            research_state.claims,
            research_state.conflicts,
            state["gaps"],
        )
        research_state.report_checks = runtime.report_auditor.audit(base_report, research_state.claims)
        research_state.report_markdown = runtime.report_auditor.append_section(base_report, research_state.report_checks)
        research_state.status = "completed"
        research_state.updated_at = now_iso()
        report_path = runtime.run_store.write_report(
            task_state,
            research_state.report_markdown,
            {
                "task_id": task_state.task_id,
                "run_id": task_state.run_id,
                "source_count": len(research_state.sources),
                "claim_count": len(research_state.claims),
                "engine": "langgraph",
            },
        )
        weak_claim_count = len([claim for claim in research_state.claims if claim.status != "supported"])
        task_state.finish_success(str(report_path), len(research_state.sources), len(research_state.claims), weak_claim_count)
        runtime.emit(task_state, research_state, emit, "run_finished", "报告已生成")
        runtime.persist(task_state, research_state, context, "finished")
        checkpoint = create_checkpoint(task_state, research_state, "run_finished")
        runtime.run_store.write_checkpoint(task_state, checkpoint)
        runtime.run_store.write_task_state(task_state)
        return state
