"""DeepResearch 的评估 Harness。

EvalHarness 用固定问题跑完整 Agent 流程，并把完成度、来源数、引用支持率、
证据覆盖率、报告审计等结果量化出来。
"""

from dataclasses import dataclass
from typing import List

from .models import ResearchState
from .run_store import RunStore
from .runtime import DeepResearchRuntime


@dataclass
class EvalResult:
    """单条评估用例的量化结果。"""

    question: str
    completed: bool
    source_count: int
    claim_count: int
    weak_claim_count: int
    citation_support_rate: float
    evidence_excerpt_rate: float
    conflict_count: int
    report_grounding_rate: float
    uncited_sentence_count: int
    source_diversity: int
    report_has_gaps_section: bool
    report_has_conflict_section: bool
    score: float


class EvalHarness:
    """轻量评估 Harness。

    它用固定问题跑完整 Agent，再统计来源数、claim 数、引用支持率、报告审计等指标。
    """

    def __init__(self, runtime: DeepResearchRuntime) -> None:
        """保存要被评估的 runtime。"""
        self.runtime = runtime

    def run_cases(self, questions: List[str]) -> List[EvalResult]:
        """批量运行评估问题。"""
        return [self._score(self.runtime.ask(question)) for question in questions]

    def _score(self, state: ResearchState) -> EvalResult:
        """根据 ResearchState 计算一组评估指标。"""
        weak_claims = [claim for claim in state.claims if claim.status != "supported"]
        supported_claims = [claim for claim in state.claims if claim.status == "supported"]
        evidence_claims = [claim for claim in state.claims if claim.evidence_excerpt]
        source_diversity = len({source.provider for source in state.sources})
        citation_support_rate = len(supported_claims) / len(state.claims) if state.claims else 0.0
        evidence_excerpt_rate = len(evidence_claims) / len(state.claims) if state.claims else 0.0
        conflict_count = len(state.conflicts)
        grounded_checks = [check for check in state.report_checks if check.status == "grounded"]
        uncited_checks = [check for check in state.report_checks if check.status == "uncited"]
        report_grounding_rate = len(grounded_checks) / len(state.report_checks) if state.report_checks else 1.0
        report_has_gaps_section = "## 剩余缺口" in state.report_markdown
        report_has_conflict_section = "## 冲突检查" in state.report_markdown
        score = 0.0
        if state.status == "completed":
            score += 0.25
        score += min(0.2, len(state.sources) * 0.025)
        score += min(0.2, len(state.claims) * 0.025)
        score += min(0.15, source_diversity * 0.04)
        score += citation_support_rate * 0.15
        score += evidence_excerpt_rate * 0.05
        if conflict_count == 0:
            score += 0.03
        if report_has_conflict_section:
            score += 0.02
        score += report_grounding_rate * 0.05
        if report_has_gaps_section:
            score += 0.05
        return EvalResult(
            question=state.question,
            completed=state.status == "completed",
            source_count=len(state.sources),
            claim_count=len(state.claims),
            weak_claim_count=len(weak_claims),
            citation_support_rate=round(citation_support_rate, 2),
            evidence_excerpt_rate=round(evidence_excerpt_rate, 2),
            conflict_count=conflict_count,
            report_grounding_rate=round(report_grounding_rate, 2),
            uncited_sentence_count=len(uncited_checks),
            source_diversity=source_diversity,
            report_has_gaps_section=report_has_gaps_section,
            report_has_conflict_section=report_has_conflict_section,
            score=round(min(score, 1.0), 2),
        )


def default_harness(engine: str = "langgraph", use_live_tools: bool = False, use_llm: bool = False) -> EvalHarness:
    """创建默认评估 Harness。"""
    return EvalHarness(DeepResearchRuntime(RunStore(), engine=engine, use_live_tools=use_live_tools, use_llm=use_llm))
