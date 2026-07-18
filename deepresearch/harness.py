"""Agent Harness for runtime governance and observability.

The harness is the engineering layer around the LangGraph workflow. It does not
decide research content; it controls execution budgets, records node telemetry
and prevents the agent loop from growing without bounds.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from .models import HarnessStep


@dataclass
class RunPolicy:
    """Runtime limits for one DeepResearch run."""

    max_search_rounds: int = 2
    max_tool_steps: int = 40
    max_sources: int = 120
    max_queries_per_round: int = 12


class AgentHarness:
    """Small runtime harness for a controlled Agent workflow."""

    def __init__(self, policy: RunPolicy = None) -> None:
        self.policy = policy or RunPolicy()

    def start_node(self, node: str) -> Dict[str, Any]:
        """Create a timing token for a node execution."""
        return {"node": node, "started_at": time.perf_counter()}

    def finish_node(self, token: Dict[str, Any], task_state, research_state, status: str = "success", reason: str = "") -> None:
        """Record one node execution in ResearchState and TaskState notes."""
        duration_ms = int((time.perf_counter() - token["started_at"]) * 1000)
        step = HarnessStep(
            node=token["node"],
            status=status,
            duration_ms=duration_ms,
            phase=task_state.phase,
            reason=reason,
        )
        research_state.harness_steps.append(step)
        task_state.note("harness:{}:{}:{}ms".format(step.node, step.status, step.duration_ms))

    def limit_queries(self, pending_queries: List[dict], task_state) -> List[dict]:
        """Trim query fan-out according to per-round and total tool budgets."""
        remaining_tool_steps = max(0, self.policy.max_tool_steps - task_state.tool_steps)
        allowed = min(self.policy.max_queries_per_round, remaining_tool_steps)
        if allowed <= 0:
            task_state.note("harness:query_budget_exhausted")
            return []
        trimmed = pending_queries[:allowed]
        dropped = len(pending_queries) - len(trimmed)
        if dropped > 0:
            task_state.note("harness:trimmed_queries={}".format(dropped))
        return trimmed

    def limit_sources(self, sources: List[Any], task_state) -> List[Any]:
        """Cap collected sources so downstream RAG and LLM context stay bounded."""
        if len(sources) <= self.policy.max_sources:
            return sources
        task_state.note("harness:trimmed_sources={}".format(len(sources) - self.policy.max_sources))
        return sources[: self.policy.max_sources]

    def should_rewrite_after_reflect(self, gaps: List[str], search_round: int) -> bool:
        """Decide whether the workflow may loop back for another search round."""
        return bool(gaps) and search_round < self.policy.max_search_rounds
