"""Focused tests for context budgeting, tool audit and structured report storage."""

import tempfile
import unittest

from deepresearch.context_manager import ContextManager
from deepresearch.agents import ReportAgent
from deepresearch.models import Claim, Source, SourceScore
from deepresearch.run_store import RunStore
from deepresearch.task_state import TaskState
from deepresearch.tools import SourceConnector, ToolRegistry


def make_source(url: str, title: str = "source") -> Source:
    return Source(title=title, url=url, kind="web", snippet="evidence", published_at="2026-01-01", provider="test")


class GoodConnector(SourceConnector):
    name = "good"

    def search(self, query):
        return [make_source("https://example.com/" + query, "good result")]


class BrokenConnector(SourceConnector):
    name = "broken"

    def search(self, query):
        raise RuntimeError("network unavailable")


class AgentGovernanceTests(unittest.TestCase):
    def test_context_prefers_high_score_sources_within_budget(self):
        context = ContextManager(task_goal="research", max_context_chars=200, max_evidence_items=2)
        low = make_source("https://low.example", "low")
        high = make_source("https://high.example", "high")
        context.add_sources([low, high])
        context.prioritize_evidence({
            low.url: SourceScore(0, 0, 0, 0, 0.2, "low"),
            high.url: SourceScore(0, 0, 0, 0, 0.9, "high"),
        })
        text, metadata = context.build()
        self.assertLessEqual(len(text), 200)
        self.assertLess(text.find("high"), text.find("low"))
        self.assertEqual(metadata["selected_evidence_count"], 2)

    def test_tool_registry_audits_failure_without_stopping_other_tools(self):
        registry = ToolRegistry([GoodConnector(), BrokenConnector()])
        results = registry.search_all("langgraph")
        self.assertEqual(len(results), 1)
        self.assertEqual([audit.status for audit in registry.last_audits], ["success", "failed"])

    def test_report_json_contains_structured_report(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            task = TaskState.create("research")
            store.start_run(task)
            store.write_report(task, "# report\n", {"schema_version": "research_report.v1"}, {"question": "research"})
            payload = store.read_json(store.report_json_path(task))
            self.assertEqual(payload["report"]["question"], "research")

    def test_rule_report_summary_uses_current_question_not_fixed_topic(self):
        report = ReportAgent()._rule_summary(
            "compare agent frameworks",
            [make_source("https://example.com/a", "Framework A")],
            {"https://example.com/a": SourceScore(0, 0, 0, 0, 0.9, "ok")},
            [Claim("Framework A has stateful workflows.", ["https://example.com/a"], 0.9, "supported")],
            [],
        )
        self.assertIn("compare agent frameworks", report)
        self.assertNotIn("Python 应该作为主实现语言", report)


if __name__ == "__main__":
    unittest.main()
