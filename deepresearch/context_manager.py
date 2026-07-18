"""DeepResearch 的上下文管理。

对齐 Pico 的 context_manager：上下文不是一坨历史文本，而是按目标、工作记忆、
证据池和已验证 claim 分层管理。后续接 LLM 时，这里会负责 prompt 预算和压缩。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .models import Claim, Source, SourceScore


@dataclass
class ContextManager:
    """Agent 的上下文管理器。

    它不是简单保存聊天历史，而是把目标、工作记忆、证据池和已验证结论拆开，
    后续接 LLM 时可以控制 prompt 预算。
    """

    task_goal: str
    working_notes: List[str] = field(default_factory=list)
    evidence_pool: Dict[str, Source] = field(default_factory=dict)
    verified_claims: List[Claim] = field(default_factory=list)
    discarded_notes: List[str] = field(default_factory=list)
    max_context_chars: int = 6000
    max_evidence_items: int = 8
    _source_priorities: Dict[str, float] = field(default_factory=dict)

    def add_note(self, note: str) -> None:
        """加入一条工作记忆；超过上限时把最旧 note 移到 discarded_notes。"""
        self.working_notes.append(note)
        if len(self.working_notes) > 20:
            self.discarded_notes.append(self.working_notes.pop(0))

    def add_sources(self, sources: List[Source]) -> None:
        """把搜索到的来源加入证据池，按 URL 去重。"""
        for source in sources:
            self.evidence_pool.setdefault(source.url, source)

    def add_claims(self, claims: List[Claim]) -> None:
        """把验证后的 claim 加入上下文，供后续报告和反思使用。"""
        known = {(claim.text, tuple(claim.source_urls)) for claim in self.verified_claims}
        for claim in claims:
            key = (claim.text, tuple(claim.source_urls))
            if key not in known:
                self.verified_claims.append(claim)
                known.add(key)

    def prioritize_evidence(self, scores: Dict[str, SourceScore]) -> None:
        """同步来源评分，供上下文选择器优先保留高质量证据。"""
        self._source_priorities = {url: score.final for url, score in scores.items()}

    def build(self) -> tuple:
        """生成压缩后的上下文文本和统计信息。"""
        text, compact_metadata = self._compact_with_metadata()
        metadata = {
            "working_note_count": len(self.working_notes),
            "evidence_count": len(self.evidence_pool),
            "verified_claim_count": len(self.verified_claims),
            "discarded_note_count": len(self.discarded_notes),
            "context_chars": len(text),
            **compact_metadata,
        }
        return text, metadata

    def compact(self) -> str:
        """把上下文压缩成受预算控制的短文本。"""
        return self._compact_with_metadata()[0]

    def _compact_with_metadata(self) -> Tuple[str, Dict[str, int]]:
        """按证据优先级和字符预算构造上下文，避免来源越多 prompt 越失控。"""
        important_sources = sorted(
            self.evidence_pool.values(),
            key=lambda source: self._source_priorities.get(source.url, 0.0),
            reverse=True,
        )[: self.max_evidence_items]
        lines = [
            "Goal: " + self.task_goal,
            "Recent notes:",
            *["- " + note for note in self.working_notes[-8:]],
            "Evidence:",
        ]
        selected_sources = 0
        for source in important_sources:
            priority = self._source_priorities.get(source.url, 0.0)
            excerpt = " ".join(source.snippet.split())[:240]
            line = "- [{:.2f}] {} | {} <{}>".format(priority, source.title, excerpt, source.url)
            if len("\n".join(lines + [line])) > self.max_context_chars:
                break
            lines.append(line)
            selected_sources += 1
        text = "\n".join(lines)[: self.max_context_chars]
        return text, {
            "selected_evidence_count": selected_sources,
            "deferred_evidence_count": max(0, len(self.evidence_pool) - selected_sources),
            "context_budget_chars": self.max_context_chars,
        }
