"""DeepResearch 的上下文管理。

对齐 Pico 的 context_manager：上下文不是一坨历史文本，而是按目标、工作记忆、
证据池和已验证 claim 分层管理。后续接 LLM 时，这里会负责 prompt 预算和压缩。
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .models import Claim, Source


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
        self.verified_claims.extend(claims)

    def build(self) -> tuple:
        """生成压缩后的上下文文本和统计信息。"""
        text = self.compact()
        metadata = {
            "working_note_count": len(self.working_notes),
            "evidence_count": len(self.evidence_pool),
            "verified_claim_count": len(self.verified_claims),
            "discarded_note_count": len(self.discarded_notes),
            "context_chars": len(text),
        }
        return text, metadata

    def compact(self) -> str:
        """把上下文压缩成可放进 prompt 或 trace 的短文本。"""
        important_sources = list(self.evidence_pool.values())[:8]
        lines = [
            "Goal: " + self.task_goal,
            "Recent notes:",
            *["- " + note for note in self.working_notes[-8:]],
            "Evidence:",
            *["- " + source.title + " <" + source.url + ">" for source in important_sources],
        ]
        return "\n".join(lines)
