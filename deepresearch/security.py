"""搜索结果安全过滤模块。

外部网页或搜索结果可能包含 prompt injection 文本。这个文件负责在结果进入
RAG 和 LLM 前先检查危险指令，避免不可信内容污染 Agent。
"""

import re
from typing import Iterable, List

from .models import Source


class PromptInjectionGuard:
    """Prompt injection 过滤器。

    搜索结果属于不可信上下文，所以进入 RAG/LLM 前先做轻量安全扫描。
    高风险内容直接丢弃；中风险内容保留但写入 metadata，供来源评分降权。
    """

    high_risk_phrases = (
        "ignore previous",
        "ignore all previous",
        "ignore above",
        "disregard previous",
        "system prompt",
        "reveal the prompt",
        "print the prompt",
        "developer message",
        "send your api key",
        "send your token",
        "private key",
        "environment variable",
        "delete files",
        "exfiltrate",
        "rm -rf",
        "~/.ssh",
    )

    medium_risk_phrases = (
        "jailbreak",
        "bypass safety",
        "do not follow",
        "do not obey",
        "you are chatgpt",
        "confidential",
        "base64 decode",
    )

    risky_patterns = (
        re.compile(r"curl\s+[^|]+\|\s*(sh|bash)", re.IGNORECASE),
        re.compile(r"(api[_-]?key|secret|token)\s*[:=]\s*[a-z0-9_\-]{12,}", re.IGNORECASE),
        re.compile(r"tool\s+output.{0,80}(ignore|disregard|override)", re.IGNORECASE),
    )

    def inspect(self, source: Source) -> List[str]:
        """检查单个 Source 是否包含可疑提示词注入短语或模式。"""
        text = self._source_text(source)
        findings = ["high:" + phrase for phrase in self.high_risk_phrases if phrase in text]
        findings.extend("medium:" + phrase for phrase in self.medium_risk_phrases if phrase in text)
        findings.extend("high:" + pattern.pattern for pattern in self.risky_patterns if pattern.search(text))
        return findings

    def filter_sources(self, sources: Iterable[Source]) -> List[Source]:
        """过滤一批 Source；高风险丢弃，中风险打标保留。"""
        safe = []
        for source in sources:
            findings = self.inspect(source)
            if findings:
                source.metadata["security_findings"] = ",".join(findings)
                source.metadata["security_risk"] = self._risk_level(findings)
            if any(finding.startswith("high:") for finding in findings):
                continue
            safe.append(source)
        return safe

    def _source_text(self, source: Source) -> str:
        """把 Source 全量文本压成安全扫描输入。"""
        metadata_text = " ".join(str(value) for value in source.metadata.values())
        return " ".join([source.title, source.snippet, metadata_text]).lower()

    def _risk_level(self, findings: List[str]) -> str:
        """根据 findings 返回风险等级。"""
        if any(finding.startswith("high:") for finding in findings):
            return "high"
        if findings:
            return "medium"
        return "low"
