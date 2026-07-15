"""搜索结果安全过滤模块。

外部网页或搜索结果可能包含 prompt injection 文本。这个文件负责在结果进入
RAG 和 LLM 前先检查危险指令，避免不可信内容污染 Agent。
"""

from typing import Iterable, List

from .models import Source


class PromptInjectionGuard:
    """简单的 prompt injection 过滤器。"""

    suspicious_phrases = (
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "developer message",
        "send your api key",
        "delete files",
        "exfiltrate",
    )

    def inspect(self, source: Source) -> List[str]:
        """检查单个 Source 是否包含可疑提示词注入短语。"""
        text = (source.title + " " + source.snippet).lower()
        return [phrase for phrase in self.suspicious_phrases if phrase in text]

    def filter_sources(self, sources: Iterable[Source]) -> List[Source]:
        """过滤一批 Source；发现风险时标记 metadata 并丢弃该来源。"""
        safe = []
        for source in sources:
            findings = self.inspect(source)
            if findings:
                source.metadata["security_findings"] = ",".join(findings)
                continue
            safe.append(source)
        return safe
