"""Fetch and clean full web-page text before RAG chunking."""

import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List

from .models import Source


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "aside", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "aside", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


class ContentFetcher:
    """受开关和长度约束的网页正文提取器，避免搜索摘要成为唯一 RAG 证据。"""

    def __init__(self, enabled=False, max_chars=12000, timeout_seconds=8):
        self.enabled = enabled
        self.max_chars = max_chars
        self.timeout_seconds = timeout_seconds

    def enrich(self, sources: List[Source]) -> List[Source]:
        if not self.enabled:
            return sources
        for source in sources:
            if source.kind not in {"web", "official"} or source.metadata.get("content"):
                continue
            parsed = urllib.parse.urlparse(source.url)
            if parsed.scheme not in {"http", "https"}:
                continue
            try:
                request = urllib.request.Request(source.url, headers={"User-Agent": "deepresearch-agent"})
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    html = response.read(self.max_chars * 4).decode("utf-8", errors="ignore")
                parser = _TextExtractor()
                parser.feed(html)
                content = " ".join(parser.parts)[: self.max_chars]
                if content:
                    source.metadata["content"] = content
                    source.metadata["content_fetch_status"] = "success"
            except Exception as exc:
                source.metadata["content_fetch_status"] = "failed"
                source.metadata["content_fetch_error"] = type(exc).__name__
        return sources


def build_content_fetcher(enabled=False):
    return ContentFetcher(enabled=enabled or os.environ.get("FETCH_SOURCE_CONTENT") == "1")
