"""Lightweight RAG retriever for evidence grounding.

This is intentionally small and dependency-free: it chunks source snippets and
uses keyword overlap for retrieval. It gives the project a real retriever
boundary that can later be replaced by embeddings and a vector database.
"""

from typing import Dict, List, Set

from .models import EvidenceChunk, Source
from .vector_store import LocalVectorStore


class EvidenceRetriever:
    """RAG 证据检索器。

    它负责把 Source 转成 EvidenceChunk，写入向量库，并按 query 找最相关的
    证据片段给 CitationVerifier 使用。
    """

    def __init__(self, vector_store: LocalVectorStore = None) -> None:
        """初始化内存 chunk 索引和向量库。"""
        self.chunks: Dict[str, EvidenceChunk] = {}
        self.vector_store = vector_store or LocalVectorStore()

    def index_sources(self, sources: List[Source]) -> List[EvidenceChunk]:
        """把 Source 拼成证据文本，生成 EvidenceChunk 并写入向量库。"""
        for source in sources:
            text = " ".join([source.title, source.snippet, " ".join(source.metadata.values())]).strip()
            chunk = EvidenceChunk(
                chunk_id=_chunk_id(source.url),
                source_url=source.url,
                source_title=source.title,
                text=text,
            )
            self.chunks[chunk.chunk_id] = chunk
        chunks = list(self.chunks.values())
        self.vector_store.upsert(chunks)
        return chunks

    def retrieve(self, query: str, limit: int = 5) -> List[EvidenceChunk]:
        """根据查询从向量库中检索最相关的证据片段。"""
        vector_results = self.vector_store.search(query, limit)
        if vector_results:
            return vector_results
        query_terms = _keywords(query)
        ranked = []
        for chunk in self.chunks.values():
            chunk_terms = _keywords(chunk.text)
            overlap = len(query_terms & chunk_terms)
            score = overlap / max(1, len(query_terms))
            if overlap:
                ranked.append(EvidenceChunk(chunk.chunk_id, chunk.source_url, chunk.source_title, chunk.text, round(score, 2)))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def best_for_source(self, source_url: str, query: str) -> EvidenceChunk:
        """在指定来源里找最适合支撑当前 query 的证据片段。"""
        candidates = [chunk for chunk in self.retrieve(query, limit=20) if chunk.source_url == source_url]
        if candidates:
            return candidates[0]
        for chunk in self.chunks.values():
            if chunk.source_url == source_url:
                return chunk
        return EvidenceChunk("", source_url, "", "")


def _chunk_id(url: str) -> str:
    """根据 URL 生成稳定的 chunk_id。"""
    safe = "".join(char if char.isalnum() else "_" for char in url)
    return "chunk_" + safe[-64:]


def _keywords(text: str) -> Set[str]:
    """提取简单关键词，用于向量不可用时的兜底检索。"""
    normalized = text.lower()
    for char in "，。,.()[]:;!?/\"'":
        normalized = normalized.replace(char, " ")
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "http",
        "https",
    }
    return {token for token in normalized.split() if len(token) > 2 and token not in stop_words}
