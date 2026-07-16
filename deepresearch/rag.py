"""Lightweight hybrid RAG retriever for evidence grounding.

Search connectors first discover candidate sources. This retriever then turns
those sources into EvidenceChunk records, indexes them in the local vector store,
and retrieves evidence with a hybrid score: vector similarity plus lexical
keyword overlap. The boundary can later be swapped for BM25 + dense retriever,
reranker, FAISS, Chroma, Milvus or pgvector without changing the LangGraph flow.
"""

from typing import Dict, List, Set, Tuple

from .models import EvidenceChunk, Source
from .vector_store import LocalVectorStore

CHUNK_SIZE = 120
CHUNK_OVERLAP = 24
NON_EVIDENCE_METADATA_KEYS = {
    "retrieved_query",
    "retrieved_by",
    "connectors",
    "canonical_url",
    "security_findings",
    "security_risk",
}


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
        """把 Source 拼成证据文本，切分为 EvidenceChunk 并写入向量库。"""
        current_chunk_ids = set()
        for source in sources:
            text = _source_text(source)
            for index, chunk_text in enumerate(_chunk_text(text)):
                chunk = EvidenceChunk(
                    chunk_id=_chunk_id(source.url, index),
                    source_url=source.url,
                    source_title=source.title,
                    text=chunk_text,
                )
                self.chunks[chunk.chunk_id] = chunk
                current_chunk_ids.add(chunk.chunk_id)
        self.chunks = {chunk_id: chunk for chunk_id, chunk in self.chunks.items() if chunk_id in current_chunk_ids}
        chunks = list(self.chunks.values())
        self.vector_store.chunks = {}
        self.vector_store.vectors = {}
        self.vector_store.upsert(chunks)
        return chunks

    def retrieve(self, query: str, limit: int = 5) -> List[EvidenceChunk]:
        """根据查询用向量相似度 + 关键词重合检索最相关证据片段。"""
        vector_results = self.vector_store.search(query, max(limit, len(self.chunks)))
        vector_scores = {chunk.chunk_id: chunk.score for chunk in vector_results}
        query_terms = _keywords(query)
        ranked: List[Tuple[float, EvidenceChunk]] = []
        for chunk in self.chunks.values():
            chunk_terms = _keywords(chunk.text)
            overlap = len(query_terms & chunk_terms)
            lexical_score = overlap / max(1, len(query_terms))
            vector_score = vector_scores.get(chunk.chunk_id, 0.0)
            score = round(vector_score * 0.65 + lexical_score * 0.35, 4)
            if score > 0:
                ranked.append(
                    (
                        score,
                        EvidenceChunk(chunk.chunk_id, chunk.source_url, chunk.source_title, chunk.text, score),
                    )
                )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:limit]]

    def source_similarity(self, source_url: str, query: str) -> float:
        """返回某个来源和 query 的向量相似度，用于 source-level relevance。"""
        limit = max(1, len(self.chunks))
        for chunk in self.retrieve(query, limit=limit):
            if chunk.source_url == source_url:
                return chunk.score
        return 0.0

    def best_for_source(self, source_url: str, query: str) -> EvidenceChunk:
        """在指定来源里找最适合支撑当前 query 的证据片段。"""
        candidates = [chunk for chunk in self.retrieve(query, limit=20) if chunk.source_url == source_url]
        if candidates:
            return candidates[0]
        for chunk in self.chunks.values():
            if chunk.source_url == source_url:
                return chunk
        return EvidenceChunk("", source_url, "", "")


def _source_text(source: Source) -> str:
    """把 Source 的标题、摘要和 metadata 拼成可检索证据文本。"""
    parts = [source.title, source.snippet]
    for key, value in source.metadata.items():
        if value and key not in NON_EVIDENCE_METADATA_KEYS:
            parts.append("{}: {}".format(key, value))
    return " ".join(parts).strip()


def _chunk_text(text: str) -> List[str]:
    """按词窗口切分文本，长来源会生成多个 EvidenceChunk。"""
    tokens = text.split()
    if not tokens:
        return [text]
    if len(tokens) <= CHUNK_SIZE:
        return [" ".join(tokens)]
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for start in range(0, len(tokens), step):
        window = tokens[start:start + CHUNK_SIZE]
        if window:
            chunks.append(" ".join(window))
        if start + CHUNK_SIZE >= len(tokens):
            break
    return chunks


def _chunk_id(url: str, index: int) -> str:
    """根据 URL 生成稳定的 chunk_id。"""
    safe = "".join(char if char.isalnum() else "_" for char in url)
    return "chunk_{}_{}".format(safe[-64:], index)


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
