"""Lightweight hybrid RAG retriever for evidence grounding.

Search connectors first discover candidate sources. This retriever then turns
those sources into EvidenceChunk records, indexes them in the local vector store,
and retrieves evidence with hybrid ranking: dense vector similarity, BM25 lexical
retrieval and RRF fusion. The boundary can later be swapped for Elasticsearch,
reranker, FAISS, Chroma, Milvus or pgvector without changing the LangGraph flow.
"""

import math
from collections import Counter
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
        self._chunk_terms: Dict[str, List[str]] = {}
        self._document_frequency: Counter = Counter()
        self._average_document_length: float = 1.0

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
        self._rebuild_lexical_index(chunks)
        self.vector_store.chunks = {}
        self.vector_store.vectors = {}
        self.vector_store.upsert(chunks)
        return chunks

    def retrieve(self, query: str, limit: int = 5) -> List[EvidenceChunk]:
        """根据查询用 BM25 + 向量检索 + RRF 融合返回最相关证据片段。"""
        vector_results = self.vector_store.search(query, max(limit, len(self.chunks)))
        bm25_results = self._bm25_search(query, max(limit, len(self.chunks)))
        fused_scores = _rrf_fuse([vector_results, bm25_results])
        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        for chunk_id, score in ranked[:limit]:
            chunk = self.chunks[chunk_id]
            results.append(EvidenceChunk(chunk.chunk_id, chunk.source_url, chunk.source_title, chunk.text, round(score, 4)))
        return results

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

    def _rebuild_lexical_index(self, chunks: List[EvidenceChunk]) -> None:
        """建立轻量 BM25 词项索引，后续可替换成 Elasticsearch/OpenSearch。"""
        self._chunk_terms = {chunk.chunk_id: _tokens(chunk.text) for chunk in chunks}
        self._document_frequency = Counter()
        for terms in self._chunk_terms.values():
            self._document_frequency.update(set(terms))
        total_terms = sum(len(terms) for terms in self._chunk_terms.values())
        self._average_document_length = total_terms / max(1, len(self._chunk_terms))

    def _bm25_search(self, query: str, limit: int) -> List[EvidenceChunk]:
        """用 BM25 计算 query 与 EvidenceChunk 的词项相关性。"""
        query_terms = _tokens(query)
        scored: List[Tuple[float, EvidenceChunk]] = []
        for chunk_id, terms in self._chunk_terms.items():
            score = _bm25_score(query_terms, terms, self._document_frequency, len(self._chunk_terms), self._average_document_length)
            if score > 0:
                chunk = self.chunks[chunk_id]
                scored.append((score, EvidenceChunk(chunk.chunk_id, chunk.source_url, chunk.source_title, chunk.text, round(score, 4))))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]


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
    return set(_tokens(text))


def _tokens(text: str) -> List[str]:
    """提取检索词项，供 BM25 和关键词重合复用。"""
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
    return [token for token in normalized.split() if len(token) > 2 and token not in stop_words]


def _bm25_score(query_terms: List[str], document_terms: List[str], document_frequency: Counter, document_count: int, average_document_length: float) -> float:
    """计算一个 chunk 的 BM25 分数。"""
    if not query_terms or not document_terms:
        return 0.0
    term_frequency = Counter(document_terms)
    score = 0.0
    k1 = 1.5
    b = 0.75
    doc_len = len(document_terms)
    for term in query_terms:
        frequency = term_frequency.get(term, 0)
        if frequency == 0:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        numerator = frequency * (k1 + 1)
        denominator = frequency + k1 * (1 - b + b * doc_len / max(1.0, average_document_length))
        score += idf * numerator / denominator
    return score


def _rrf_fuse(result_lists: List[List[EvidenceChunk]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion，把向量和 BM25 的排序融合成一个分数。"""
    scores: Dict[str, float] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    return scores
