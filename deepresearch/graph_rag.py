"""Lightweight evidence graph for GraphRAG-style grounding.

The project does not require Neo4j for the MVP, but it still records relations
between key terms, chunks and sources. This gives the report/verifier a
traceable memory graph today and leaves a clean boundary for Neo4j later.
"""

from collections import Counter
from typing import Dict, Iterable, List, Set

from .models import EvidenceChunk, EvidenceRelation


class EvidenceGraphBuilder:
    """Builds a small in-memory evidence graph from EvidenceChunk records."""

    def __init__(self, max_entities_per_chunk: int = 8) -> None:
        self.max_entities_per_chunk = max_entities_per_chunk

    def build(self, chunks: List[EvidenceChunk]) -> List[EvidenceRelation]:
        """Extract entities and co-occurrence relations from evidence chunks."""
        relations: Dict[tuple, EvidenceRelation] = {}
        for chunk in chunks:
            entities = _entities(chunk.text, self.max_entities_per_chunk)
            for entity in entities:
                _merge_relation(
                    relations,
                    EvidenceRelation(
                        source=chunk.source_title or chunk.source_url,
                        relation="MENTIONS",
                        target=entity,
                        chunk_id=chunk.chunk_id,
                        source_url=chunk.source_url,
                        weight=1.0,
                    ),
                )
            for left, right in _pairs(entities):
                _merge_relation(
                    relations,
                    EvidenceRelation(
                        source=left,
                        relation="CO_OCCURS_WITH",
                        target=right,
                        chunk_id=chunk.chunk_id,
                        source_url=chunk.source_url,
                        weight=1.0,
                    ),
                )
        return sorted(relations.values(), key=lambda item: item.weight, reverse=True)


def _merge_relation(relations: Dict[tuple, EvidenceRelation], relation: EvidenceRelation) -> None:
    key = (relation.source, relation.relation, relation.target, relation.source_url)
    if key in relations:
        relations[key].weight += relation.weight
        return
    relations[key] = relation


def _entities(text: str, limit: int) -> List[str]:
    """Extract stable project/framework terms without requiring a heavy NER model."""
    tokens = _tokens(text)
    candidates = []
    for token in tokens:
        if _looks_like_entity(token):
            candidates.append(token)
    if not candidates:
        candidates = [term for term, _ in Counter(tokens).most_common(limit)]
    return _unique(candidates)[:limit]


def _tokens(text: str) -> List[str]:
    normalized = text.replace("，", " ").replace("。", " ")
    for char in ",.()[]{}:;!?/\"'|":
        normalized = normalized.replace(char, " ")
    return [token.strip() for token in normalized.split() if len(token.strip()) > 2]


def _looks_like_entity(token: str) -> bool:
    has_upper = any(char.isupper() for char in token)
    has_digit = any(char.isdigit() for char in token)
    has_symbol = any(char in token for char in "-_+")
    known = token.lower() in {
        "rag",
        "mcp",
        "bm25",
        "rrf",
        "github",
        "arxiv",
        "langgraph",
        "langchain",
        "deepseek",
        "fastapi",
        "sqlite",
    }
    return known or has_upper or has_digit or has_symbol


def _pairs(values: Iterable[str]) -> Set[tuple]:
    items = list(values)
    pairs = set()
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left != right:
                pairs.add((left, right))
    return pairs


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
