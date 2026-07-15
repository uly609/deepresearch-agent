"""Small local vector store used by the RAG retriever.

It uses hashed bag-of-words vectors so the project has a real vector-store
boundary without extra dependencies. The implementation can later be swapped
for FAISS, Chroma, Milvus or pgvector without changing the agent workflow.
"""

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .models import EvidenceChunk


class EmbeddingProvider:
    """Embedding Provider 抽象基类。

    具体实现可以是远程 OpenAI-compatible embedding，也可以是本地
    sentence-transformers 模型。
    """

    dimensions = 128

    def available(self) -> bool:
        """判断当前 embedding provider 是否可用。"""
        return False

    def embed(self, text: str) -> List[float]:
        """把文本转换成向量；基类返回空向量。"""
        return []


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int = 1024,
        timeout_seconds: int = 20,
    ) -> None:
        """保存远程 embedding 服务配置。"""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        """有 API key 和模型名时认为远程 embedding 可用。"""
        return bool(self.api_key and self.model)

    def embed(self, text: str) -> List[float]:
        """调用远程 embeddings 接口，把文本变成向量。"""
        if not self.available():
            return []
        payload = {"model": self.model, "input": text[:8000]}
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "deepresearch-agent",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []
        data = body.get("data") or []
        if not data:
            return []
        embedding = data[0].get("embedding") or []
        return [float(value) for value in embedding]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """本地 sentence-transformers embedding provider。

    默认使用中文友好的 `BAAI/bge-small-zh-v1.5`，不需要额外 embedding API key。
    """

    def __init__(self, model_name: str = None) -> None:
        """保存本地模型名称；模型会懒加载。"""
        self.model_name = model_name or os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self._model = None
        self.dimensions = int(os.environ.get("LOCAL_EMBEDDING_DIMENSIONS", "512"))

    def available(self) -> bool:
        """尝试加载模型，加载成功则可用。"""
        return self._load_model() is not None

    def embed(self, text: str) -> List[float]:
        """用本地模型把文本编码成归一化向量。"""
        model = self._load_model()
        if model is None:
            return []
        vector = model.encode(text[:8000], normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]

    def _load_model(self):
        """懒加载 sentence-transformers 模型，避免启动时强制下载。"""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            return None
        try:
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimensions = int(self._model.get_embedding_dimension())
            else:
                self.dimensions = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            pass
        return self._model


class LocalVectorStore:
    """本地向量库。

    它保存 chunk_id 到向量和 EvidenceChunk 的映射，支持 upsert、search、
    save、load。真实 embedding 不可用时会自动回退到哈希向量。
    """

    def __init__(self, dimensions: int = 128, embedding_provider: EmbeddingProvider = None) -> None:
        """初始化向量维度和 embedding provider。"""
        self.dimensions = dimensions
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        if self.embedding_provider.available():
            self.dimensions = self.embedding_provider.dimensions
        self.vectors: Dict[str, List[float]] = {}
        self.chunks: Dict[str, EvidenceChunk] = {}

    def upsert(self, chunks: List[EvidenceChunk]) -> None:
        """插入或更新一批 EvidenceChunk 的向量。"""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = self._embed(chunk.text)

    def search(self, query: str, limit: int = 5) -> List[EvidenceChunk]:
        """根据 query 向量相似度返回 top-k EvidenceChunk。"""
        query_vector = self._embed(query)
        scored: List[Tuple[float, EvidenceChunk]] = []
        for chunk_id, vector in self.vectors.items():
            score = cosine_similarity(query_vector, vector)
            if score > 0:
                chunk = self.chunks[chunk_id]
                scored.append((score, EvidenceChunk(chunk.chunk_id, chunk.source_url, chunk.source_title, chunk.text, round(score, 4))))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    def _embed(self, text: str) -> List[float]:
        """优先用真实 embedding；失败时回退到哈希 embedding。"""
        if self.embedding_provider.available():
            vector = self.embedding_provider.embed(text)
            if vector:
                return normalize_vector(vector)
        return embed_text(text, self.dimensions)

    def save(self, path: Path) -> Path:
        """把向量库保存成 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.dimensions,
            "vectors": self.vectors,
            "chunks": {chunk_id: asdict(chunk) for chunk_id, chunk in self.chunks.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def load(self, path: Path) -> None:
        """从 JSON 文件加载向量库。"""
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.dimensions = int(payload.get("dimensions", self.dimensions))
        self.vectors = {key: [float(value) for value in values] for key, values in payload.get("vectors", {}).items()}
        self.chunks = {
            key: EvidenceChunk(**value)
            for key, value in payload.get("chunks", {}).items()
        }


def build_vector_store() -> LocalVectorStore:
    """根据环境变量创建向量库。

    优先顺序：远程 embedding API -> 本地 sentence-transformers -> 哈希向量。
    """
    api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("EMBEDDING_MODEL", "")
    if api_key and model:
        provider = OpenAICompatibleEmbeddingProvider(
            api_key=api_key,
            base_url=os.environ.get("EMBEDDING_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            model=model,
            dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1024")),
        )
        return LocalVectorStore(embedding_provider=provider)
    if os.environ.get("DISABLE_LOCAL_EMBEDDINGS", "") != "1":
        local_provider = SentenceTransformerEmbeddingProvider()
        if local_provider.available():
            return LocalVectorStore(embedding_provider=local_provider)
    return LocalVectorStore()


def embed_text(text: str, dimensions: int = 128) -> List[float]:
    """哈希版文本向量化，用作无依赖兜底方案。"""
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dimensions
        vector[index] += 1.0
    length = math.sqrt(sum(value * value for value in vector))
    if not length:
        return vector
    return [value / length for value in vector]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """计算两个向量的余弦相似度；输入向量默认已归一化。"""
    return sum(a * b for a, b in zip(left, right))


def normalize_vector(vector: List[float]) -> List[float]:
    """把向量归一化为单位长度。"""
    length = math.sqrt(sum(value * value for value in vector))
    if not length:
        return vector
    return [value / length for value in vector]


def tokenize(text: str) -> List[str]:
    """把文本切成简单 token，供哈希 embedding 使用。"""
    normalized = text.lower()
    for char in "，。,.()[]:;!?/\"'<>-_":
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
