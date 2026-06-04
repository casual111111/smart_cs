import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorDocument:
    chunk_id: str
    source: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    chunk_id: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """
    内存向量库。

    当前用于本地 RAG MVP；后续可替换为 FAISS / Chroma / Milvus。
    """

    def __init__(self):
        self.documents: list[VectorDocument] = []

    def clear(self) -> None:
        self.documents.clear()

    def add_documents(self, documents: list[VectorDocument]) -> None:
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[VectorSearchResult]:
        results: list[VectorSearchResult] = []

        for document in self.documents:
            score = self._cosine_similarity(
                query_embedding,
                document.embedding,
            )

            if score <= 0:
                continue

            results.append(
                VectorSearchResult(
                    chunk_id=document.chunk_id,
                    source=document.source,
                    content=document.content,
                    score=score,
                    metadata=document.metadata,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)

        return results[:top_k]

    def _cosine_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        if not left or not right:
            return 0.0

        size = min(len(left), len(right))
        dot = sum(left[index] * right[index] for index in range(size))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)
