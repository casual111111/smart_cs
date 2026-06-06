import math
from typing import Any

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import KnowledgeChunk
from app.rag.vector_store import VectorDocument, VectorSearchResult


class MySQLVectorStore:
    """
    MySQL 向量库。

    当前阶段：
    - MySQL 保存 chunk、content、embedding、metadata
    - Python 读取 MySQL 里的 embedding
    - Python 计算余弦相似度

    这个类保持和 InMemoryVectorStore 接近的接口：
    - clear()
    - add_documents()
    - search()
    """

    def __init__(self):
        pass

    def _get_session(self) -> Session:
        return SessionLocal()

    def count(self) -> int:
        db = self._get_session()
        try:
            return db.query(KnowledgeChunk).count()
        finally:
            db.close()

    def clear(self) -> None:
        db = self._get_session()
        try:
            db.query(KnowledgeChunk).delete()
            db.commit()
        finally:
            db.close()

    def add_documents(self, documents: list[VectorDocument]) -> None:
        """
        批量写入 MySQL。

        如果 chunk_id 已存在，就更新；
        如果 chunk_id 不存在，就插入。
        """
        db = self._get_session()

        try:
            for document in documents:
                chunk = (
                    db.query(KnowledgeChunk)
                    .filter(KnowledgeChunk.chunk_id == document.chunk_id)
                    .first()
                )

                if chunk is None:
                    chunk = KnowledgeChunk(
                        chunk_id=document.chunk_id,
                        source=document.source,
                        content=document.content,
                        embedding=document.embedding,
                        metadata_json=document.metadata,
                    )
                    db.add(chunk)
                else:
                    chunk.source = document.source
                    chunk.content = document.content
                    chunk.embedding = document.embedding
                    chunk.metadata_json = document.metadata

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        source: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        从 MySQL 中取出 chunk，然后在 Python 中计算余弦相似度。
        """
        db = self._get_session()

        try:
            query = db.query(KnowledgeChunk)

            if source is not None:
                query = query.filter(KnowledgeChunk.source == source)

            results: list[VectorSearchResult] = []

            for chunk in query.yield_per(1000):
                score = self._cosine_similarity(
                    query_embedding,
                    chunk.embedding,
                )

                if score <= 0:
                    continue

                results.append(
                    VectorSearchResult(
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        content=chunk.content,
                        score=score,
                        metadata=chunk.metadata_json or {},
                    )
                )

            results.sort(key=lambda item: item.score, reverse=True)

            return results[:top_k]

        finally:
            db.close()

    def _cosine_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        if not left or not right:
            return 0.0

        if len(left) != len(right):
            return 0.0

        dot = sum(
            left[index] * right[index]
            for index in range(len(left))
        )

        left_norm = math.sqrt(
            sum(value * value for value in left)
        )

        right_norm = math.sqrt(
            sum(value * value for value in right)
        )

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)