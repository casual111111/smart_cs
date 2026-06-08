import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.rag.vector_store import VectorDocument, VectorSearchResult


class QdrantVectorStore:
    """
    Qdrant 向量库。

    只负责：
    - 保存 embedding
    - 根据 query_embedding 搜索 top-k chunk_id
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION",
            "smart_cs_chunks",
        )
        self.client = QdrantClient(url=self.url)
        self.available = True

    def point_id(self, chunk_id: str) -> str:
        """
        Qdrant 的 point id 建议使用 UUID。

        chunk_id 可能是 ecommerce_order.md#1 这种字符串，
        所以这里把它稳定转换成 UUID。
        """
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"smart_cs:{chunk_id}",
            )
        )

    def _collection_exists(self) -> bool:
        if not self.available:
            return False

        try:
            collections = self.client.get_collections().collections
        except Exception as exc:
            print(f"[RAG] Qdrant unavailable: {exc}")
            self.available = False
            return False

        collection_names = {
            collection.name
            for collection in collections
        }
        return self.collection_name in collection_names

    def create_collection_if_not_exists(
        self,
        vector_size: int,
    ) -> None:
        if self._collection_exists():
            return

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size or 768,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as exc:
            print(f"[RAG] Qdrant create_collection skipped: {exc}")
            self.available = False

    def count(self) -> int:
        if not self._collection_exists():
            return 0

        try:
            result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
        except Exception as exc:
            print(f"[RAG] Qdrant count skipped: {exc}")
            self.available = False
            return 0

        return result.count

    def clear(self) -> None:
        if self._collection_exists():
            try:
                self.client.delete_collection(
                    collection_name=self.collection_name,
                )
            except Exception as exc:
                print(f"[RAG] Qdrant clear skipped: {exc}")
                self.available = False

    def add_documents(
        self,
        documents: list[VectorDocument],
        batch_size: int = 128,
    ) -> None:
        if not documents:
            return

        if not self.available:
            return

        vector_size = len(documents[0].embedding)
        self.create_collection_if_not_exists(vector_size=vector_size)

        if not self.available:
            return

        points: list[PointStruct] = []

        for document in documents:
            points.append(
                PointStruct(
                    id=self.point_id(document.chunk_id),
                    vector=document.embedding,
                    payload={
                        "chunk_id": document.chunk_id,
                        "source": document.source,
                        "metadata": document.metadata,
                    },
                )
            )

            if len(points) >= batch_size:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                except Exception as exc:
                    print(f"[RAG] Qdrant upsert skipped: {exc}")
                    self.available = False
                    return
                points.clear()

        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
            except Exception as exc:
                print(f"[RAG] Qdrant upsert skipped: {exc}")
                self.available = False

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        source: str | None = None,
    ) -> list[VectorSearchResult]:
        if not self._collection_exists():
            return []

        conditions = []

        if source is not None:
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            )

        query_filter = None

        if conditions:
            query_filter = Filter(must=conditions)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
            )
        except Exception as exc:
            print(f"[RAG] Qdrant search skipped: {exc}")
            self.available = False
            return []

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                VectorSearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    source=payload.get("source", ""),
                    content="",
                    score=point.score or 0.0,
                    metadata=payload.get("metadata") or {},
                )
            )

        return results
