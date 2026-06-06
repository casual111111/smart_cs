from collections.abc import Callable

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import KnowledgeChunk
from app.rag.vector_store import VectorDocument


class MySQLChunkStore:
    """
    MySQL chunk 原文仓库。

    只负责保存：
    - chunk_id
    - source
    - content
    - metadata
    - qdrant_point_id

    不负责向量检索。
    """

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
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def upsert_documents(
        self,
        documents: list[VectorDocument],
        point_id_getter: Callable[[str], str],
    ) -> None:
        if not documents:
            return

        db = self._get_session()

        try:
            chunk_ids = [doc.chunk_id for doc in documents]

            existing_chunks = (
                db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.chunk_id.in_(chunk_ids))
                .all()
            )

            existing_map = {
                chunk.chunk_id: chunk
                for chunk in existing_chunks
            }

            for document in documents:
                point_id = point_id_getter(document.chunk_id)

                chunk = existing_map.get(document.chunk_id)

                if chunk is None:
                    chunk = KnowledgeChunk(
                        chunk_id=document.chunk_id,
                        source=document.source,
                        content=document.content,
                        qdrant_point_id=point_id,
                        metadata_json=document.metadata,
                    )
                    db.add(chunk)
                else:
                    chunk.source = document.source
                    chunk.content = document.content
                    chunk.qdrant_point_id = point_id
                    chunk.metadata_json = document.metadata

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def get_by_chunk_ids(
        self,
        chunk_ids: list[str],
    ) -> dict[str, KnowledgeChunk]:
        if not chunk_ids:
            return {}

        db = self._get_session()

        try:
            chunks = (
                db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.chunk_id.in_(chunk_ids))
                .all()
            )

            return {
                chunk.chunk_id: chunk
                for chunk in chunks
            }

        finally:
            db.close()