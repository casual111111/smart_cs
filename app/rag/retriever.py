from dataclasses import dataclass

from app.rag.bge_embeddings import BGEEmbeddingClient
from app.rag.loader import DocumentLoader
from app.rag.mysql_chunk_store import MySQLChunkStore
from app.rag.qdrant_vector_store import QdrantVectorStore
from app.rag.reranker import KeywordReranker
from app.rag.splitter import TextChunk, TextSplitter
from app.rag.vector_store import VectorDocument


@dataclass
class RetrievedChunk:
    source: str
    content: str
    score: float
    chunk_id: str = ""


class SimpleRetriever:
    """
    Qdrant + MySQL RAG 检索器。

    当前流程：
    1. loader 加载 data/knowledge_base
    2. splitter 切 chunk
    3. BGE 生成 embedding
    4. MySQL 保存 chunk 原文
    5. Qdrant 保存 embedding
    6. 查询时 Qdrant 搜 top-k chunk_id
    7. MySQL 根据 chunk_id 查原文
    8. reranker 做关键词重排
    """

    def __init__(
        self,
        knowledge_dir: str = "data/knowledge_base",
        chunk_size: int = 300,
    ):
        self.loader = DocumentLoader(knowledge_dir=knowledge_dir)
        self.splitter = TextSplitter(chunk_size=chunk_size)

        self.embedding_client = BGEEmbeddingClient()

        self.chunk_store = MySQLChunkStore()
        self.vector_store = QdrantVectorStore()
        self.reranker = KeywordReranker()
        self.chunks: list[TextChunk] = []

        # 启动时：
        # 如果 MySQL 和 Qdrant 都有数据，就跳过导入。
        # 如果 Qdrant 是空的，就重新从文件生成 embedding 并写入 Qdrant。
        self.load_documents(rebuild=False)

    def load_documents(self, rebuild: bool = True) -> None:
        mysql_count = self.chunk_store.count()
        qdrant_count = self.vector_store.count()

        print(
            f"[RAG] load_documents rebuild={rebuild}, "
            f"mysql_count={mysql_count}, qdrant_count={qdrant_count}"
        )

        if not rebuild and mysql_count > 0 and qdrant_count > 0:
            print("[RAG] skip loading, MySQL and Qdrant already have data")
            return

        if rebuild:
            print("[RAG] rebuild=True, clearing MySQL and Qdrant")
            self.chunk_store.clear()
            self.vector_store.clear()

        self.chunks.clear()

        for document in self.loader.load():
            print(f"[RAG] loading document: {document.source}")

            self.chunks.extend(
                self.splitter.split(
                    source=document.source,
                    text=document.content,
                )
            )

        print(f"[RAG] total chunks={len(self.chunks)}")

        if not self.chunks:
            print("[RAG] no chunks found")
            return

        print("[RAG] start embedding...")

        embeddings = self.embedding_client.embed_documents(
            [
                chunk.content
                for chunk in self.chunks
            ]
        )

        print("[RAG] embedding finished")

        vector_documents = [
            VectorDocument(
                chunk_id=f"{chunk.source}#{index}",
                source=chunk.source,
                content=chunk.content,
                embedding=embedding,
                metadata={
                    "index": index,
                    "source": chunk.source,
                },
            )
            for index, (chunk, embedding) in enumerate(
                zip(self.chunks, embeddings),
                start=1,
            )
        ]

        print("[RAG] writing chunks to MySQL...")
        self.chunk_store.upsert_documents(
            documents=vector_documents,
            point_id_getter=self.vector_store.point_id,
        )

        print("[RAG] writing vectors to Qdrant...")
        self.vector_store.add_documents(vector_documents)

        print("[RAG] loading finished")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        query_embedding = self.embedding_client.embed_query(query)

        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=max(top_k * 4, top_k),
        )

        if not candidates:
            return []

        chunk_ids = [
            item.chunk_id
            for item in candidates
            if item.chunk_id
        ]

        chunk_map = self.chunk_store.get_by_chunk_ids(chunk_ids)

        chunks: list[RetrievedChunk] = []

        for item in candidates:
            chunk = chunk_map.get(item.chunk_id)

            if chunk is None:
                continue

            chunks.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    source=chunk.source,
                    content=chunk.content,
                    score=item.score,
                )
            )

        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=chunks,
        )

        return reranked_chunks[:top_k]