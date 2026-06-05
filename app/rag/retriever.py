from dataclasses import dataclass

from app.rag.embeddings import LocalEmbeddingClient
from app.rag.loader import DocumentLoader
from app.rag.reranker import KeywordReranker
from app.rag.splitter import TextChunk, TextSplitter
from app.rag.vector_store import InMemoryVectorStore, VectorDocument


@dataclass
class RetrievedChunk:
    source: str
    content: str
    score: float
    chunk_id: str = ""


class SimpleRetriever:
    """
    轻量 RAG 检索器。

    流程：
    1. loader 加载文档
    2. splitter 切分 chunk
    3. embedding client 生成向量
    4. vector store 做 top-k 召回
    5. reranker 做关键词重排
    """

    def __init__(
        self,
        knowledge_dir: str = "data/knowledge_base",
        chunk_size: int = 300,
        embedding_dimension: int = 256,
    ):
        self.loader = DocumentLoader(knowledge_dir=knowledge_dir)
        self.splitter = TextSplitter(chunk_size=chunk_size)
        self.embedding_client = LocalEmbeddingClient(
            dimension=embedding_dimension,
        )
        self.vector_store = InMemoryVectorStore()
        self.reranker = KeywordReranker()
        self.chunks: list[TextChunk] = []
        self.load_documents()

    def load_documents(self) -> None:
        self.chunks.clear()
        self.vector_store.clear()

        for document in self.loader.load():
            self.chunks.extend(
                self.splitter.split(
                    source=document.source,
                    text=document.content,
                )
            )

        embeddings = self.embedding_client.embed_documents(
            [
                chunk.content
                for chunk in self.chunks
            ]
        )

        vector_documents = [
            VectorDocument(
                chunk_id=f"{chunk.source}#{index}",
                source=chunk.source,
                content=chunk.content,
                embedding=embedding,
                metadata={"index": index},
            )
            for index, (chunk, embedding) in enumerate(
                zip(self.chunks, embeddings),
                start=1,
            )
        ]

        self.vector_store.add_documents(vector_documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        query_embedding = self.embedding_client.embed_query(query)
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=max(top_k * 4, top_k),
        )

        chunks = [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                source=item.source,
                content=item.content,
                score=item.score,
            )
            for item in candidates
        ]

        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=chunks,
        )

        return reranked_chunks[:top_k]
