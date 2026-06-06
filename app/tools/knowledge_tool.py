from typing import TYPE_CHECKING

from app.rag.context_builder import ContextBuilder, RagContext

if TYPE_CHECKING:
    from app.rag.retriever import RetrievedChunk


class KnowledgeTool:
    """
    知识库检索工具。

    Agent 不直接操作 Retriever，
    而是通过 KnowledgeTool 检索知识。
    """

    def __init__(self):
        from app.rag.retriever import SimpleRetriever

        self.retriever = SimpleRetriever()
        self.context_builder = ContextBuilder()

    def search_knowledge(
        self,
        query: str,
        top_k: int = 3,
    ) -> list["RetrievedChunk"]:
        return self.retriever.retrieve(query, top_k=top_k)

    def build_rag_context(
        self,
        query: str,
        top_k: int = 3,
        max_chars: int = 1600,
    ) -> RagContext:
        chunks = self.search_knowledge(query=query, top_k=top_k)
        return self.context_builder.build(
            query=query,
            chunks=chunks,
            max_chars=max_chars,
        )

    def reload_knowledge_base(self) -> None:
        self.retriever.load_documents()
