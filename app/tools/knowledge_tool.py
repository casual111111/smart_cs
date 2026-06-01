from app.RAG.simple_retriever import RetrievedChunk, SimpleRetriever


class KnowledgeTool:
    """
    知识库检索工具。

    Agent 不直接操作 Retriever，
    而是通过 KnowledgeTool 检索知识。
    """

    def __init__(self):
        self.retriever = SimpleRetriever()

    def search_knowledge(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k=top_k)

    def reload_knowledge_base(self) -> None:
        self.retriever.load_documents()