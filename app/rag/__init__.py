from app.rag.vector_store import InMemoryVectorStore


__all__ = [
    "ContextBuilder",
    "InMemoryVectorStore",
    "RagContext",
    "RetrievedChunk",
    "SimpleRetriever",
]


def __getattr__(name: str):
    if name in {"ContextBuilder", "RagContext"}:
        from app.rag.context_builder import ContextBuilder, RagContext

        return {
            "ContextBuilder": ContextBuilder,
            "RagContext": RagContext,
        }[name]

    if name in {"RetrievedChunk", "SimpleRetriever"}:
        from app.rag.retriever import RetrievedChunk, SimpleRetriever

        return {
            "RetrievedChunk": RetrievedChunk,
            "SimpleRetriever": SimpleRetriever,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
