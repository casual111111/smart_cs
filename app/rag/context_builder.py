from dataclasses import dataclass

from app.rag.retriever import RetrievedChunk


@dataclass
class RagContext:
    query: str
    context: str
    sources: list[str]
    chunks: list[RetrievedChunk]


class ContextBuilder:
    """
    RAG 上下文构造器。

    把 top-k 片段整理成可直接喂给 LLM 的上下文，并保留引用来源。
    """

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_chars: int = 1600,
    ) -> RagContext:
        context_parts: list[str] = []
        sources: list[str] = []
        used_chars = 0

        for index, chunk in enumerate(chunks, start=1):
            block = (
                f"[资料 {index}]\n"
                f"来源：{chunk.source}\n"
                f"相关度：{chunk.score:.2f}\n"
                f"{chunk.content}"
            )

            if used_chars + len(block) > max_chars:
                break

            context_parts.append(block)
            used_chars += len(block)

            if chunk.source not in sources:
                sources.append(chunk.source)

        return RagContext(
            query=query,
            context="\n\n".join(context_parts),
            sources=sources,
            chunks=chunks,
        )
