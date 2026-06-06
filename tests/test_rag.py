from app.rag.context_builder import ContextBuilder
from app.rag.retriever import RetrievedChunk
from app.rag.vector_store import InMemoryVectorStore, VectorDocument
from app.tools.knowledge_tool import KnowledgeTool


def test_vector_store_returns_most_similar_document():
    store = InMemoryVectorStore()

    store.add_documents(
        [
            VectorDocument(
                chunk_id="refund.md#1",
                source="refund.md",
                content="退款一般会在 3-5 个工作日内处理。",
                embedding=[1.0, 0.0, 0.0],
            ),
            VectorDocument(
                chunk_id="account.md#1",
                source="account.md",
                content="登录账号需要手机号和验证码。",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert results[0].source == "refund.md"
    assert results[0].chunk_id == "refund.md#1"


def test_context_builder_keeps_sources_and_limits_context():
    chunks = [
        RetrievedChunk(
            chunk_id="refund.md#1",
            source="refund.md",
            content="退款一般会在 3-5 个工作日内处理。",
            score=0.95,
        ),
        RetrievedChunk(
            chunk_id="account.md#1",
            source="account.md",
            content="登录账号需要手机号和验证码。",
            score=0.5,
        ),
    ]

    context = ContextBuilder().build(
        query="退款多久到账？",
        chunks=chunks,
        max_chars=500,
    )

    assert context.sources == ["refund.md", "account.md"]
    assert context.chunks == chunks
    assert "refund.md" in context.context
    assert "3-5 个工作日" in context.context


class FakeRetriever:
    def retrieve(self, query, top_k=3):
        return [
            RetrievedChunk(
                chunk_id="refund.md#1",
                source="refund.md",
                content="退款一般会在 3-5 个工作日内处理。",
                score=0.95,
            )
        ]


def test_knowledge_tool_builds_rag_context_with_sources():
    tool = KnowledgeTool.__new__(KnowledgeTool)
    tool.retriever = FakeRetriever()
    tool.context_builder = ContextBuilder()

    context = tool.build_rag_context(
        query="退款多久到账？",
        top_k=3,
    )

    assert context.sources == ["refund.md"]
    assert context.chunks[0].chunk_id == "refund.md#1"
    assert "3-5 个工作日" in context.context
