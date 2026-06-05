from app.rag.context_builder import ContextBuilder
from app.rag.embeddings import LocalEmbeddingClient
from app.rag.retriever import SimpleRetriever
from app.rag.vector_store import InMemoryVectorStore, VectorDocument
from app.tools.knowledge_tool import KnowledgeTool


def test_local_embedding_is_deterministic():
    client = LocalEmbeddingClient(dimension=32)

    first = client.embed_query("退款多久到账")
    second = client.embed_query("退款多久到账")

    assert first == second
    assert len(first) == 32


def test_vector_store_returns_most_similar_document():
    client = LocalEmbeddingClient()
    store = InMemoryVectorStore()

    store.add_documents(
        [
            VectorDocument(
                chunk_id="refund#1",
                source="refund.md",
                content="退款一般会在 3-5 个工作日内处理。",
                embedding=client.embed_query("退款 3-5 个工作日"),
            ),
            VectorDocument(
                chunk_id="account#1",
                source="account.md",
                content="开户需要完成身份认证和银行卡绑定。",
                embedding=client.embed_query("开户 身份认证 银行卡"),
            ),
        ]
    )

    results = store.search(
        query_embedding=client.embed_query("退款多久到账"),
        top_k=1,
    )

    assert results[0].source == "refund.md"


def test_retriever_loads_documents_and_reranks_results(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    (knowledge_dir / "refund.md").write_text(
        "# 退款规则\n\n退款一般会在 3-5 个工作日内处理，请关注订单状态。",
        encoding="utf-8",
    )
    (knowledge_dir / "account.md").write_text(
        "# 开户流程\n\n开户需要完成身份认证，并绑定本人银行卡。",
        encoding="utf-8",
    )

    retriever = SimpleRetriever(
        knowledge_dir=str(knowledge_dir),
        chunk_size=200,
        embedding_dimension=64,
    )

    results = retriever.retrieve("退款多久到账？", top_k=1)

    assert results
    assert results[0].source == "refund.md"
    assert "3-5 个工作日" in results[0].content
    assert results[0].chunk_id.startswith("refund.md#")


def test_retriever_reload_documents(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    (knowledge_dir / "refund.md").write_text(
        "退款一般会在 3-5 个工作日内处理。",
        encoding="utf-8",
    )

    retriever = SimpleRetriever(
        knowledge_dir=str(knowledge_dir),
        embedding_dimension=64,
    )

    assert retriever.retrieve("退款多久到账", top_k=1)

    (knowledge_dir / "complaint.txt").write_text(
        "投诉问题会创建高优先级工单，并转交人工客服处理。",
        encoding="utf-8",
    )

    retriever.load_documents()
    results = retriever.retrieve("我要投诉，需要人工处理", top_k=1)

    assert results[0].source == "complaint.txt"


def test_context_builder_keeps_sources_and_limits_context(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    (knowledge_dir / "refund.md").write_text(
        "# 退款规则\n\n退款一般会在 3-5 个工作日内处理。",
        encoding="utf-8",
    )

    retriever = SimpleRetriever(
        knowledge_dir=str(knowledge_dir),
        embedding_dimension=64,
    )
    chunks = retriever.retrieve("退款多久到账？", top_k=3)
    context = ContextBuilder().build(
        query="退款多久到账？",
        chunks=chunks,
        max_chars=500,
    )

    assert context.sources == ["refund.md"]
    assert "来源：refund.md" in context.context
    assert "退款" in context.context


def test_knowledge_tool_builds_rag_context_with_sources(tmp_path):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    (knowledge_dir / "refund.md").write_text(
        "# 退款规则\n\n退款一般会在 3-5 个工作日内处理。",
        encoding="utf-8",
    )

    tool = KnowledgeTool()
    tool.retriever = SimpleRetriever(
        knowledge_dir=str(knowledge_dir),
        embedding_dimension=64,
    )

    context = tool.build_rag_context(
        query="退款多久到账？",
        top_k=3,
    )

    assert context.sources == ["refund.md"]
    assert context.chunks[0].chunk_id.startswith("refund.md#")
    assert "来源：refund.md" in context.context
