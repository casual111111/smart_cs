import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.rag.loader import DocumentLoader
from app.rag.mysql_chunk_store import MySQLChunkStore
from app.rag.reranker import KeywordReranker
from app.rag.splitter import TextSplitter
from app.rag.vector_store import VectorDocument


class FakeEmbeddingClient:
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        if "退款" in text or "3-5" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "账号" in text or "登录" in text:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


@pytest.fixture
def test_session_local(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'rag_test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    monkeypatch.setattr(
        "app.rag.mysql_chunk_store.SessionLocal",
        session_local,
    )

    return session_local


@pytest.fixture
def qdrant_store():
    qdrant_client = pytest.importorskip("qdrant_client")
    qdrant_vector_store = pytest.importorskip("app.rag.qdrant_vector_store")

    QdrantClient = qdrant_client.QdrantClient
    QdrantVectorStore = qdrant_vector_store.QdrantVectorStore

    client = QdrantClient(url="http://localhost:6333")

    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant is not available at http://localhost:6333: {exc}")

    store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name=f"smart_cs_test_{uuid.uuid4().hex}",
    )

    yield store

    store.clear()


def test_mysql_chunk_store_can_write_and_read_chunks(test_session_local):
    store = MySQLChunkStore()
    document = VectorDocument(
        chunk_id="refund.md#1",
        source="refund.md",
        content="退款一般会在 3-5 个工作日内处理。",
        embedding=[1.0, 0.0, 0.0, 0.0],
        metadata={"source": "refund.md", "index": 1},
    )

    store.upsert_documents(
        documents=[document],
        point_id_getter=lambda chunk_id: f"point-{chunk_id}",
    )

    chunks = store.get_by_chunk_ids(["refund.md#1"])

    assert store.count() == 1
    assert chunks["refund.md#1"].source == "refund.md"
    assert chunks["refund.md#1"].content == "退款一般会在 3-5 个工作日内处理。"
    assert chunks["refund.md#1"].metadata_json == {"source": "refund.md", "index": 1}


@pytest.mark.integration
def test_qdrant_vector_store_can_write_and_search_vectors(qdrant_store):
    qdrant_store.add_documents(
        [
            VectorDocument(
                chunk_id="refund.md#1",
                source="refund.md",
                content="退款一般会在 3-5 个工作日内处理。",
                embedding=[1.0, 0.0, 0.0, 0.0],
            ),
            VectorDocument(
                chunk_id="account.md#1",
                source="account.md",
                content="登录失败时请重置账号密码。",
                embedding=[0.0, 1.0, 0.0, 0.0],
            ),
        ]
    )

    results = qdrant_store.search(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=1,
    )

    assert results
    assert results[0].chunk_id == "refund.md#1"
    assert results[0].source == "refund.md"


@pytest.mark.integration
def test_qdrant_rag_retrieve_refund_policy(
    tmp_path,
    test_session_local,
    qdrant_store,
):
    knowledge_dir = tmp_path / "knowledge_base"
    knowledge_dir.mkdir()

    (knowledge_dir / "refund.md").write_text(
        "# 退款规则\n\n退款一般会在 3-5 个工作日内处理，请关注订单状态。",
        encoding="utf-8",
    )
    (knowledge_dir / "account.md").write_text(
        "# 账号登录\n\n如果无法登录账号，请先检查手机号和验证码。",
        encoding="utf-8",
    )

    retriever_module = pytest.importorskip("app.rag.retriever")
    SimpleRetriever = retriever_module.SimpleRetriever

    retriever = SimpleRetriever.__new__(SimpleRetriever)
    retriever.loader = DocumentLoader(knowledge_dir=str(knowledge_dir))
    retriever.splitter = TextSplitter(chunk_size=200)
    retriever.embedding_client = FakeEmbeddingClient()
    retriever.chunk_store = MySQLChunkStore()
    retriever.vector_store = qdrant_store
    retriever.reranker = KeywordReranker()
    retriever.chunks = []

    retriever.load_documents(rebuild=True)
    results = retriever.retrieve("退款多久到账", top_k=1)

    assert results
    assert results[0].source == "refund.md"
    assert "3-5 个工作日" in results[0].content
