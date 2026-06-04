import re
from dataclasses import dataclass

from app.RAG.loader import DocumentLoader
from app.RAG.splitter import TextChunk, TextSplitter


@dataclass
class RetrievedChunk:
    source: str
    content: str
    score: float


class SimpleRetriever:
    """
    轻量 RAG 检索器。

    当前仍使用关键词打分，但加载、切分、检索已经拆成独立组件，
    后续可以把打分层替换为 embedding + vector store。
    """

    def __init__(
        self,
        knowledge_dir: str = "data/knowledge_base",
        chunk_size: int = 300,
    ):
        self.loader = DocumentLoader(knowledge_dir=knowledge_dir)
        self.splitter = TextSplitter(chunk_size=chunk_size)
        self.chunks: list[TextChunk] = []
        self.load_documents()

    def load_documents(self) -> None:
        self.chunks.clear()

        for document in self.loader.load():
            self.chunks.extend(
                self.splitter.split(
                    source=document.source,
                    text=document.content,
                )
            )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        scored_chunks: list[RetrievedChunk] = []

        for chunk in self.chunks:
            score = self._score(query, chunk.content)

            if score > 0:
                scored_chunks.append(
                    RetrievedChunk(
                        source=chunk.source,
                        content=chunk.content,
                        score=score,
                    )
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)

        return scored_chunks[:top_k]

    def _score(self, query: str, content: str) -> float:
        score = 0.0

        for token in self._extract_tokens(query):
            if token and token in content:
                score += 2.0

        business_keywords = [
            "退款",
            "退货",
            "订单号",
            "开户",
            "身份认证",
            "审核",
            "理财",
            "收益",
            "风险",
            "银行卡",
            "手机号",
        ]

        for keyword in business_keywords:
            if keyword in query and keyword in content:
                score += 3.0

        if content.startswith("#"):
            score += 0.5

        return score

    def _extract_tokens(self, text: str) -> list[str]:
        tokens = []

        english_tokens = re.findall(r"[A-Za-z0-9]+", text)
        tokens.extend(english_tokens)

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_text = "".join(chinese_chars)

        for window_size in [2, 3, 4]:
            for index in range(0, len(chinese_text) - window_size + 1):
                tokens.append(chinese_text[index : index + window_size])

        return list(set(tokens))
