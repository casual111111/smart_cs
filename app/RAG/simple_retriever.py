import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievedChunk:
    source: str
    content: str
    score: float


class SimpleRetriever:
    """
    简化版本地知识库检索器。

    当前版本不使用向量数据库，只做规则打分：
    1. 加载 data/knowledge_base/*.md
    2. 按段落切分
    3. 根据关键词重合度打分
    4. 返回 top_k 个最相关片段

    后面可以升级为：
    FAISS / Chroma / Milvus + Embedding。
    """

    def __init__(
        self,
        knowledge_dir: str = "data/knowledge_base",
        chunk_size: int = 300,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.chunk_size = chunk_size
        self.chunks: list[dict] = []
        self.load_documents()

    def load_documents(self) -> None:
        self.chunks.clear()

        if not self.knowledge_dir.exists():
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in self.knowledge_dir.glob("*.md"):
            text = file_path.read_text(encoding="utf-8")
            chunks = self._split_text(text)

            for chunk in chunks:
                self.chunks.append(
                    {
                        "source": file_path.name,
                        "content": chunk,
                    }
                )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        scored_chunks = []

        for chunk in self.chunks:
            score = self._score(query, chunk["content"])

            if score > 0:
                scored_chunks.append(
                    RetrievedChunk(
                        source=chunk["source"],
                        content=chunk["content"],
                        score=score,
                    )
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)

        return scored_chunks[:top_k]

    def _split_text(self, text: str) -> list[str]:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
            else:
                for i in range(0, len(paragraph), self.chunk_size):
                    chunks.append(paragraph[i : i + self.chunk_size])

        return chunks

    def _score(self, query: str, content: str) -> float:
        score = 0.0

        # 1. 完整问题中关键词直接出现，加分
        for token in self._extract_tokens(query):
            if token and token in content:
                score += 2.0

        # 2. 常见业务词命中，加额外分
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

        # 3. 标题命中加分
        if content.startswith("#"):
            score += 0.5

        return score

    def _extract_tokens(self, text: str) -> list[str]:
        """
        简单分词：
        - 英文、数字按词提取
        - 中文按 2~4 字滑动窗口提取
        """
        tokens = []

        english_tokens = re.findall(r"[A-Za-z0-9]+", text)
        tokens.extend(english_tokens)

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_text = "".join(chinese_chars)

        for window_size in [2, 3, 4]:
            for i in range(0, len(chinese_text) - window_size + 1):
                tokens.append(chinese_text[i : i + window_size])

        return list(set(tokens))