import hashlib
import math
import re


class LocalEmbeddingClient:
    """
    本地 deterministic embedding 客户端。

    这个实现不依赖外部模型，适合本地开发和回归测试。
    后续可以用同样接口替换为 OpenAI / BGE / Qwen embedding。
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension

        for token in self._extract_tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [
            value / norm
            for value in vector
        ]

    def _extract_tokens(self, text: str) -> list[str]:
        tokens: list[str] = []

        tokens.extend(
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", text)
        )

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_text = "".join(chinese_chars)

        for window_size in [1, 2, 3, 4]:
            for index in range(0, len(chinese_text) - window_size + 1):
                tokens.append(chinese_text[index : index + window_size])

        return list(set(tokens))
