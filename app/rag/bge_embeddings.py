from __future__ import annotations

from typing import List

from FlagEmbedding import BGEM3FlagModel


class BGEEmbeddingClient:
    """
    BGE-M3 embedding client.

    用真正的深度学习 embedding encoder 替代原来的 LocalEmbeddingClient。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        batch_size: int = 8,
        max_length: int = 8192,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
        )

    def embed_query(self, text: str) -> List[float]:
        normalized_text = text.strip() if text and text.strip() else " "

        result = self.model.encode(
            [normalized_text],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return result["dense_vecs"][0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = [
            text.strip() if text and text.strip() else " "
            for text in texts
        ]

        if not normalized_texts:
            return []

        result = self.model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return [vector.tolist() for vector in result["dense_vecs"]]