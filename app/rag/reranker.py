import re
from dataclasses import replace
from typing import Any


class KeywordReranker:
    """
    轻量重排序器。

    用关键词覆盖度对向量召回结果做二次排序，避免纯 hash embedding
    在中文短文本上偶发召回不稳定。
    """

    def rerank(
        self,
        query: str,
        chunks: list[Any],
    ) -> list[Any]:
        reranked = [
            replace(
                chunk,
                score=round(
                    chunk.score + self._keyword_score(query, chunk.content),
                    4,
                ),
            )
            for chunk in chunks
        ]

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _keyword_score(self, query: str, content: str) -> float:
        score = 0.0

        for token in self._extract_tokens(query):
            if token in content:
                score += 0.2

        business_keywords = [
            "退款",
            "退货",
            "订单",
            "开户",
            "身份认证",
            "审核",
            "理财",
            "收益",
            "风险",
            "银行卡",
            "手机号",
            "人工",
            "投诉",
        ]

        for keyword in business_keywords:
            if keyword in query and keyword in content:
                score += 0.5

        return score

    def _extract_tokens(self, text: str) -> list[str]:
        tokens = []

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
