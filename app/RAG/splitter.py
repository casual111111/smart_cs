import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    source: str
    content: str


class TextSplitter:
    """
    简单文本切分器。

    先按段落切分，过长段落再按固定长度切块。
    """

    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def split(self, source: str, text: str) -> list[TextChunk]:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

        paragraphs = self._merge_headings(paragraphs)

        chunks: list[TextChunk] = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(TextChunk(source=source, content=paragraph))
                continue

            for start in range(0, len(paragraph), self.chunk_size):
                chunks.append(
                    TextChunk(
                        source=source,
                        content=paragraph[start : start + self.chunk_size],
                    )
                )

        return chunks

    def _merge_headings(self, paragraphs: list[str]) -> list[str]:
        merged: list[str] = []
        index = 0

        while index < len(paragraphs):
            paragraph = paragraphs[index]

            if paragraph.startswith("#") and index + 1 < len(paragraphs):
                merged.append(f"{paragraph}\n{paragraphs[index + 1]}")
                index += 2
                continue

            merged.append(paragraph)
            index += 1

        return merged
