from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    source: str
    content: str


class DocumentLoader:
    """
    本地知识库文档加载器。

    当前支持 Markdown 和纯文本文件。
    """

    def __init__(self, knowledge_dir: str = "data/knowledge_base"):
        self.knowledge_dir = Path(knowledge_dir)

    def load(self) -> list[LoadedDocument]:
        if not self.knowledge_dir.exists():
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            return []

        documents: list[LoadedDocument] = []

        for pattern in ["*.md", "*.txt"]:
            for file_path in self.knowledge_dir.glob(pattern):
                documents.append(
                    LoadedDocument(
                        source=file_path.name,
                        content=file_path.read_text(encoding="utf-8"),
                    )
                )

        return documents
