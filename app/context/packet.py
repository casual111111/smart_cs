from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ContextPacket:
    content: str
    source: str
    packet_type: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    token_count: int = 0
    relevance_score: float = 1.0
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    must_keep: bool = False

    def with_content(self, content: str, token_count: int) -> "ContextPacket":
        return ContextPacket(
            content=content,
            source=self.source,
            packet_type=self.packet_type,
            timestamp=self.timestamp,
            token_count=token_count,
            relevance_score=self.relevance_score,
            importance=self.importance,
            metadata=dict(self.metadata),
            must_keep=self.must_keep,
        )
