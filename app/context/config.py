from dataclasses import dataclass


@dataclass
class ContextConfig:
    max_tokens: int = 3200
    reserve_output_tokens: int = 600
    min_relevance: float = 0.15
    relevance_weight: float = 0.55
    recency_weight: float = 0.25
    importance_weight: float = 0.20
    recent_message_limit: int = 8
    rag_top_k: int = 5
    max_rag_chunk_tokens: int = 360
    max_tool_observation_tokens: int = 300
    enable_compression: bool = True

    @property
    def input_token_budget(self) -> int:
        return max(1, self.max_tokens - self.reserve_output_tokens)
