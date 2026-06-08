from dataclasses import dataclass


@dataclass
class ContextTrace:
    total_candidates: int
    selected_packets: list[str]
    dropped_packets: list[str]
    token_budget: int
    used_tokens: int
    compression_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "selected_packets": self.selected_packets,
            "dropped_packets": self.dropped_packets,
            "token_budget": self.token_budget,
            "used_tokens": self.used_tokens,
            "compression_applied": self.compression_applied,
        }
