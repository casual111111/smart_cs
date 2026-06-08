from app.context.config import ContextConfig
from app.context.packet import ContextPacket
from app.context.token_counter import SimpleTokenCounter


NEVER_COMPRESS_TYPES = {
    "system_policy",
    "current_task",
    "working_memory",
    "critical_state",
}

COMPRESSIBLE_TYPES = {
    "short_term_memory",
    "long_term_memory",
    "rag_evidence",
    "tool_observation",
}


class ContextCompressor:
    def __init__(
        self,
        config: ContextConfig | None = None,
        token_counter: SimpleTokenCounter | None = None,
    ):
        self.config = config or ContextConfig()
        self.token_counter = token_counter or SimpleTokenCounter()

    def compress_packet(self, packet: ContextPacket) -> tuple[ContextPacket, bool]:
        if packet.packet_type in NEVER_COMPRESS_TYPES or packet.must_keep:
            return packet, False

        if packet.packet_type not in COMPRESSIBLE_TYPES:
            return packet, False

        max_tokens = self._max_tokens_for(packet)
        if packet.token_count <= max_tokens:
            return packet, False

        content = self.token_counter.truncate(packet.content, max_tokens)
        if content and content != packet.content:
            content = content.rstrip() + "\n...[truncated]"

        return packet.with_content(
            content=content,
            token_count=self.token_counter.count(content),
        ), True

    def compress(self, packets: list[ContextPacket]) -> tuple[list[ContextPacket], bool]:
        compressed = []
        changed = False

        for packet in packets:
            new_packet, packet_changed = self.compress_packet(packet)
            compressed.append(new_packet)
            changed = changed or packet_changed

        return compressed, changed

    def _max_tokens_for(self, packet: ContextPacket) -> int:
        if packet.packet_type == "rag_evidence":
            return self.config.max_rag_chunk_tokens

        if packet.packet_type == "tool_observation":
            return self.config.max_tool_observation_tokens

        return max(80, self.config.input_token_budget // 6)
