from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.context.compressors import ContextCompressor
from app.context.config import ContextConfig
from app.context.packet import ContextPacket
from app.context.serializers import ContextSerializer
from app.context.token_counter import SimpleTokenCounter
from app.context.trace import ContextTrace


class SmartCSContextBuilder:
    def __init__(
        self,
        config: ContextConfig | None = None,
        token_counter: SimpleTokenCounter | None = None,
        serializer: ContextSerializer | None = None,
        compressor: ContextCompressor | None = None,
    ):
        self.config = config or ContextConfig()
        self.token_counter = token_counter or SimpleTokenCounter()
        self.serializer = serializer or ContextSerializer()
        self.compressor = compressor or ContextCompressor(
            self.config,
            self.token_counter,
        )

    def build(
        self,
        *,
        message: str,
        state: dict[str, Any] | None = None,
        agent_name: str = "router",
        system_policy: str = "",
        current_task: str = "",
        working_state: dict[str, Any] | None = None,
        base_context: str = "",
        long_context: str = "",
        rag_context: str | list[Any] | None = None,
        tool_observations: list[Any] | None = None,
        output_requirements: str = "",
    ) -> tuple[str, dict]:
        packets = self.gather(
            message=message,
            state=state or {},
            agent_name=agent_name,
            system_policy=system_policy,
            current_task=current_task,
            working_state=working_state,
            base_context=base_context,
            long_context=long_context,
            rag_context=rag_context,
            tool_observations=tool_observations,
            output_requirements=output_requirements,
        )

        selected, dropped, compression_applied = self.select(packets)
        optimized_context = self.serializer.serialize(selected)
        used_tokens = self.token_counter.count(optimized_context)

        if (
            used_tokens > self.config.input_token_budget
            and self.config.enable_compression
        ):
            selected, compressed = self.compressor.compress(selected)
            compression_applied = compression_applied or compressed
            selected, dropped_after, _ = self.select(selected)
            dropped.extend(dropped_after)
            optimized_context = self.serializer.serialize(selected)
            used_tokens = self.token_counter.count(optimized_context)

        trace = ContextTrace(
            total_candidates=len(packets),
            selected_packets=[self._packet_id(packet) for packet in selected],
            dropped_packets=[self._packet_id(packet) for packet in dropped],
            token_budget=self.config.input_token_budget,
            used_tokens=used_tokens,
            compression_applied=compression_applied,
        )

        return optimized_context, trace.to_dict()

    def gather(
        self,
        *,
        message: str,
        state: dict[str, Any],
        agent_name: str,
        system_policy: str,
        current_task: str,
        working_state: dict[str, Any] | None,
        base_context: str,
        long_context: str,
        rag_context: str | list[Any] | None,
        tool_observations: list[Any] | None,
        output_requirements: str,
    ) -> list[ContextPacket]:
        packets = [
            self._packet(
                system_policy or "You are a Smart-CS customer service agent.",
                "supervisor",
                "system_policy",
                must_keep=True,
                importance=1.0,
            ),
            self._packet(
                current_task or f"Handle the user message: {message}",
                agent_name,
                "current_task",
                must_keep=True,
                importance=1.0,
            ),
            self._packet(
                self._format_working_memory(
                    working_state if working_state is not None else state,
                    agent_name,
                ),
                "working_memory",
                "working_memory",
                must_keep=True,
                importance=1.0,
            ),
        ]

        if long_context:
            packets.append(
                self._packet(long_context, "long_term_memory", "long_term_memory")
            )

        for index, item in enumerate(self._split_context(base_context)):
            packets.append(
                self._packet(
                    item,
                    f"base_context:{index}",
                    "short_term_memory",
                    relevance_score=0.75,
                    importance=0.6,
                )
            )

        for index, item in enumerate(self._normalize_items(rag_context)):
            packets.append(
                self._packet(
                    item,
                    f"rag:{index}",
                    "rag_evidence",
                    relevance_score=0.85,
                    importance=0.75,
                )
            )

        for index, item in enumerate(tool_observations or []):
            packets.append(
                self._packet(
                    self._to_text(item),
                    f"tool:{index}",
                    "tool_observation",
                    relevance_score=0.7,
                    importance=0.65,
                )
            )

        packets.append(
            self._packet(
                output_requirements
                or "Answer clearly, politely, and only with supported facts.",
                "supervisor",
                "output_requirements",
                must_keep=True,
                importance=0.9,
            )
        )

        return [packet for packet in packets if packet.content]

    def select(
        self,
        packets: list[ContextPacket],
    ) -> tuple[list[ContextPacket], list[ContextPacket], bool]:
        budget = self.config.input_token_budget
        must_keep = [packet for packet in packets if packet.must_keep]
        optional = [packet for packet in packets if not packet.must_keep]

        selected = list(must_keep)
        dropped = []
        used = sum(packet.token_count for packet in selected)
        compression_applied = False

        optional.sort(key=self._score, reverse=True)

        for packet in optional:
            if packet.relevance_score < self.config.min_relevance:
                dropped.append(packet)
                continue

            candidate = packet
            if (
                used + candidate.token_count > budget
                and self.config.enable_compression
            ):
                candidate, changed = self.compressor.compress_packet(packet)
                compression_applied = compression_applied or changed

            if used + candidate.token_count <= budget:
                selected.append(candidate)
                used += candidate.token_count
            else:
                dropped.append(packet)

        return selected, dropped, compression_applied

    def _packet(
        self,
        content: str,
        source: str,
        packet_type: str,
        relevance_score: float = 1.0,
        importance: float = 0.5,
        must_keep: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ContextPacket:
        return ContextPacket(
            content=content,
            source=source,
            packet_type=packet_type,
            token_count=self.token_counter.count(content),
            relevance_score=relevance_score,
            importance=importance,
            metadata=metadata or {},
            must_keep=must_keep,
        )

    def _score(self, packet: ContextPacket) -> float:
        age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - packet.timestamp).total_seconds(),
        )
        recency = 1.0 / (1.0 + age_seconds / 3600.0)
        return (
            packet.relevance_score * self.config.relevance_weight
            + recency * self.config.recency_weight
            + packet.importance * self.config.importance_weight
        )

    def _format_working_memory(
        self,
        state: dict[str, Any],
        agent_name: str,
    ) -> str:
        allowed_by_agent = {
            "router": [
                "trace_id",
                "session_id",
                "intent",
                "intent_confidence",
                "retry_count",
            ],
            "ticket": [
                "trace_id",
                "session_id",
                "intent",
                "intent_confidence",
                "retry_count",
                "need_human_review",
                "review_reason",
            ],
            "knowledge": [
                "trace_id",
                "session_id",
                "intent",
                "current_agent",
            ],
            "compliance": [
                "final_response",
                "raw_response",
                "intent",
                "need_human_review",
            ],
        }
        fields = allowed_by_agent.get(agent_name, allowed_by_agent["router"])
        filtered = {
            key: state.get(key)
            for key in fields
            if key in state and state.get(key) is not None
        }
        return json.dumps(filtered, ensure_ascii=False, sort_keys=True)

    def _split_context(self, context: str) -> list[str]:
        if not context:
            return []

        parts = [part.strip() for part in context.split("\n\n") if part.strip()]
        if len(parts) <= self.config.recent_message_limit:
            return parts

        return parts[-self.config.recent_message_limit :]

    def _normalize_items(self, value: str | list[Any] | None) -> list[str]:
        if not value:
            return []

        if isinstance(value, str):
            return [value]

        return [self._to_text(item) for item in value if item]

    def _to_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value

        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def _packet_id(self, packet: ContextPacket) -> str:
        return f"{packet.packet_type}:{packet.source}"
