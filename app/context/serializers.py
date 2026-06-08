from collections import defaultdict

from app.context.packet import ContextPacket


SECTION_ORDER = [
    ("Role & Policies", {"system_policy"}),
    ("Task", {"current_task"}),
    ("State", {"working_memory", "critical_state"}),
    ("User Profile", {"long_term_memory", "user_profile"}),
    ("Conversation Summary", {"conversation_summary", "short_term_memory"}),
    ("Recent Messages", {"recent_message"}),
    ("Evidence", {"rag_evidence"}),
    ("Tool Observations", {"tool_observation"}),
    ("Compliance Policy", {"compliance_policy"}),
    ("Output Requirements", {"output_requirements"}),
]


class ContextSerializer:
    def serialize(self, packets: list[ContextPacket]) -> str:
        grouped: dict[str, list[ContextPacket]] = defaultdict(list)

        for packet in packets:
            grouped[packet.packet_type].append(packet)

        sections = []

        for title, packet_types in SECTION_ORDER:
            body_parts = []
            for packet_type in packet_types:
                for packet in grouped.get(packet_type, []):
                    if packet.content:
                        body_parts.append(packet.content.strip())

            body = "\n\n".join(body_parts).strip() or "(none)"
            sections.append(f"[{title}]\n{body}")

        return "\n\n".join(sections)
