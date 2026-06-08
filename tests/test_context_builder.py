from app.context import ContextConfig, SmartCSContextBuilder


def build_small_builder() -> SmartCSContextBuilder:
    return SmartCSContextBuilder(
        ContextConfig(
            max_tokens=120,
            reserve_output_tokens=20,
            min_relevance=0.3,
            max_rag_chunk_tokens=20,
            max_tool_observation_tokens=20,
        )
    )


def test_system_policy_is_always_kept():
    context, trace = build_small_builder().build(
        message="hello",
        system_policy="SYSTEM_POLICY_MUST_KEEP",
        current_task="task",
        base_context="x" * 1000,
    )

    assert "SYSTEM_POLICY_MUST_KEEP" in context
    assert any("system_policy" in item for item in trace["selected_packets"])


def test_current_task_is_always_kept():
    context, trace = build_small_builder().build(
        message="hello",
        system_policy="policy",
        current_task="CURRENT_TASK_MUST_KEEP",
        base_context="x" * 1000,
    )

    assert "CURRENT_TASK_MUST_KEEP" in context
    assert any("current_task" in item for item in trace["selected_packets"])


def test_working_memory_is_structured_and_not_compressed():
    context, trace = build_small_builder().build(
        message="hello",
        agent_name="router",
        state={
            "trace_id": "trace-1",
            "session_id": "session-1",
            "intent": "knowledge",
            "intent_confidence": 0.8,
            "retry_count": 1,
            "tool_steps": ["should-not-appear"],
        },
        base_context="x" * 1000,
    )

    assert '"trace_id": "trace-1"' in context
    assert "tool_steps" not in context
    assert "should-not-appear" not in context
    assert any("working_memory" in item for item in trace["selected_packets"])


def test_rag_evidence_enters_evidence_section():
    context, _ = build_small_builder().build(
        message="refund",
        rag_context="RAG_EVIDENCE_TEXT",
    )

    assert "[Evidence]" in context
    assert "RAG_EVIDENCE_TEXT" in context


def test_compression_is_triggered_over_budget():
    context, trace = build_small_builder().build(
        message="refund",
        rag_context="重要资料" * 120,
    )

    assert trace["compression_applied"] is True
    assert "...[truncated]" in context


def test_low_relevance_packet_is_dropped():
    builder = build_small_builder()
    packets = builder.gather(
        message="hello",
        state={},
        agent_name="router",
        system_policy="policy",
        current_task="task",
        working_state={},
        base_context="LOW_RELEVANCE_CONTEXT",
        long_context="",
        rag_context=None,
        tool_observations=None,
        output_requirements="requirements",
    )
    for packet in packets:
        if packet.packet_type == "short_term_memory":
            packet.relevance_score = 0.01

    selected, dropped, _ = builder.select(packets)

    assert all("LOW_RELEVANCE_CONTEXT" not in packet.content for packet in selected)
    assert any("LOW_RELEVANCE_CONTEXT" in packet.content for packet in dropped)


def test_output_template_contains_fixed_sections():
    context, _ = build_small_builder().build(message="hello")

    for section in [
        "[Role & Policies]",
        "[Task]",
        "[State]",
        "[User Profile]",
        "[Conversation Summary]",
        "[Recent Messages]",
        "[Evidence]",
        "[Tool Observations]",
        "[Compliance Policy]",
        "[Output Requirements]",
    ]:
        assert section in context
