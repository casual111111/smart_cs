from __future__ import annotations

from typing import Any


def calculate_metrics(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, float]:
    result_by_id = {
        result["id"]: result
        for result in results
    }

    return {
        "intent_accuracy": _accuracy(
            cases,
            result_by_id,
            expected_key="expected_intent",
            actual_key="actual_intent",
        ),
        "agent_route_accuracy": _accuracy(
            cases,
            result_by_id,
            expected_key="expected_agent",
            actual_key="actual_agent",
        ),
        "tool_selection_accuracy": _tool_selection_accuracy(cases, result_by_id),
        "rag_source_hit_rate": _source_hit_rate(cases, result_by_id),
        "human_review_trigger_accuracy": _human_review_accuracy(cases, result_by_id),
        "avg_latency_ms": _avg_latency(results),
    }


def _accuracy(
    cases: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    expected_key: str,
    actual_key: str,
) -> float:
    checked = 0
    matched = 0

    for case in cases:
        if expected_key not in case:
            continue

        result = result_by_id.get(case["id"])
        if result is None:
            continue

        checked += 1
        if result.get(actual_key) == case[expected_key]:
            matched += 1

    return _ratio(matched, checked)


def _tool_selection_accuracy(
    cases: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> float:
    checked = 0
    matched = 0

    for case in cases:
        expected_tools = case.get("expected_tools")
        if not expected_tools:
            continue

        result = result_by_id.get(case["id"])
        if result is None:
            continue

        actual_tools = result.get("actual_tools")
        if actual_tools is None:
            continue

        checked += 1
        if all(tool in actual_tools for tool in expected_tools):
            matched += 1

    return _ratio(matched, checked)


def _source_hit_rate(
    cases: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> float:
    checked = 0
    matched = 0

    for case in cases:
        expected_sources = case.get("expected_sources")
        if not expected_sources:
            continue

        result = result_by_id.get(case["id"])
        if result is None:
            continue

        actual_sources = set(result.get("actual_sources") or [])
        response = result.get("response", "")

        checked += 1
        if actual_sources.intersection(expected_sources) or any(
            source in response
            for source in expected_sources
        ):
            matched += 1

    return _ratio(matched, checked)


def _human_review_accuracy(
    cases: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> float:
    checked = 0
    matched = 0

    for case in cases:
        if "expected_human_review" not in case:
            continue

        result = result_by_id.get(case["id"])
        if result is None:
            continue

        checked += 1
        if bool(result.get("actual_human_review")) == bool(
            case["expected_human_review"]
        ):
            matched += 1

    return _ratio(matched, checked)


def _avg_latency(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    total = sum(float(result.get("latency_ms", 0.0)) for result in results)
    return round(total / len(results), 2)


def _ratio(matched: int, checked: int) -> float:
    if checked == 0:
        return 0.0

    return round(matched / checked, 4)
