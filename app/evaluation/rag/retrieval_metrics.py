from __future__ import annotations

from pathlib import PurePath
from typing import Any


def normalize_source(source: str) -> str:
    if not source:
        return ""

    normalized = source.replace("\\", "/").split("#", 1)[0]
    return PurePath(normalized).name.lower()


def is_relevant(
    result: dict[str, Any],
    expected_sources: list[str],
    expected_keywords: list[str],
) -> bool:
    source = normalize_source(str(result.get("source", "")))
    expected_source_set = {
        normalize_source(str(expected_source))
        for expected_source in expected_sources
    }
    if source and source in expected_source_set:
        return True

    content = str(result.get("content", "")).lower()
    return any(
        str(keyword).lower() in content
        for keyword in expected_keywords
        if str(keyword).strip()
    )


def hit_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    top_results = _top_k(results, k)
    return float(
        any(is_relevant(result, expected_sources, expected_keywords) for result in top_results)
    )


def precision_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    top_results = _top_k(results, k)
    if not top_results or k <= 0:
        return 0.0

    relevant_count = sum(
        1
        for result in top_results
        if is_relevant(result, expected_sources, expected_keywords)
    )
    return round(relevant_count / k, 4)


def recall_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    expected_items = _expected_items(expected_sources, expected_keywords)
    if not expected_items:
        return 0.0

    top_results = _top_k(results, k)
    hit_count = sum(
        1
        for item in expected_items
        if _expected_item_hit(item, top_results)
    )
    return round(hit_count / len(expected_items), 4)


def mrr_at_k(
    results: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    for index, result in enumerate(_top_k(results, k), start=1):
        if is_relevant(result, expected_sources, expected_keywords):
            return round(1 / index, 4)

    return 0.0


def evaluate_retrieval_cases(
    cases: list[dict[str, Any]],
    retrieval_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_case = []
    for case in cases:
        results = retrieval_results.get(case["id"], [])
        expected_sources = list(case.get("expected_sources", []))
        expected_keywords = list(case.get("expected_keywords", []))

        per_case.append(
            {
                "id": case["id"],
                "hit@3": hit_at_k(results, expected_sources, expected_keywords, 3),
                "hit@5": hit_at_k(results, expected_sources, expected_keywords, 5),
                "precision@3": precision_at_k(results, expected_sources, expected_keywords, 3),
                "precision@5": precision_at_k(results, expected_sources, expected_keywords, 5),
                "recall@3": recall_at_k(results, expected_sources, expected_keywords, 3),
                "recall@5": recall_at_k(results, expected_sources, expected_keywords, 5),
                "mrr@5": mrr_at_k(results, expected_sources, expected_keywords, 5),
                "sources": _unique_sources(results),
            }
        )

    metrics = {
        metric_name: _average(per_case, metric_name)
        for metric_name in [
            "hit@3",
            "hit@5",
            "precision@3",
            "precision@5",
            "recall@3",
            "recall@5",
            "mrr@5",
        ]
    }
    return {
        "case_count": len(cases),
        "metrics": metrics,
        "per_case": per_case,
    }


def _top_k(results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []

    return results[:k]


def _expected_items(
    expected_sources: list[str],
    expected_keywords: list[str],
) -> list[tuple[str, str]]:
    items = [
        ("source", normalize_source(str(source)))
        for source in expected_sources
        if str(source).strip()
    ]
    items.extend(
        ("keyword", str(keyword).lower())
        for keyword in expected_keywords
        if str(keyword).strip()
    )
    return items


def _expected_item_hit(
    item: tuple[str, str],
    results: list[dict[str, Any]],
) -> bool:
    item_type, value = item
    if item_type == "source":
        return any(normalize_source(str(result.get("source", ""))) == value for result in results)

    return any(value in str(result.get("content", "")).lower() for result in results)


def _unique_sources(results: list[dict[str, Any]]) -> list[str]:
    sources = []
    for result in results:
        source = normalize_source(str(result.get("source", "")))
        if source and source not in sources:
            sources.append(source)
    return sources


def _average(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0

    return round(sum(float(item.get(key, 0.0)) for item in items) / len(items), 4)
