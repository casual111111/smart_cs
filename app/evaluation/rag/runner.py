from __future__ import annotations

import json
import sys
import time
import contextlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from app.evaluation.rag.retrieval_metrics import evaluate_retrieval_cases


DEFAULT_CASES_PATH = Path(__file__).with_name("rag_eval_cases.json")


def load_rag_eval_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def run_rag_evaluation(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cases = cases or load_rag_eval_cases()
    retrieval_results: dict[str, list[dict[str, Any]]] = {}
    latencies = []
    warning: str | None = None

    try:
        retriever = _build_default_retriever()
    except Exception as exc:
        warning = (
            "RAG retriever is unavailable. Start MySQL/Qdrant and rebuild the "
            f"knowledge index for real retrieval evaluation. Details: {exc}"
        )
        retriever = None

    for case in cases:
        start_time = time.perf_counter()
        items = _retrieve_case_items(retriever, case, warning)
        if isinstance(items, tuple):
            items, warning = items

        latencies.append(int((time.perf_counter() - start_time) * 1000))
        retrieval_results[case["id"]] = [_normalize_result_item(item) for item in items]

    report = evaluate_retrieval_cases(cases, retrieval_results)
    report["avg_latency_ms"] = _average(latencies)
    if warning is None and not any(retrieval_results.values()):
        warning = (
            "RAG retrieval returned no chunks for all cases. If this is not expected, "
            "start MySQL/Qdrant and rebuild the knowledge index."
        )
    report["warning"] = warning
    return report


def print_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    print("RAG Retrieval Evaluation")
    print(f"cases: {report['case_count']}")
    print()
    print(f"Hit@3:        {metrics['hit@3']:.2f}")
    print(f"Hit@5:        {metrics['hit@5']:.2f}")
    print(f"Precision@3: {metrics['precision@3']:.2f}")
    print(f"Precision@5: {metrics['precision@5']:.2f}")
    print(f"Recall@3:    {metrics['recall@3']:.2f}")
    print(f"Recall@5:    {metrics['recall@5']:.2f}")
    print(f"MRR@5:        {metrics['mrr@5']:.2f}")
    print(f"Avg latency:  {report['avg_latency_ms']:.0f} ms")

    if report.get("warning"):
        print()
        print(f"Warning: {report['warning']}", file=sys.stderr)

    print()
    print("Per-case:")
    for item in report["per_case"]:
        print(
            f"- {item['id']} hit@5={item['hit@5']:.0f} "
            f"mrr@5={item['mrr@5']:.2f} sources={item['sources']}"
        )


def _build_default_retriever():
    from app.tools.knowledge_tool import KnowledgeTool

    with contextlib.redirect_stdout(sys.stderr):
        tool = KnowledgeTool()

    def retrieve(query: str, top_k: int = 5) -> list[Any]:
        rag_context = tool.build_rag_context(query=query, top_k=top_k)
        return list(getattr(rag_context, "chunks", []) or [])

    return retrieve


def _normalize_result_item(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        data = asdict(item)
    elif isinstance(item, dict):
        data = item
    else:
        data = vars(item)

    return {
        "chunk_id": str(data.get("chunk_id", "")),
        "source": str(data.get("source", "")),
        "content": str(data.get("content", "")),
        "score": float(data.get("score", 0.0) or 0.0),
    }


def _retrieve_case_items(
    retriever: Any | None,
    case: dict[str, Any],
    warning: str | None,
) -> list[Any] | tuple[list[Any], str]:
    if retriever is None:
        return []

    try:
        with contextlib.redirect_stdout(sys.stderr):
            return retriever(case["query"], top_k=5)
    except Exception as exc:
        return [], (
            "RAG retrieval failed. Start MySQL/Qdrant and rebuild the "
            f"knowledge index for real retrieval evaluation. Details: {exc}"
        )


def _average(values: list[int]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def main() -> None:
    print_report(run_rag_evaluation())


if __name__ == "__main__":
    main()
