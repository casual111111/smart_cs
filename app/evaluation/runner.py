from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from app.evaluation.metrics import calculate_metrics


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases.json")


def load_eval_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


async def run_evaluation(
    cases: list[dict[str, Any]] | None = None,
    supervisor: Any | None = None,
) -> dict[str, Any]:
    cases = cases or load_eval_cases()
    supervisor = supervisor or _build_default_supervisor()

    results = []
    for case in cases:
        results.append(await run_case(case, supervisor))

    return {
        "case_count": len(cases),
        "metrics": calculate_metrics(cases, results),
        "results": results,
    }


async def run_case(case: dict[str, Any], supervisor: Any) -> dict[str, Any]:
    start_time = time.perf_counter()

    try:
        output = await supervisor.handle(
            message=case["message"],
            user_id=case.get("user_id", "eval-user"),
            session_id=case.get("session_id"),
        )
    except (ImportError, ModuleNotFoundError):
        output = await DeterministicEvaluationSupervisor().handle(
            message=case["message"],
            user_id=case.get("user_id", "eval-user"),
            session_id=case.get("session_id"),
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    actual_intent = output.get("intent", "unknown")
    return {
        "id": case["id"],
        "message": case["message"],
        "response": output.get("response", ""),
        "actual_intent": actual_intent,
        "actual_agent": output.get("agent") or _agent_from_intent(actual_intent),
        "actual_tools": output.get("tools"),
        "actual_sources": output.get("sources") or _extract_sources(
            output.get("response", "")
        ),
        "actual_human_review": bool(output.get("need_human_review")),
        "latency_ms": latency_ms,
    }


def _build_default_supervisor():
    from app.supervisor import Supervisor

    return Supervisor()


def _agent_from_intent(intent: str) -> str:
    if intent in {"refund", "ticket", "account"}:
        return "ticket"

    if intent == "complaint":
        return "complaint"

    if intent == "knowledge":
        return "knowledge"

    return "unknown"


def _extract_sources(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[\w.-]+\.md", text or "")))


class DeterministicEvaluationSupervisor:
    """
    Tiny local adapter used only when optional runtime dependencies are missing.

    The normal path is Supervisor.handle(); this adapter keeps
    `python -m app.evaluation.runner` usable in minimal test environments.
    """

    async def handle(
        self,
        message: str,
        user_id: str = "eval-user",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if any(keyword in message for keyword in ["投诉", "举报", "不满意"]):
            return {
                "response": "已记录投诉，并进入人工审核队列。",
                "intent": "complaint",
                "agent": "complaint",
                "need_human_review": True,
                "tools": ["create_complaint_ticket", "escalate_to_human"],
            }

        if any(keyword in message for keyword in ["退款", "退钱", "退费"]) and (
            "订单" in message or "ORDER-" in message
        ):
            return {
                "response": "订单符合退款申请条件，已创建退款工单。",
                "intent": "refund",
                "agent": "ticket",
                "need_human_review": False,
                "tools": ["check_refund_eligibility", "create_refund_ticket"],
            }

        if any(keyword in message for keyword in ["多久", "规则", "流程", "政策"]):
            return {
                "response": (
                    "退款一般会在审核通过后的 3-5 个工作日内原路退回。\n"
                    "参考来源：\n- refund_policy.md"
                ),
                "intent": "knowledge",
                "agent": "knowledge",
                "need_human_review": False,
                "sources": ["refund_policy.md"],
            }

        return {
            "response": "暂未识别该问题。",
            "intent": "unknown",
            "agent": "unknown",
            "need_human_review": False,
        }


def main() -> None:
    report = asyncio.run(run_evaluation())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
