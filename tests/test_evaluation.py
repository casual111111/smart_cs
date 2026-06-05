import asyncio

from app.evaluation.metrics import calculate_metrics
from app.evaluation.runner import load_eval_cases, run_evaluation


class FakeSupervisor:
    async def handle(self, message, user_id="eval-user", session_id=None):
        if "投诉" in message:
            return {
                "response": "已创建投诉工单，并进入人工审核队列。",
                "intent": "complaint",
                "agent": "complaint",
                "need_human_review": True,
                "tools": ["create_complaint_ticket"],
            }

        if "订单号" in message:
            return {
                "response": "已创建退款工单。",
                "intent": "refund",
                "agent": "ticket",
                "need_human_review": False,
                "tools": ["check_refund_eligibility", "create_refund_ticket"],
            }

        return {
            "response": "退款一般会在 3-5 个工作日内处理。\n参考来源：\n- refund.md",
            "intent": "knowledge",
            "agent": "knowledge",
            "need_human_review": False,
            "sources": ["refund.md"],
        }


def test_load_eval_cases():
    cases = load_eval_cases()

    assert cases
    assert {case["id"] for case in cases} >= {
        "refund_001",
        "knowledge_001",
        "complaint_001",
    }


def test_run_evaluation_with_fake_supervisor():
    report = asyncio.run(run_evaluation(supervisor=FakeSupervisor()))

    assert report["case_count"] == 3
    assert report["metrics"]["intent_accuracy"] == 1.0
    assert report["metrics"]["agent_route_accuracy"] == 1.0
    assert report["metrics"]["rag_source_hit_rate"] == 1.0
    assert report["metrics"]["human_review_trigger_accuracy"] == 1.0


def test_calculate_metrics_handles_partial_tool_observations():
    cases = [
        {
            "id": "case-1",
            "expected_intent": "refund",
            "expected_agent": "ticket",
            "expected_tools": ["check_refund_eligibility"],
        }
    ]
    results = [
        {
            "id": "case-1",
            "actual_intent": "refund",
            "actual_agent": "ticket",
            "actual_tools": ["check_refund_eligibility", "create_refund_ticket"],
            "latency_ms": 20,
        }
    ]

    metrics = calculate_metrics(cases, results)

    assert metrics["intent_accuracy"] == 1.0
    assert metrics["agent_route_accuracy"] == 1.0
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["avg_latency_ms"] == 20.0
