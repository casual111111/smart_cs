import asyncio

from app.evaluation.metrics import calculate_metrics
from app.evaluation.runner import load_eval_cases, run_evaluation


class FakeSupervisor:
    def __init__(self):
        self.pending_refund_sessions = set()
        self.trace_tool = type("TraceTool", (), {"records": []})()
        self.trace_counter = 0

    async def handle(self, message, user_id="eval-user", session_id=None):
        session_id = session_id or f"session-{user_id}"
        self.trace_counter += 1
        trace_id = f"trace-{self.trace_counter}"

        if "投诉" in message:
            return {
                "response": "已创建投诉工单，并进入人工审核队列。",
                "session_id": session_id,
                "trace_id": trace_id,
                "intent": "complaint",
                "agent": "complaint",
                "need_human_review": True,
                "tools": ["create_complaint_ticket"],
            }

        if "多久" in message:
            return {
                "response": "退款一般会在 3-5 个工作日内处理。\n参考来源：\n- refund.md",
                "session_id": session_id,
                "trace_id": trace_id,
                "intent": "knowledge",
                "agent": "knowledge",
                "need_human_review": False,
                "sources": ["refund.md"],
            }

        if "退款" in message and "订单" not in message:
            self.pending_refund_sessions.add(session_id)
            return {
                "response": "请提供需要退款的订单号。",
                "session_id": session_id,
                "trace_id": trace_id,
                "intent": "refund",
                "agent": "ticket",
                "need_human_review": False,
                "tools": [],
            }

        if "订单号" in message or "ORDER-" in message:
            self.pending_refund_sessions.discard(session_id)
            self.trace_tool.records.extend(
                [
                    {
                        "trace_id": trace_id,
                        "action": "tool",
                        "tool_name": "check_refund_eligibility",
                    },
                    {
                        "trace_id": trace_id,
                        "action": "tool",
                        "tool_name": "create_refund_ticket",
                    },
                ]
            )
            return {
                "response": "已创建退款工单。",
                "session_id": session_id,
                "trace_id": trace_id,
                "intent": "refund",
                "agent": "ticket",
                "need_human_review": False,
            }

        return {
            "response": "暂未识别该问题。",
            "session_id": session_id,
            "trace_id": trace_id,
            "intent": "unknown",
            "agent": "unknown",
            "need_human_review": False,
        }


def test_load_eval_cases():
    cases = load_eval_cases()

    assert cases
    assert {case["id"] for case in cases} >= {
        "refund_001",
        "refund_002",
        "knowledge_001",
        "complaint_001",
        "trace_001",
    }


def test_run_evaluation_with_fake_supervisor():
    report = asyncio.run(run_evaluation(supervisor=FakeSupervisor()))

    assert report["case_count"] == 5
    assert report["metrics"]["intent_accuracy"] == 1.0
    assert report["metrics"]["agent_route_accuracy"] == 1.0
    assert report["metrics"]["tool_selection_accuracy"] == 1.0
    assert report["metrics"]["rag_source_hit_rate"] == 1.0
    assert report["metrics"]["human_review_trigger_accuracy"] == 1.0
    assert report["metrics"]["trace_consistency_accuracy"] == 1.0


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
