import json
from datetime import datetime
from typing import Any

from app.database import get_db_session
from app.models import AgentTrace


class TraceTool:
    """
    Agent Trace 工具。

    负责记录和查询 Agent 执行轨迹。
    """

    def record_trace(
        self,
        trace_id: str,
        session_id: str,
        user_id: str,
        agent_name: str,
        node_name: str,
        step: int,
        action: str,
        tool_name: str | None = None,
        tool_args: Any = None,
        tool_result: Any = None,
        status: str = "success",
        error: str | None = None,
        latency_ms: int | None = None,
    ) -> AgentTrace:
        db = get_db_session()

        try:
            trace = AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                agent_name=agent_name,
                node_name=node_name,
                step=step,
                action=action,
                tool_name=tool_name,
                tool_args=self._to_json(tool_args),
                tool_result=self._to_json(tool_result),
                status=status,
                error=error,
                latency_ms=latency_ms,
                created_at=datetime.now(),
            )

            db.add(trace)
            db.commit()
            db.refresh(trace)

            return trace

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

    def list_traces(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AgentTrace], int]:
        db = get_db_session()

        try:
            query = db.query(AgentTrace)

            if user_id:
                query = query.filter(AgentTrace.user_id == user_id)

            if session_id:
                query = query.filter(AgentTrace.session_id == session_id)

            if trace_id:
                query = query.filter(AgentTrace.trace_id == trace_id)

            total = query.count()

            traces = (
                query
                .order_by(AgentTrace.created_at.asc(), AgentTrace.step.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return traces, total

        finally:
            db.close()

    def get_metrics(self) -> dict:
        db = get_db_session()

        try:
            traces = db.query(AgentTrace).all()

            trace_ids = {
                item.trace_id
                for item in traces
                if item.trace_id
            }

            latency_values = [
                item.latency_ms
                for item in traces
                if item.latency_ms is not None
            ]

            tool_calls = [
                item
                for item in traces
                if item.action == "tool"
            ]
            tool_success = [
                item
                for item in tool_calls
                if item.status == "success"
            ]

            compliance_checks = [
                item
                for item in traces
                if item.node_name == "compliance_agent_node"
            ]
            compliance_blocks = [
                item
                for item in compliance_checks
                if item.status != "success"
            ]

            intent_distribution: dict[str, int] = {}
            for item in traces:
                if item.node_name != "router_agent_node":
                    continue

                result = self.parse_json(item.tool_result)
                if isinstance(result, dict):
                    intent = result.get("intent", "unknown")
                    intent_distribution[intent] = (
                        intent_distribution.get(intent, 0) + 1
                    )

            rag_calls = [
                item
                for item in tool_calls
                if item.tool_name == "search_knowledge"
            ]
            rag_hits = 0
            for item in rag_calls:
                result = self.parse_json(item.tool_result)
                if isinstance(result, dict) and result.get("items"):
                    rag_hits += 1

            return {
                "total_requests": len(trace_ids),
                "avg_latency_ms": self._avg(latency_values),
                "p95_latency_ms": self._p95(latency_values),
                "tool_success_rate": self._rate(len(tool_success), len(tool_calls)),
                "tool_error_rate": self._rate(
                    len(tool_calls) - len(tool_success),
                    len(tool_calls),
                ),
                "intent_distribution": intent_distribution,
                "compliance_block_rate": self._rate(
                    len(compliance_blocks),
                    len(compliance_checks),
                ),
                "rag_hit_rate": self._rate(rag_hits, len(rag_calls)),
            }

        finally:
            db.close()

    def _to_json(self, value: Any) -> str | None:
        if value is None:
            return None

        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def parse_json(self, value: str | None):
        if value is None:
            return None

        try:
            return json.loads(value)
        except Exception:
            return value

    def _avg(self, values: list[int]) -> float:
        if not values:
            return 0.0

        return round(sum(values) / len(values), 2)

    def _p95(self, values: list[int]) -> float:
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = max(0, int(len(sorted_values) * 0.95) - 1)
        return float(sorted_values[index])

    def _rate(self, numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0

        return round(numerator / denominator, 4)
