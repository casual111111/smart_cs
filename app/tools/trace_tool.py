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