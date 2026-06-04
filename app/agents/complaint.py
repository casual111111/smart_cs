import re
import time
import uuid

from app.agents.base_tool_agent import BaseToolAgent
from app.tools.registry import ToolResult


class ComplaintAgent(BaseToolAgent):
    """
    投诉处理 Agent。

    负责投诉、举报、不满意和明确要求人工客服的场景。
    """

    def __init__(self):
        super().__init__(
            agent_name="ComplaintAgent 投诉处理智能体",
            allowed_tools=[
                "create_complaint_ticket",
                "escalate_to_human",
                "list_tickets",
            ],
        )

    async def run(
        self,
        message: str,
        user_id: str,
        context: str = "",
        session_id: str = "",
        trace_id: str | None = None,
    ) -> str:
        trace_id = trace_id or str(uuid.uuid4())

        if not self.llm.enabled:
            return await self._fallback_run(
                message=message,
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
            )

        system_instruction = """
你负责处理客服系统中的投诉、举报、不满意和人工介入请求。

你需要遵守以下规则：

1. 用户表达投诉、举报、不满意时：
   - 调用 create_complaint_ticket 创建投诉工单。
   - 如果用户明确要求人工客服，再调用 escalate_to_human。

2. 用户只想查看自己的投诉或工单时：
   - 调用 list_tickets。

3. 不要承诺一定赔偿、一定退款或具体处理结果。
4. 最终回复要礼貌、安抚情绪，并告知已记录和后续会跟进。
"""

        result = await self.react_with_tools(
            message=message,
            user_id=user_id,
            context=context,
            system_instruction=system_instruction,
            session_id=session_id,
            node_name="complaint_agent_node",
            max_steps=4,
            trace_id=trace_id,
        )

        return result.final_answer

    async def _fallback_run(
        self,
        message: str,
        user_id: str,
        session_id: str,
        trace_id: str,
    ) -> str:
        if self._looks_like_ticket_list_query(message):
            result = await self._execute_fallback_tool(
                tool_name="list_tickets",
                arguments={},
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                step=1,
            )
            answer = self._format_list_tickets_result(result)
            self._record_fallback_final(trace_id, session_id, user_id, 2, answer)
            return answer

        ticket = await self._execute_fallback_tool(
            tool_name="create_complaint_ticket",
            arguments={
                "summary": self._build_summary(message),
            },
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            step=1,
        )

        answer = self._format_complaint_ticket_result(ticket)
        self._record_fallback_final(trace_id, session_id, user_id, 2, answer)
        return answer

    async def _execute_fallback_tool(
        self,
        tool_name: str,
        arguments: dict,
        user_id: str,
        session_id: str,
        trace_id: str,
        step: int,
    ) -> ToolResult:
        start_time = time.perf_counter()
        result = await self.tool_registry.execute(
            tool_name=tool_name,
            arguments=arguments,
            user_id=user_id,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        self.trace_tool.record_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            agent_name=self.agent_name,
            node_name="complaint_agent_node",
            step=step,
            action="tool",
            tool_name=tool_name,
            tool_args=arguments,
            tool_result=(
                result.data
                if result.success
                else {"success": False, "error": result.error}
            ),
            status="success" if result.success else "failed",
            error=result.error,
            latency_ms=latency_ms,
        )

        return result

    def _record_fallback_final(
        self,
        trace_id: str,
        session_id: str,
        user_id: str,
        step: int,
        answer: str,
    ) -> None:
        self.trace_tool.record_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            agent_name=self.agent_name,
            node_name="complaint_agent_node",
            step=step,
            action="final",
            tool_name=None,
            tool_args=None,
            tool_result={"answer": answer},
            status="success",
            error=None,
            latency_ms=None,
        )

    def _build_summary(self, message: str) -> str:
        summary = re.sub(r"\s+", " ", message).strip()
        return summary[:200] or "用户发起投诉，需要人工跟进"

    def _looks_like_ticket_list_query(self, text: str) -> bool:
        return any(
            keyword in text
            for keyword in ["我的工单", "投诉工单", "工单列表", "处理进度"]
        )

    def _format_complaint_ticket_result(self, result: ToolResult) -> str:
        if not result.success:
            return f"抱歉，投诉工单创建失败：{result.error}"

        ticket = result.data
        return (
            "很抱歉给你带来不好的体验。已为你创建投诉工单，"
            f"工单号：{ticket['ticket_id']}，当前状态：{ticket['status']}。"
            "我们会转交人工客服继续跟进。"
        )

    def _format_list_tickets_result(self, result: ToolResult) -> str:
        if not result.success:
            return f"工单列表查询失败：{result.error}"

        data = result.data or {}
        items = data.get("items") or []
        if not items:
            return "目前没有查询到你的工单记录。"

        lines = [f"共查询到 {data.get('total', len(items))} 个工单，最近记录如下："]
        for item in items[:5]:
            lines.append(
                f"- {item['ticket_id']}：{item['status']}，{item['summary']}"
            )

        return "\n".join(lines)
