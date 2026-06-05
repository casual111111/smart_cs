from __future__ import annotations

import re
import time
import uuid
from typing import TYPE_CHECKING

from app.agents.base_tool_agent import BaseToolAgent
from app.tools.result import ToolResult

if TYPE_CHECKING:
    from app.models import Ticket


class TicketAgent(BaseToolAgent):
    """
    工单处理 Agent。

    当前版本：
    - 仍然是专业 TicketAgent
    - 但内部具备自主工具调用能力
    - 只能调用订单和工单相关工具
    """

    def __init__(self):
        super().__init__(
            agent_name="TicketAgent 工单处理智能体",
            allowed_tools=[
                "get_order",
                "check_refund_eligibility",
                "create_refund_ticket",
                "get_ticket",
                "list_tickets",
            ],
        )

        # 保留原工具，供 API 查询接口直接使用；真实用到时再初始化。
        self._ticket_tool = None
        self._order_tool = None

    @property
    def ticket_tool(self):
        if self._ticket_tool is None:
            from app.tools.ticket_tool import TicketTool

            self._ticket_tool = TicketTool()

        return self._ticket_tool

    @ticket_tool.setter
    def ticket_tool(self, value):
        self._ticket_tool = value

    @property
    def order_tool(self):
        if self._order_tool is None:
            from app.tools.order_tool import OrderTool

            self._order_tool = OrderTool()

        return self._order_tool

    @order_tool.setter
    def order_tool(self, value):
        self._order_tool = value

    async def run(
        self,
        message: str,
        user_id: str,
        context: str = "",
        session_id: str = "",
        trace_id: str | None = None,
    ) -> str:
        """
        聊天入口。

        现在不再手写：
        - 查订单
        - 判断可退款
        - 创建工单

        而是让 TicketAgent 自己选择工具。
        """

        trace_id = trace_id or str(uuid.uuid4())

        if not self.llm.enabled:
            return await self._fallback_run(
                message=message,
                user_id=user_id,
                context=context,
                session_id=session_id,
                trace_id=trace_id,
            )

        system_instruction = """
你负责处理客服系统中的工单、退款、投诉、订单相关问题。

你需要遵守以下规则：

1. 用户申请退款时：
   - 必须先调用 check_refund_eligibility 检查退款资格。
   - 如果 refundable 为 true，再调用 create_refund_ticket 创建退款工单。
   - 如果 refundable 为 false，不要创建工单，要告诉用户原因。

2. 用户查询工单时：
   - 如果消息里有工单号，例如 TK-20260529-ABC123，调用 get_ticket。

3. 用户想看自己的工单列表时：
   - 调用 list_tickets。

4. 用户只是提供订单号，并且上下文中存在退款需求：
   - 按退款流程处理，先检查退款资格。

5. 不要编造订单、工单或状态。
6. 工具返回错误时，必须把错误原因解释给用户。
7. 最终回答要简洁、礼貌、适合客服场景。
"""

        result = await self.react_with_tools(
            message=message,
            user_id=user_id,
            context=context,
            system_instruction=system_instruction,
            session_id=session_id,
            node_name="ticket_agent_node",
            max_steps=5,
            trace_id=trace_id,
        )

        return result.final_answer

    async def _fallback_run(
        self,
        message: str,
        user_id: str,
        context: str,
        session_id: str,
        trace_id: str,
    ) -> str:
        """
        未配置 LLM 时的基础业务兜底。

        覆盖退款链路、订单查询、工单查询和工单列表查询，
        让本地测试环境也能跑通核心客服流程。
        """
        normalized = message.strip()
        combined_text = f"{context}\n{normalized}"

        ticket_id = self._extract_ticket_id(normalized)
        if ticket_id:
            result = await self._execute_fallback_tool(
                tool_name="get_ticket",
                arguments={"ticket_id": ticket_id},
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                step=1,
            )
            answer = self._format_get_ticket_result(result)
            self._record_fallback_final(trace_id, session_id, user_id, 2, answer)
            return answer

        if self._looks_like_ticket_list_query(normalized):
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

        order_id = self._extract_order_id(normalized)
        if self._contains_refund_intent(combined_text):
            if not order_id:
                answer = "可以，请提供需要退款的订单号，我会先帮你检查是否符合退款条件。"
                self._record_fallback_final(trace_id, session_id, user_id, 1, answer)
                return answer

            eligibility = await self._execute_fallback_tool(
                tool_name="check_refund_eligibility",
                arguments={"order_id": order_id},
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                step=1,
            )

            if not eligibility.success:
                answer = f"退款资格检查失败：{eligibility.error}"
                self._record_fallback_final(trace_id, session_id, user_id, 2, answer)
                return answer

            data = eligibility.data or {}
            if not data.get("refundable"):
                reason = data.get("reason") or "该订单暂不符合退款条件"
                answer = f"抱歉，该订单暂不能申请退款。原因：{reason}"
                self._record_fallback_final(trace_id, session_id, user_id, 2, answer)
                return answer

            ticket = await self._execute_fallback_tool(
                tool_name="create_refund_ticket",
                arguments={
                    "order_id": order_id,
                    "summary": f"用户申请退款，订单号：{order_id}",
                },
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                step=2,
            )
            answer = self._format_create_refund_ticket_result(ticket)
            self._record_fallback_final(trace_id, session_id, user_id, 3, answer)
            return answer

        if order_id:
            result = await self._execute_fallback_tool(
                tool_name="get_order",
                arguments={"order_id": order_id},
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                step=1,
            )
            answer = self._format_get_order_result(result)
            self._record_fallback_final(trace_id, session_id, user_id, 2, answer)
            return answer

        answer = "我可以帮你处理退款、订单和工单问题。请提供订单号或工单号，我会继续为你查询。"
        self._record_fallback_final(trace_id, session_id, user_id, 1, answer)
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
            node_name="ticket_agent_node",
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
            node_name="ticket_agent_node",
            step=step,
            action="final",
            tool_name=None,
            tool_args=None,
            tool_result={"answer": answer},
            status="success",
            error=None,
            latency_ms=None,
        )

    def _extract_ticket_id(self, text: str) -> str | None:
        match = re.search(r"TK-\d{8}-[A-Za-z0-9]{6}", text)
        return match.group(0) if match else None

    def _extract_order_id(self, text: str) -> str | None:
        labeled_match = re.search(
            r"订单(?:号|编号)?[:：]?\s*([A-Za-z0-9][A-Za-z0-9-]{5,63})",
            text,
            re.IGNORECASE,
        )
        if labeled_match:
            return labeled_match.group(1)

        order_match = re.search(r"\bORDER-[A-Za-z0-9-]{3,58}\b", text, re.IGNORECASE)
        if order_match:
            return order_match.group(0)

        stripped = text.strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{5,63}", stripped):
            return stripped

        return None

    def _contains_refund_intent(self, text: str) -> bool:
        return any(
            keyword in text
            for keyword in ["退款", "退钱", "退货", "退费", "申请退款"]
        )

    def _looks_like_ticket_list_query(self, text: str) -> bool:
        return any(
            keyword in text
            for keyword in ["我的工单", "工单列表", "查询工单列表", "所有工单"]
        )

    def _format_get_ticket_result(self, result: ToolResult) -> str:
        if not result.success:
            return f"工单查询失败：{result.error}"

        ticket = result.data
        return (
            f"已查询到工单 {ticket['ticket_id']}："
            f"状态为 {ticket['status']}，类型为 {ticket['type']}，"
            f"摘要：{ticket['summary']}。"
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

    def _format_create_refund_ticket_result(self, result: ToolResult) -> str:
        if not result.success:
            return f"退款工单创建失败：{result.error}"

        ticket = result.data
        return (
            "订单符合退款申请条件，已为你创建退款工单。"
            f"工单号：{ticket['ticket_id']}，当前状态：{ticket['status']}。"
        )

    def _format_get_order_result(self, result: ToolResult) -> str:
        if not result.success:
            return f"订单查询失败：{result.error}"

        order = result.data
        refundable = "支持退款" if order["refundable"] else "不支持退款"
        return (
            f"已查询到订单 {order['order_id']}："
            f"商品为 {order['product_name']}，金额 {order['amount']}，"
            f"当前状态 {order['status']}，{refundable}。"
        )

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        return self.ticket_tool.get_ticket(ticket_id)

    def list_tickets(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Ticket], int]:
        return self.ticket_tool.list_tickets(
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def update_ticket_status(
        self,
        ticket_id: str,
        new_status: str,
    ) -> tuple[Ticket | None, str | None]:
        return self.ticket_tool.update_ticket_status(
            ticket_id=ticket_id,
            new_status=new_status,
        )
