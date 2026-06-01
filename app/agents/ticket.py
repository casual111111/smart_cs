import re

from app.agents.base_tool_agent import BaseToolAgent
from app.models import Ticket
from app.tools.order_tool import OrderTool
from app.tools.ticket_tool import TicketTool


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

        # 保留原工具，供 API 查询接口直接使用
        self.ticket_tool = TicketTool()
        self.order_tool = OrderTool()

    async def run(
    self,
    message: str,
    user_id: str,
    context: str = "",
    session_id: str = "",
) -> str:
        """
        聊天入口。

        现在不再手写：
        - 查订单
        - 判断可退款
        - 创建工单

        而是让 TicketAgent 自己选择工具。
        """

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
        )

        return result.final_answer

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