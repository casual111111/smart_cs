# 1. initialize
#    客户端和服务端握手，协商协议版本和能力。

# 2. initialized
#    客户端通知服务端：初始化完成。

# 3. tools/list
#    客户端发现服务端有哪些工具。

# 4. tools/call
#    客户端调用某个工具。
import json
from typing import Any

from app.mcp.schemas import (
    MCPContentItem,
    MCPTool,
    MCPToolsCallResponse,
    MCPToolsCallResult,
    MCPToolsListResponse,
    MCPToolsListResult,
)
from app.tools.registry import ToolRegistry


class MCPToolServer:
    """
    MCP-style 工具协议适配层。

    注意：
    - 不重写业务工具
    - 只把 MCP 工具名映射到 ToolRegistry 内部工具名
    - user_id 必须由认证层传入，不能由前端传入
    """

    TOOL_NAME_MAP: dict[str, str] = {
        "order.query": "get_order",
        "risk.refund_check": "check_refund_eligibility",
        "ticket.refund.create": "create_refund_ticket",
        "ticket.query": "get_ticket",
        "ticket.list": "list_tickets",
        "complaint.create": "create_complaint_ticket",
        "human_review.escalate": "escalate_to_human",
        "knowledge.rag_context": "build_rag_context",
        "knowledge.search": "search_knowledge",
    }

    TITLE_MAP: dict[str, str] = {
        "order.query": "查询订单",
        "risk.refund_check": "退款资格检查",
        "ticket.refund.create": "创建退款工单",
        "ticket.query": "查询工单",
        "ticket.list": "查询工单列表",
        "complaint.create": "创建投诉工单",
        "human_review.escalate": "升级人工处理",
        "knowledge.rag_context": "构建 RAG 上下文",
        "knowledge.search": "检索知识库",
    }

    def __init__(self):
        self.registry = ToolRegistry()

    def list_tools(
        self,
        request_id: int | str | None = None,
    ) -> MCPToolsListResponse:
        internal_tools = {
            tool["name"]: tool
            for tool in self.registry.list_tools()
        }

        tools: list[MCPTool] = []

        for public_name, internal_name in self.TOOL_NAME_MAP.items():
            internal = internal_tools.get(internal_name)
            if internal is None:
                continue

            tools.append(
                MCPTool(
                    name=public_name,
                    title=self.TITLE_MAP.get(public_name),
                    description=internal["description"],
                    inputSchema=internal["parameters"],
                )
            )

        return MCPToolsListResponse(
            id=request_id,
            result=MCPToolsListResult(tools=tools),
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        user_id: str,
        request_id: int | str | None = None,
    ) -> MCPToolsCallResponse:
        internal_name = self.TOOL_NAME_MAP.get(name)

        if internal_name is None:
            return MCPToolsCallResponse(
                id=request_id,
                error={
                    "code": -32602,
                    "message": f"Unknown MCP tool: {name}",
                },
            )

        tool_result = await self.registry.execute(
            tool_name=internal_name,
            arguments=arguments,
            user_id=user_id,
        )

        if not tool_result.success:
            return MCPToolsCallResponse(
                id=request_id,
                result=MCPToolsCallResult(
                    content=[
                        MCPContentItem(
                            text=tool_result.error or "Tool execution failed"
                        )
                    ],
                    structuredContent={
                        "success": False,
                        "tool_name": name,
                        "internal_tool_name": internal_name,
                        "error": tool_result.error,
                    },
                    isError=True,
                ),
            )

        structured = {
            "success": True,
            "tool_name": name,
            "internal_tool_name": internal_name,
            "data": tool_result.data,
        }

        return MCPToolsCallResponse(
            id=request_id,
            result=MCPToolsCallResult(
                content=[
                    MCPContentItem(
                        text=json.dumps(
                            structured,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                ],
                structuredContent=structured,
                isError=False,
            ),
        )