from typing import Any

from app.mcp.server import MCPToolServer


class MCPClient:
    """
    项目内部 MCP-style client。

    后续 Skill / Workflow 可以通过它调用工具，
    避免直接依赖 ToolRegistry。
    """

    def __init__(self, server: MCPToolServer | None = None):
        self.server = server or MCPToolServer()

    def list_tools(self):
        return self.server.list_tools()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str,
    ):
        return await self.server.call_tool(
            name=tool_name,
            arguments=arguments,
            user_id=user_id,
        )