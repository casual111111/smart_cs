import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.mcp.server import MCPToolServer


mcp = FastMCP(
    "smart-cs",
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
    json_response=True,
)

mcp_tool_server = MCPToolServer()


def get_mcp_user_id() -> str:
    return os.getenv("MCP_USER_ID", "mcp-demo-user")


async def _call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = await mcp_tool_server.call_tool(
        name=name,
        arguments=arguments,
        user_id=get_mcp_user_id(),
    )

    if response.error:
        raise ValueError(response.error.get("message", "MCP tool call failed"))

    if response.result is None:
        raise ValueError("MCP tool call returned no result")

    if response.result.isError:
        structured = response.result.structuredContent or {}
        raise ValueError(structured.get("error") or "MCP tool execution failed")

    return response.result.structuredContent or {}


@mcp.tool()
async def knowledge_rag_context(query: str, top_k: int = 3) -> dict[str, Any]:
    return await _call_mcp_tool(
        name="knowledge.rag_context",
        arguments={
            "query": query,
            "top_k": top_k,
        },
    )


@mcp.tool()
async def order_query(order_id: str) -> dict[str, Any]:
    return await _call_mcp_tool(
        name="order.query",
        arguments={
            "order_id": order_id,
        },
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()