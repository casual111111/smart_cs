import asyncio

import pytest

from app.mcp.schemas import (
    MCPToolsCallResponse,
    MCPToolsCallResult,
)


def test_stdio_server_imports():
    import app.mcp.stdio_server as stdio_server

    assert stdio_server.mcp is not None


def test_knowledge_rag_context_calls_mcp_tool_server(monkeypatch):
    import app.mcp.stdio_server as stdio_server

    calls = []

    async def fake_call_tool(name, arguments, user_id, request_id=None):
        calls.append(
            {
                "name": name,
                "arguments": arguments,
                "user_id": user_id,
                "request_id": request_id,
            }
        )
        return MCPToolsCallResponse(
            result=MCPToolsCallResult(
                content=[],
                structuredContent={"ok": True},
                isError=False,
            )
        )

    monkeypatch.setenv("MCP_USER_ID", "test-user")
    monkeypatch.setattr(stdio_server.mcp_tool_server, "call_tool", fake_call_tool)

    result = asyncio.run(
        stdio_server.knowledge_rag_context("refund policy", top_k=2)
    )

    assert result == {"ok": True}
    assert calls == [
        {
            "name": "knowledge.rag_context",
            "arguments": {
                "query": "refund policy",
                "top_k": 2,
            },
            "user_id": "test-user",
            "request_id": None,
        }
    ]


def test_order_query_calls_mcp_tool_server(monkeypatch):
    import app.mcp.stdio_server as stdio_server

    calls = []

    async def fake_call_tool(name, arguments, user_id, request_id=None):
        calls.append((name, arguments, user_id, request_id))
        return MCPToolsCallResponse(
            result=MCPToolsCallResult(
                content=[],
                structuredContent={"order_id": arguments["order_id"]},
                isError=False,
            )
        )

    monkeypatch.delenv("MCP_USER_ID", raising=False)
    monkeypatch.setattr(stdio_server.mcp_tool_server, "call_tool", fake_call_tool)

    result = asyncio.run(stdio_server.order_query("ORDER-TEST-001"))

    assert result == {"order_id": "ORDER-TEST-001"}
    assert calls == [
        (
            "order.query",
            {"order_id": "ORDER-TEST-001"},
            "mcp-demo-user",
            None,
        )
    ]
