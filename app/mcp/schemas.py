from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPTool(BaseModel):
    name: str
    title: str | None = None
    description: str
    inputSchema: dict[str, Any]


class MCPToolsListResult(BaseModel):
    tools: list[MCPTool]


class MCPToolsListResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: MCPToolsListResult


class MCPToolsCallParams(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPToolsCallRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: Literal["tools/call"] = "tools/call"
    params: MCPToolsCallParams


class MCPToolsListRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: Literal["tools/list"] = "tools/list"
    params: dict[str, Any] = Field(default_factory=dict)


class MCPContentItem(BaseModel):
    type: str = "text"
    text: str


class MCPToolsCallResult(BaseModel):
    content: list[MCPContentItem]
    structuredContent: dict[str, Any] | None = None
    isError: bool = False


class MCPToolsCallResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: MCPToolsCallResult | None = None
    error: dict[str, Any] | None = None