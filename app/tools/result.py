from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """
    工具执行结果。

    success: 工具是否执行成功
    tool_name: 工具名称
    data: 工具返回数据
    error: 错误信息
    """

    success: bool
    tool_name: str
    data: Any = None
    error: str | None = None
