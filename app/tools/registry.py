from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.models import Order, Ticket
from app.tools.knowledge_tool import KnowledgeTool
from app.tools.order_tool import OrderTool
from app.tools.ticket_tool import TicketTool


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


class ToolRegistry:
    """
    工具注册中心。

    负责：
    1. 统一管理所有工具
    2. 提供工具 schema 给 LLM
    3. 根据 tool_name 执行具体工具
    4. 把 ORM 对象转换成可 JSON 序列化的数据

    这一步是后续 ToolCallingAgent 的基础。
    """

    def __init__(self):
        self.order_tool = OrderTool()
        self.ticket_tool = TicketTool()
        self.knowledge_tool = KnowledgeTool()

        self.tools: dict[str, dict[str, Any]] = {
            "get_order": {
                "description": "根据订单号查询当前用户的订单信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "订单号，例如 ORDER-TEST-001",
                        }
                    },
                    "required": ["order_id"],
                },
                "handler": self._get_order,
            },
            "check_refund_eligibility": {
                "description": "检查订单是否存在、是否属于当前用户、是否可以退款。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "需要检查退款资格的订单号",
                        }
                    },
                    "required": ["order_id"],
                },
                "handler": self._check_refund_eligibility,
            },
            "create_refund_ticket": {
                "description": "在订单可退款的情况下，创建退款工单。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "需要退款的订单号",
                        },
                        "summary": {
                            "type": "string",
                            "description": "工单摘要，可选",
                        },
                    },
                    "required": ["order_id"],
                },
                "handler": self._create_refund_ticket,
            },
            "get_ticket": {
                "description": "根据工单号查询当前用户的工单信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticket_id": {
                            "type": "string",
                            "description": "工单号，例如 TK-20260529-ABC123",
                        }
                    },
                    "required": ["ticket_id"],
                },
                "handler": self._get_ticket,
            },
            "list_tickets": {
                "description": "查询当前用户的工单列表，可以按状态过滤。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "工单状态，例如 created、processing、resolved、closed",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回数量，默认 20",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "分页偏移量，默认 0",
                        },
                    },
                    "required": [],
                },
                "handler": self._list_tickets,
            },
            "create_complaint_ticket": {
                "description": "创建高优先级投诉工单，用于投诉、举报、不满意等场景。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "投诉摘要",
                        },
                        "order_id": {
                            "type": "string",
                            "description": "相关订单号，可选",
                        },
                    },
                    "required": ["summary"],
                },
                "handler": self._create_complaint_ticket,
            },
            "escalate_to_human": {
                "description": "标记当前问题需要人工客服介入。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "需要人工介入的原因",
                        }
                    },
                    "required": ["reason"],
                },
                "handler": self._escalate_to_human,
            },
            "build_rag_context": {
                "description": "构建可直接用于回答的 RAG 上下文，返回知识片段、来源和合并后的 context。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用户问题或检索关键词",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回最相关的知识片段数量，默认 3",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "RAG context 最大字符数，默认 1600",
                        },
                    },
                    "required": ["query"],
                },
                "handler": self._build_rag_context,
            },
            "search_knowledge": {
                "description": "检索本地知识库，回答政策、流程、规则类问题。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用户问题或检索关键词",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回最相关的知识片段数量，默认 3",
                        },
                    },
                    "required": ["query"],
                },
                "handler": self._search_knowledge,
            },
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """
        返回工具 schema。

        注意：
        不返回 handler，因为 handler 是 Python 函数，不能给前端或 LLM。
        """
        result = []

        for name, config in self.tools.items():
            result.append(
                {
                    "name": name,
                    "description": config["description"],
                    "parameters": config["parameters"],
                }
            )

        return result

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str,
    ) -> ToolResult:
        """
        执行工具。

        tool_name: 工具名称
        arguments: 工具参数
        user_id: 当前登录用户 ID，由 JWT 解析得到，不能由前端伪造
        """
        tool_config = self.tools.get(tool_name)

        if tool_config is None:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"未知工具：{tool_name}",
            )

        handler: Callable[..., Any] = tool_config["handler"]

        try:
            data = handler(
                user_id=user_id,
                **arguments,
            )

            return ToolResult(
                success=True,
                tool_name=tool_name,
                data=data,
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(exc),
            )

    def _get_order(
        self,
        user_id: str,
        order_id: str,
    ) -> dict[str, Any]:
        order = self.order_tool.get_order(order_id)

        if order is None:
            raise ValueError(f"订单不存在：{order_id}")

        if order.user_id != user_id:
            raise PermissionError("无权查看该订单")

        return self._serialize_order(order)

    def _check_refund_eligibility(
        self,
        user_id: str,
        order_id: str,
    ) -> dict[str, Any]:
        order = self.order_tool.get_order(order_id)

        if order is None:
            return {
                "refundable": False,
                "reason": f"订单不存在：{order_id}",
                "order": None,
            }

        if order.user_id != user_id:
            return {
                "refundable": False,
                "reason": "该订单不属于当前用户",
                "order": None,
            }

        refundable, reason = self.order_tool.is_refundable(order)

        return {
            "refundable": refundable,
            "reason": reason,
            "order": self._serialize_order(order),
        }

    def _create_refund_ticket(
        self,
        user_id: str,
        order_id: str,
        summary: str | None = None,
    ) -> dict[str, Any]:
        order = self.order_tool.get_order(order_id)

        if order is None:
            raise ValueError(f"订单不存在：{order_id}")

        if order.user_id != user_id:
            raise PermissionError("该订单不属于当前用户，无法创建退款工单")

        refundable, reason = self.order_tool.is_refundable(order)

        if not refundable:
            raise ValueError(f"该订单暂不支持退款：{reason}")

        ticket = self.ticket_tool.create_ticket(
            user_id=user_id,
            ticket_type="refund",
            priority="medium",
            summary=summary or f"用户申请退款，订单号：{order_id}",
            order_id=order_id,
        )

        return self._serialize_ticket(ticket)

    def _get_ticket(
        self,
        user_id: str,
        ticket_id: str,
    ) -> dict[str, Any]:
        ticket = self.ticket_tool.get_ticket(ticket_id)

        if ticket is None:
            raise ValueError(f"工单不存在：{ticket_id}")

        if ticket.user_id != user_id:
            raise PermissionError("无权查看该工单")

        return self._serialize_ticket(ticket)

    def _list_tickets(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        tickets, total = self.ticket_tool.list_tickets(
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset,
        )

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [
                self._serialize_ticket(ticket)
                for ticket in tickets
            ],
        }

    def _create_complaint_ticket(
        self,
        user_id: str,
        summary: str,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id:
            order = self.order_tool.get_order(order_id)

            if order is None:
                raise ValueError(f"订单不存在：{order_id}")

            if order.user_id != user_id:
                raise PermissionError("该订单不属于当前用户，无法创建投诉工单")

        ticket = self.ticket_tool.create_ticket(
            user_id=user_id,
            ticket_type="complaint",
            priority="high",
            summary=summary[:500] or "用户发起投诉，需要人工跟进",
            order_id=order_id,
        )

        return self._serialize_ticket(ticket)

    def _escalate_to_human(
        self,
        user_id: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "need_human_review": True,
            "reason": reason,
            "message": "已标记为需要人工客服介入",
        }

    def _search_knowledge(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        return self._build_rag_context(
            user_id=user_id,
            query=query,
            top_k=top_k,
        )

    def _build_rag_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        max_chars: int = 1600,
    ) -> dict[str, Any]:
        top_k = max(1, min(top_k, 10))
        max_chars = max(200, min(max_chars, 5000))

        rag_context = self.knowledge_tool.build_rag_context(
            query=query,
            top_k=top_k,
            max_chars=max_chars,
        )

        return {
            "query": query,
            "top_k": top_k,
            "max_chars": max_chars,
            "context": rag_context.context,
            "sources": rag_context.sources,
            "items": [
                {
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "content": item.content,
                    "score": item.score,
                }
                for item in rag_context.chunks
            ],
        }

    def _serialize_order(self, order: Order) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "product_name": order.product_name,
            "amount": order.amount,
            "status": order.status,
            "refundable": order.refundable,
            "created_at": self._format_datetime(order.created_at),
        }

    def _serialize_ticket(self, ticket: Ticket) -> dict[str, Any]:
        return {
            "ticket_id": ticket.ticket_id,
            "user_id": ticket.user_id,
            "type": ticket.type,
            "priority": ticket.priority,
            "status": ticket.status,
            "summary": ticket.summary,
            "order_id": ticket.order_id,
            "created_at": self._format_datetime(ticket.created_at),
        }

    def _format_datetime(self, value: datetime | None) -> str | None:
        if value is None:
            return None

        return value.isoformat(timespec="seconds")
