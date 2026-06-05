import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from app.agents.complaint import ComplaintAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.ticket import TicketAgent
from app.supervisor import Supervisor
from app.tools.registry import ToolResult


class FakeLLM:
    enabled = False


class FakeKnowledgeLLM:
    enabled = True

    def __init__(self):
        self.calls = 0

    async def chat(
        self,
        system_prompt,
        user_prompt,
        temperature=0.0,
    ):
        self.calls += 1

        if self.calls == 1:
            return (
                '{"action": "tool", "tool_name": "build_rag_context", '
                '"arguments": {"query": "退款多久到账？", "top_k": 3}}'
            )

        return (
            '{"action": "final", '
            '"answer": "退款一般会在审核通过后的 3-5 个工作日内原路退回。"}'
        )


class FakeTraceTool:
    def __init__(self):
        self.records = []

    def record_trace(self, **kwargs):
        self.records.append(kwargs)
        return kwargs


class FakeToolRegistry:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {
                "name": "build_rag_context",
                "description": "构建 RAG 上下文",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "search_knowledge",
                "description": "检索知识库",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    async def execute(self, tool_name, arguments, user_id):
        self.calls.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "user_id": user_id,
            }
        )

        if tool_name == "check_refund_eligibility":
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "refundable": True,
                    "reason": "订单符合退款申请条件",
                    "order": {
                        "order_id": arguments["order_id"],
                        "user_id": user_id,
                    },
                },
            )

        if tool_name == "create_refund_ticket":
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "ticket_id": "TK-20260604-ABC123",
                    "status": "created",
                    "type": "refund",
                    "summary": arguments["summary"],
                },
            )

        if tool_name == "get_order":
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "order_id": arguments["order_id"],
                    "product_name": "测试商品",
                    "amount": 99.0,
                    "status": "paid",
                    "refundable": True,
                },
            )

        if tool_name == "create_complaint_ticket":
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "ticket_id": "TK-20260604-COMPLA",
                    "status": "created",
                    "type": "complaint",
                    "summary": arguments["summary"],
                },
            )

        if tool_name == "build_rag_context":
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "query": arguments["query"],
                    "top_k": arguments["top_k"],
                    "max_chars": 1600,
                    "context": (
                        "[资料 1]\n"
                        "来源：refund.md\n"
                        "退款一般会在审核通过后的 3-5 个工作日内原路退回。"
                    ),
                    "sources": ["refund.md"],
                    "items": [
                        {
                            "chunk_id": "refund.md#1",
                            "source": "refund.md",
                            "content": "退款一般会在审核通过后的 3-5 个工作日内原路退回。",
                            "score": 9.0,
                        }
                    ],
                },
            )

        raise AssertionError(f"unexpected tool call: {tool_name}")


def build_ticket_agent():
    agent = TicketAgent()
    agent.llm = FakeLLM()
    agent.tool_registry = FakeToolRegistry()
    agent.trace_tool = FakeTraceTool()
    return agent


def build_complaint_agent():
    agent = ComplaintAgent()
    agent.llm = FakeLLM()
    agent.tool_registry = FakeToolRegistry()
    agent.trace_tool = FakeTraceTool()
    return agent


def test_refund_without_order_id_asks_for_order_id():
    agent = build_ticket_agent()

    answer = asyncio.run(
        agent.run(
            message="我要申请退款",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-1",
        )
    )

    assert "订单号" in answer
    assert agent.tool_registry.calls == []
    assert agent.trace_tool.records[-1]["action"] == "final"
    assert agent.trace_tool.records[-1]["trace_id"] == "trace-1"


def test_second_turn_order_id_continues_refund_from_context():
    agent = build_ticket_agent()

    answer = asyncio.run(
        agent.run(
            message="ORDER-TEST-001",
            user_id="user-1",
            context="用户：我要申请退款\n客服：请提供需要退款的订单号",
            session_id="session-1",
            trace_id="trace-2",
        )
    )

    tool_names = [
        call["tool_name"]
        for call in agent.tool_registry.calls
    ]

    assert tool_names == [
        "check_refund_eligibility",
        "create_refund_ticket",
    ]
    assert "TK-20260604-ABC123" in answer
    assert {
        record["trace_id"]
        for record in agent.trace_tool.records
    } == {"trace-2"}


def test_direct_refund_with_order_id_creates_ticket():
    agent = build_ticket_agent()

    answer = asyncio.run(
        agent.run(
            message="我要退款，订单号：ORDER-TEST-002",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-3",
        )
    )

    assert "已为你创建退款工单" in answer
    assert agent.tool_registry.calls[0]["tool_name"] == "check_refund_eligibility"
    assert agent.tool_registry.calls[1]["tool_name"] == "create_refund_ticket"


@dataclass
class FakeKnowledgeItem:
    source: str
    content: str
    score: float


class FakeKnowledgeTool:
    def search_knowledge(self, query, top_k=3):
        return [
            FakeKnowledgeItem(
                source="refund.md",
                content="退款一般会在 3-5 个工作日内处理。",
                score=8.0,
            )
        ]


def test_knowledge_agent_fallback_searches_knowledge_base():
    agent = KnowledgeAgent()
    agent.llm = FakeLLM()
    agent.knowledge_tool = FakeKnowledgeTool()

    answer = asyncio.run(
        agent.run(
            message="退款多久到账？",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-4",
        )
    )

    assert "refund.md" in answer
    assert "3-5 个工作日" in answer


def test_knowledge_agent_uses_rag_context_and_keeps_sources():
    agent = KnowledgeAgent()
    agent.llm = FakeKnowledgeLLM()
    agent.tool_registry = FakeToolRegistry()
    agent.trace_tool = FakeTraceTool()

    answer = asyncio.run(
        agent.run(
            message="退款多久到账？",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-rag",
        )
    )

    assert "3-5 个工作日" in answer
    assert "参考来源" in answer
    assert "refund.md" in answer
    assert agent.tool_registry.calls[0]["tool_name"] == "build_rag_context"


def test_complaint_creates_ticket_and_review_task():
    complaint_agent = build_complaint_agent()

    answer = asyncio.run(
        complaint_agent.run(
            message="我要投诉，你们服务太差了，转人工",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-5",
        )
    )

    assert "投诉工单" in answer
    assert complaint_agent.tool_registry.calls[0]["tool_name"] == (
        "create_complaint_ticket"
    )

    supervisor = Supervisor()
    supervisor.human_review_tool = SimpleNamespace(
        create_review_task=lambda **kwargs: SimpleNamespace(
            review_id="RV-20260604-ABC123",
            status="pending",
        )
    )
    supervisor.trace_tool = FakeTraceTool()

    result = asyncio.run(
        supervisor._create_review_task_node(
            {
                "session_id": "session-1",
                "trace_id": "trace-5",
                "user_id": "user-1",
                "review_reason": "用户发起投诉；用户要求人工客服",
                "message": "我要投诉，你们服务太差了，转人工",
                "final_response": answer,
            }
        )
    )

    assert result["review_task_id"] == "RV-20260604-ABC123"
    assert supervisor.trace_tool.records[0]["action"] == "create_review_task"
    assert supervisor.trace_tool.records[0]["trace_id"] == "trace-5"


def test_trace_records_share_same_trace_id_for_refund_flow():
    agent = build_ticket_agent()

    asyncio.run(
        agent.run(
            message="我要退款，订单号：ORDER-TRACE-001",
            user_id="user-1",
            session_id="session-1",
            trace_id="trace-shared",
        )
    )

    actions = [
        record["action"]
        for record in agent.trace_tool.records
    ]

    assert actions == ["tool", "tool", "final"]
    assert {
        record["trace_id"]
        for record in agent.trace_tool.records
    } == {"trace-shared"}
