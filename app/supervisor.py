import uuid
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.memory import ConversationMemory
from app.agents.intent_router import IntentRouterAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.ticket import TicketAgent
from app.agents.compliance import ComplianceAgent
from app.tools.chat_history_tool import ChatHistoryTool


IntentType = Literal[
    "knowledge",
    "ticket",
    "complaint",
    "account",
    "refund",
    "unknown",
]


RouteType = Literal[
    "ticket",
    "knowledge",
    "unknown",
]


class ChatState(TypedDict, total=False):
    """
    LangGraph 共享状态。

    每个节点都从 state 里读取数据，
    然后返回新的字段更新 state。
    """

    message: str
    user_id: str
    session_id: str

    context: str

    intent: IntentType
    intent_confidence: float
    intent_reason: str

    raw_response: str
    final_response: str

    compliance_passed: bool
    compliance_violations: list
    compliance_decision_source: str | None

    memory_count: int


class Supervisor:
    """
    LangGraph 版 Supervisor。

    负责：
    1. 加载会话记忆
    2. 调用 LLM Router Agent 判断意图
    3. 根据意图走不同 Agent 节点
    4. 合规审查
    5. 写入会话记忆
    6. 返回最终结果
    """

    def __init__(self):
        self.memory = ConversationMemory()
        self.intent_router = IntentRouterAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.ticket_agent = TicketAgent()
        self.compliance_agent = ComplianceAgent()

        self.chat_history_tool = ChatHistoryTool()

        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatState)

        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("route_intent", self._route_intent_node)
        graph.add_node("ticket", self._ticket_node)
        graph.add_node("knowledge", self._knowledge_node)
        graph.add_node("unknown", self._unknown_node)
        graph.add_node("compliance", self._compliance_node)
        graph.add_node("save_memory", self._save_memory_node)

        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "route_intent")

        graph.add_conditional_edges(
            "route_intent",
            self._choose_route,
            {
                "ticket": "ticket",
                "knowledge": "knowledge",
                "unknown": "unknown",
            },
        )

        graph.add_edge("ticket", "compliance")
        graph.add_edge("knowledge", "compliance")
        graph.add_edge("unknown", "compliance")

        graph.add_edge("compliance", "save_memory")
        graph.add_edge("save_memory", END)

        return graph.compile()

    async def handle(
        self,
        message: str,
        user_id: str = "anonymous",
        session_id: str | None = None,
    ) -> dict:
        if not session_id or session_id == "string":
            session_id = str(uuid.uuid4())

        initial_state: ChatState = {
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
        }

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "response": final_state["final_response"],
            "session_id": final_state["session_id"],
            "intent": final_state.get("intent", "unknown"),
            "intent_confidence": final_state.get("intent_confidence", 0.0),
            "intent_reason": final_state.get("intent_reason", ""),
            "compliance_passed": final_state.get("compliance_passed", False),
            "memory_count": final_state.get("memory_count", 0),
        }

    async def _load_memory_node(self, state: ChatState) -> dict:
        session_id = state["session_id"]

        context = self.memory.get_recent_context(session_id)

        return {
            "context": context,
        }

    async def _route_intent_node(self, state: ChatState) -> dict:
        intent_result = await self.intent_router.run(
            message=state["message"],
            context=state.get("context", ""),
        )

        return {
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
            "intent_reason": intent_result.reason,
        }

    def _choose_route(self, state: ChatState) -> RouteType:
        intent = state.get("intent", "unknown")

        if intent in ["refund", "complaint", "account", "ticket"]:
            return "ticket"

        if intent == "knowledge":
            return "knowledge"

        return "unknown"

    async def _ticket_node(self, state: ChatState) -> dict:
        raw_response = await self.ticket_agent.run(
            message=state["message"],
            user_id=state["user_id"],
            context=state.get("context", ""),
            session_id=state["session_id"],
        )

        return {
            "raw_response": raw_response,
        }

    async def _knowledge_node(self, state: ChatState) -> dict:
        raw_response = await self.knowledge_agent.run(
            message=state["message"],
        )

        return {
            "raw_response": raw_response,
        }

    async def _unknown_node(self, state: ChatState) -> dict:
        return {
            "raw_response": "抱歉，我暂时没有理解你的问题，可以换一种说法吗？"
        }

    async def _compliance_node(self, state: ChatState) -> dict:
        compliance_result = await self.compliance_agent.check(
            state["raw_response"]
        )

        if not compliance_result["passed"]:
            final_response = (
                "抱歉，系统检测到回复内容可能存在合规风险，"
                "已转交人工客服进一步处理。"
            )
        else:
            final_response = compliance_result["sanitized_content"]

        return {
            "final_response": final_response,
            "compliance_passed": compliance_result["passed"],
            "compliance_violations": compliance_result.get("violations", []),
            "compliance_decision_source": compliance_result.get(
                "decision_source"
            ),
        }

    async def _save_memory_node(self, state: ChatState) -> dict:
        session_id = state["session_id"]
        user_id = state["user_id"]

        user_message = state["message"]
        assistant_message = state["final_response"]

        # 1. 写入 Redis 短期记忆
        self.memory.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )

        self.memory.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
        )

        # 2. 写入 MySQL 长期聊天记录
        title = user_message[:30]

        self.chat_history_tool.ensure_session(
            session_id=session_id,
            user_id=user_id,
            title=title,
        )

        self.chat_history_tool.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=user_message,
        )

        self.chat_history_tool.add_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=assistant_message,
        )

        return {
            "memory_count": self.memory.count(session_id),
        }