import uuid
import time
from typing import Literal, TypedDict

from app.memory.working_memory import create_working_memory

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
    "complaint",
    "unknown",
]


ReviewRouteType = Literal[
    "create_review_task",
    "save_memory",
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
    optimized_context: str
    context_trace: dict

    intent: IntentType
    intent_confidence: float
    intent_reason: str

    raw_response: str
    final_response: str

    compliance_passed: bool
    compliance_violations: list
    compliance_decision_source: str | None

    memory_count: int
####################################
    trace_id: str
    current_agent: str
    sub_results: dict
    tool_steps: list
    tool_observations: list
    retry_count: int
    compliance_result: dict
    need_human_review: bool
    review_reason: str
    review_task_id: str | None


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
        self._memory = None
        self._intent_router = None
        self._knowledge_agent = None
        self._ticket_agent = None
        self._complaint_agent = None
        self._compliance_agent = None

        self._chat_history_tool = None
        self._human_review_tool = None
        self._long_term_memory_tool = None
        self._trace_tool = None
        self._context_builder = None

        self.graph = None

    @property
    def memory(self):
        if self._memory is None:
            from app.memory import ConversationMemory

            self._memory = ConversationMemory()

        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    @property
    def intent_router(self):
        if self._intent_router is None:
            from app.agents.intent_router import IntentRouterAgent

            self._intent_router = IntentRouterAgent()

        return self._intent_router

    @intent_router.setter
    def intent_router(self, value):
        self._intent_router = value

    @property
    def knowledge_agent(self):
        if self._knowledge_agent is None:
            from app.agents.knowledge import KnowledgeAgent

            self._knowledge_agent = KnowledgeAgent()

        return self._knowledge_agent

    @knowledge_agent.setter
    def knowledge_agent(self, value):
        self._knowledge_agent = value

    @property
    def ticket_agent(self):
        if self._ticket_agent is None:
            from app.agents.ticket import TicketAgent

            self._ticket_agent = TicketAgent()

        return self._ticket_agent

    @ticket_agent.setter
    def ticket_agent(self, value):
        self._ticket_agent = value

    @property
    def complaint_agent(self):
        if self._complaint_agent is None:
            from app.agents.complaint import ComplaintAgent

            self._complaint_agent = ComplaintAgent()

        return self._complaint_agent

    @complaint_agent.setter
    def complaint_agent(self, value):
        self._complaint_agent = value

    @property
    def compliance_agent(self):
        if self._compliance_agent is None:
            from app.agents.compliance import ComplianceAgent

            self._compliance_agent = ComplianceAgent()

        return self._compliance_agent

    @compliance_agent.setter
    def compliance_agent(self, value):
        self._compliance_agent = value

    @property
    def chat_history_tool(self):
        if self._chat_history_tool is None:
            from app.tools.chat_history_tool import ChatHistoryTool

            self._chat_history_tool = ChatHistoryTool()

        return self._chat_history_tool

    @chat_history_tool.setter
    def chat_history_tool(self, value):
        self._chat_history_tool = value

    @property
    def human_review_tool(self):
        if self._human_review_tool is None:
            from app.tools.human_review_tool import HumanReviewTool

            self._human_review_tool = HumanReviewTool()

        return self._human_review_tool

    @human_review_tool.setter
    def human_review_tool(self, value):
        self._human_review_tool = value

    @property
    def long_term_memory_tool(self):
        if self._long_term_memory_tool is None:
            from app.tools.long_term_memory_tool import LongTermMemoryTool

            self._long_term_memory_tool = LongTermMemoryTool()

        return self._long_term_memory_tool

    @long_term_memory_tool.setter
    def long_term_memory_tool(self, value):
        self._long_term_memory_tool = value

    @property
    def trace_tool(self):
        if self._trace_tool is None:
            from app.tools.trace_tool import TraceTool

            self._trace_tool = TraceTool()

        return self._trace_tool

    @trace_tool.setter
    def trace_tool(self, value):
        self._trace_tool = value

    @property
    def context_builder(self):
        if self._context_builder is None:
            from app.context import SmartCSContextBuilder

            self._context_builder = SmartCSContextBuilder()

        return self._context_builder

    @context_builder.setter
    def context_builder(self, value):
        self._context_builder = value

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(ChatState)

        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("build_context", self._build_context_node)
        graph.add_node("route_intent", self._route_intent_node)
        graph.add_node("ticket", self._ticket_node)
        graph.add_node("knowledge", self._knowledge_node)
        graph.add_node("complaint", self._complaint_node)
        graph.add_node("unknown", self._unknown_node)
        graph.add_node("compliance", self._compliance_node)
        graph.add_node("human_review_check", self._human_review_check_node)
        graph.add_node("create_review_task", self._create_review_task_node)
        graph.add_node("save_memory", self._save_memory_node)

        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "build_context")
        graph.add_edge("build_context", "route_intent")

        graph.add_conditional_edges(
            "route_intent",
            self._choose_route,
            {
                "ticket": "ticket",
                "knowledge": "knowledge",
                "complaint": "complaint",
                "unknown": "unknown",
            },
        )

        graph.add_edge("ticket", "compliance")
        graph.add_edge("knowledge", "compliance")
        graph.add_edge("complaint", "compliance")
        graph.add_edge("unknown", "compliance")

        graph.add_edge("compliance", "human_review_check")
        graph.add_conditional_edges(
            "human_review_check",
            self._choose_review_route,
            {
                "create_review_task": "create_review_task",
                "save_memory": "save_memory",
            },
        )
        graph.add_edge("create_review_task", "save_memory")
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
            **create_working_memory(),
        }

        if self.graph is None:
            try:
                self.graph = self._build_graph()
            except ModuleNotFoundError as exc:
                if exc.name != "langgraph":
                    raise
                self.graph = None

        if self.graph is None:
            final_state = await self._run_fallback_graph(initial_state)
        else:
            final_state = await self.graph.ainvoke(initial_state)

        return {
            "response": final_state["final_response"],
            "session_id": final_state["session_id"],
            "trace_id": final_state.get("trace_id", ""),
            "intent": final_state.get("intent", "unknown"),
            "intent_confidence": final_state.get("intent_confidence", 0.0),
            "intent_reason": final_state.get("intent_reason", ""),
            "compliance_passed": final_state.get("compliance_passed", False),
            "need_human_review": final_state.get("need_human_review", False),
            "review_task_id": final_state.get("review_task_id"),
            "memory_count": final_state.get("memory_count", 0),
            "context_trace": final_state.get("context_trace", {}),
        }

    async def _run_fallback_graph(self, initial_state: ChatState) -> ChatState:
        state: ChatState = dict(initial_state)

        for node in [
            self._load_memory_node,
            self._build_context_node,
            self._route_intent_node,
        ]:
            state.update(await node(state))

        route = self._choose_route(state)
        if route == "ticket":
            state.update(await self._ticket_node(state))
        elif route == "knowledge":
            state.update(await self._knowledge_node(state))
        elif route == "complaint":
            state.update(await self._complaint_node(state))
        else:
            state.update(await self._unknown_node(state))

        state.update(await self._compliance_node(state))
        state.update(await self._human_review_check_node(state))

        if self._choose_review_route(state) == "create_review_task":
            state.update(await self._create_review_task_node(state))

        state.update(await self._save_memory_node(state))
        return state

    async def _load_memory_node(self, state: ChatState) -> dict:
        session_id = state["session_id"]
        user_id = state["user_id"]

        short_context = self.memory.get_recent_context(session_id)
        long_context = self.long_term_memory_tool.build_user_context(user_id)

        context_parts = []

        if long_context:
            context_parts.append("【长期记忆】")
            context_parts.append(long_context)

        if short_context:
            context_parts.append("【短期记忆】")
            context_parts.append(short_context)

        return {
            "context": "\n".join(context_parts),
            "current_agent": "memory_loader",
        }

    async def _build_context_node(self, state: ChatState) -> dict:
        optimized_context, context_trace = self._build_agent_context(
            state,
            agent_name="router",
            current_task="Classify the user's current customer-service intent.",
        )

        return {
            "optimized_context": optimized_context,
            "context_trace": context_trace,
            "current_agent": "context_builder",
        }

    async def _route_intent_node(self, state: ChatState) -> dict:
        start_time = time.perf_counter()
        intent_result = await self.intent_router.run(
            message=state["message"],
            context=state.get("optimized_context") or state.get("context", ""),
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        self.trace_tool.record_trace(
            trace_id=state["trace_id"],
            session_id=state["session_id"],
            user_id=state["user_id"],
            agent_name="IntentRouterAgent 意图识别智能体",
            node_name="router_agent_node",
            step=1,
            action="route",
            tool_name=None,
            tool_args={"message": state["message"]},
            tool_result={
                "intent": intent_result.intent,
                "confidence": intent_result.confidence,
                "reason": intent_result.reason,
            },
            status="success",
            error=None,
            latency_ms=latency_ms,
        )

        return {
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
            "intent_reason": intent_result.reason,
        }

    def _choose_route(self, state: ChatState) -> RouteType:
        intent = state.get("intent", "unknown")

        if intent == "complaint":
            return "complaint"

        if intent in ["refund", "account", "ticket"]:
            return "ticket"

        if intent == "knowledge":
            return "knowledge"

        return "unknown"

    async def _ticket_node(self, state: ChatState) -> dict:
        optimized_context, context_trace = self._build_agent_context(
            state,
            agent_name="ticket",
            current_task="Resolve ticket, refund, order, or account-service needs.",
        )
        raw_response = await self.ticket_agent.run(
            message=state["message"],
            user_id=state["user_id"],
            context=optimized_context or state.get("context", ""),
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        )

        return {
            "raw_response": raw_response,
            "current_agent": "ticket_agent",
            "optimized_context": optimized_context,
            "context_trace": context_trace,
        }

    async def _complaint_node(self, state: ChatState) -> dict:
        optimized_context, context_trace = self._build_agent_context(
            state,
            agent_name="complaint",
            current_task="Handle a complaint or human escalation request.",
        )
        raw_response = await self.complaint_agent.run(
            message=state["message"],
            user_id=state["user_id"],
            context=optimized_context or state.get("context", ""),
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        )

        return {
            "raw_response": raw_response,
            "current_agent": "complaint_agent",
            "optimized_context": optimized_context,
            "context_trace": context_trace,
        }

    async def _knowledge_node(self, state: ChatState) -> dict:
        optimized_context, context_trace = self._build_agent_context(
            state,
            agent_name="knowledge",
            current_task="Answer the user's knowledge-base or policy question.",
        )
        raw_response = await self.knowledge_agent.run(
            message=state["message"],
            user_id=state["user_id"],
            context=optimized_context or state.get("context", ""),
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        )

        return {
            "raw_response": raw_response,
            "current_agent": "knowledge_agent",
            "optimized_context": optimized_context,
            "context_trace": context_trace,
        }

    async def _unknown_node(self, state: ChatState) -> dict:
        return {
            "raw_response": "抱歉，我暂时没有理解你的问题，可以换一种说法吗？"
        }

    async def _compliance_node(self, state: ChatState) -> dict:
        optimized_context, context_trace = self._build_agent_context(
            state,
            agent_name="compliance",
            current_task="Check the draft response for compliance and safety.",
        )
        start_time = time.perf_counter()
        if hasattr(self.compliance_agent, "run"):
            compliance_result = await self.compliance_agent.run(
                state["raw_response"],
                context=optimized_context or state.get("context", ""),
            )
        else:
            compliance_result = await self.compliance_agent.check(
                state["raw_response"]
            )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if not compliance_result["passed"]:
            final_response = (
                "抱歉，系统检测到回复内容可能存在合规风险，"
                "已转交人工客服进一步处理。"
            )
        else:
            final_response = compliance_result["sanitized_content"]

        self.trace_tool.record_trace(
            trace_id=state["trace_id"],
            session_id=state["session_id"],
            user_id=state["user_id"],
            agent_name="ComplianceAgent 合规审查智能体",
            node_name="compliance_agent_node",
            step=1,
            action="compliance_check",
            tool_name=None,
            tool_args={"content": state["raw_response"]},
            tool_result=compliance_result,
            status="success" if compliance_result["passed"] else "failed",
            error=None if compliance_result["passed"] else "合规审查未通过",
            latency_ms=latency_ms,
        )

        return {
            "final_response": final_response,
            "compliance_passed": compliance_result["passed"],
            "compliance_violations": compliance_result.get("violations", []),
            "compliance_decision_source": compliance_result.get(
                "decision_source"
            ),
            "compliance_result": compliance_result,
            "optimized_context": optimized_context,
            "context_trace": context_trace,
        }

    def _build_agent_context(
        self,
        state: ChatState,
        agent_name: str,
        current_task: str,
    ) -> tuple[str, dict]:
        return self.context_builder.build(
            message=state["message"],
            state=dict(state),
            agent_name=agent_name,
            system_policy=(
                "You are Smart-CS, a multi-agent customer service system. "
                "Use provided memory, evidence, and policies faithfully."
            ),
            current_task=current_task,
            working_state=dict(state),
            base_context=state.get("context", ""),
            long_context="",
            rag_context=None,
            tool_observations=state.get("tool_observations", []),
            output_requirements=(
                "Return a concise, polite customer-service answer. "
                "Do not invent tickets, orders, policies, or tool results."
            ),
        )

    async def _human_review_check_node(self, state: ChatState) -> dict:
        reasons = self._build_human_review_reasons(state)

        if not reasons:
            return {
                "need_human_review": False,
                "review_reason": "",
            }

        return {
            "need_human_review": True,
            "review_reason": "；".join(reasons),
            "final_response": self._format_human_review_response(state),
        }

    def _choose_review_route(self, state: ChatState) -> ReviewRouteType:
        if state.get("need_human_review"):
            return "create_review_task"

        return "save_memory"

    async def _create_review_task_node(self, state: ChatState) -> dict:
        task = self.human_review_tool.create_review_task(
            session_id=state["session_id"],
            trace_id=state["trace_id"],
            user_id=state["user_id"],
            reason=state.get("review_reason", "需要人工客服介入"),
            request_content=state["message"],
            agent_response=state["final_response"],
        )

        self.trace_tool.record_trace(
            trace_id=state["trace_id"],
            session_id=state["session_id"],
            user_id=state["user_id"],
            agent_name="Supervisor",
            node_name="human_review_node",
            step=1,
            action="create_review_task",
            tool_name="create_review_task",
            tool_args={
                "reason": state.get("review_reason", ""),
            },
            tool_result={
                "review_id": task.review_id,
                "status": task.status,
            },
            status="success",
            error=None,
            latency_ms=None,
        )

        return {
            "review_task_id": task.review_id,
        }

    def _build_human_review_reasons(self, state: ChatState) -> list[str]:
        message = state["message"]
        final_response = state.get("final_response", "")
        reasons: list[str] = []

        if state.get("intent") == "complaint":
            reasons.append("用户发起投诉")

        if not state.get("compliance_passed", True):
            reasons.append("合规审查未通过")

        if state.get("intent_confidence", 1.0) < 0.6:
            reasons.append("路由置信度较低")

        if any(keyword in message for keyword in ["人工", "人工客服", "转人工"]):
            reasons.append("用户要求人工客服")

        if any(keyword in final_response for keyword in ["创建失败", "查询失败", "建议转人工"]):
            reasons.append("自动处理未完成")

        if self._contains_high_amount(message):
            reasons.append("疑似高金额业务")

        return reasons

    def _format_human_review_response(self, state: ChatState) -> str:
        if state.get("intent") == "complaint":
            return (
                f"{state.get('final_response', '')}\n"
                "同时，该问题已进入人工审核队列，请等待客服进一步处理。"
            ).strip()

        if not state.get("compliance_passed", True):
            return (
                "抱歉，系统检测到回复内容可能存在合规风险，"
                "已转交人工客服进一步处理。"
            )

        return (
            f"{state.get('final_response', '')}\n"
            "该问题已进入人工审核队列，请等待客服进一步处理。"
        ).strip()

    def _contains_high_amount(self, text: str) -> bool:
        import re

        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(元|块|人民币)?", text):
            try:
                if float(match.group(1)) >= 10000:
                    return True
            except ValueError:
                continue

        return False

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

        # 2. 写入 MySQL 短期记忆
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

                # 3. 更新长期记忆：用户画像
        self.long_term_memory_tool.update_profile_by_intent(
            user_id=user_id,
            intent=state.get("intent", "unknown"),
        )

        # 4. 保存会话摘要
        summary = (
            f"用户问题：{user_message}\n"
            f"系统回复：{assistant_message}\n"
            f"识别意图：{state.get('intent', 'unknown')}"
        )

        self.long_term_memory_tool.upsert_conversation_summary(
            session_id=session_id,
            user_id=user_id,
            summary=summary[:1000],
            message_count=self.memory.count(session_id),
        )

        return {
            "memory_count": self.memory.count(session_id),
        }
