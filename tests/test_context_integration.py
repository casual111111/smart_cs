import asyncio
from types import SimpleNamespace

from app.supervisor import Supervisor


class FakeMemory:
    def __init__(self):
        self.messages = []

    def get_recent_context(self, session_id):
        return "user: previous message\nassistant: previous answer"

    def add_message(self, session_id, role, content):
        self.messages.append((session_id, role, content))

    def count(self, session_id):
        return len(self.messages)


class FakeLongTermMemoryTool:
    def build_user_context(self, user_id):
        return "user profile: normal customer"

    def update_profile_by_intent(self, user_id, intent):
        return None

    def upsert_conversation_summary(
        self,
        session_id,
        user_id,
        summary,
        message_count,
    ):
        return None


class FakeRouter:
    def __init__(self):
        self.context = ""

    async def run(self, message, context=""):
        self.context = context
        return SimpleNamespace(
            intent="knowledge",
            confidence=0.9,
            reason="test router",
        )


class FakeKnowledgeAgent:
    def __init__(self):
        self.context = ""

    async def run(
        self,
        message,
        user_id,
        context="",
        session_id="",
        trace_id=None,
    ):
        self.context = context
        return "knowledge answer"


class FakeComplianceAgent:
    def __init__(self):
        self.context = ""

    async def run(self, content, context=""):
        self.context = context
        return {
            "passed": True,
            "violations": [],
            "sanitized_content": content,
            "decision_source": "test",
        }


class FakeTraceTool:
    def record_trace(self, **kwargs):
        return kwargs


class FakeChatHistoryTool:
    def ensure_session(self, **kwargs):
        return None

    def add_message(self, **kwargs):
        return None


def test_supervisor_handle_generates_context_trace():
    supervisor = Supervisor()
    supervisor.memory = FakeMemory()
    supervisor.long_term_memory_tool = FakeLongTermMemoryTool()
    supervisor.intent_router = FakeRouter()
    supervisor.knowledge_agent = FakeKnowledgeAgent()
    supervisor.compliance_agent = FakeComplianceAgent()
    supervisor.trace_tool = FakeTraceTool()
    supervisor.chat_history_tool = FakeChatHistoryTool()

    result = asyncio.run(
        supervisor.handle(
            message="refund policy?",
            user_id="user-1",
            session_id="session-1",
        )
    )

    assert result["response"] == "knowledge answer"
    assert result["context_trace"]["total_candidates"] >= 3
    assert result["context_trace"]["used_tokens"] > 0
    assert "[Role & Policies]" in supervisor.intent_router.context
    assert "[Task]" in supervisor.knowledge_agent.context
    assert "[State]" in supervisor.compliance_agent.context
