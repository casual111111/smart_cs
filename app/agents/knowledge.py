from app.agents.base_tool_agent import BaseToolAgent
from app.tools.knowledge_tool import KnowledgeTool


class KnowledgeAgent(BaseToolAgent):
    """
    知识问答 Agent。

    当前版本：
    - 继承 BaseToolAgent
    - 优先调用 build_rag_context / search_knowledge
    - LLM 可用时走 ReAct 工具调用
    - LLM 不可用时走本地兜底检索
    """

    def __init__(self):
        super().__init__(
            agent_name="KnowledgeAgent 知识库问答智能体",
            allowed_tools=[
                "build_rag_context",
                "search_knowledge",
            ],
        )

        # 保留直接工具，给 reload_knowledge_base 和 fallback 用
        self.knowledge_tool = KnowledgeTool()

    async def run(
        self,
        message: str,
        user_id: str,
        context: str = "",
        session_id: str = "",
        trace_id: str | None = None,
    ) -> str:
        """
        聊天入口。

        LLM 开启：
        - 让 KnowledgeAgent 自己决定是否调用 build_rag_context / search_knowledge

        LLM 未开启：
        - 直接走本地知识库检索，避免测试环境不可用
        """
        if not self.llm.enabled:
            return self._fallback_search(message)

        system_instruction = """
你负责处理客服系统中的知识库问答问题。

你的职责：
1. 回答政策、规则、流程、说明类问题。
2. 必须优先调用 build_rag_context 构建 RAG 上下文；如果不可用，再调用 search_knowledge。
3. 不要编造知识库中不存在的政策。
4. 如果工具没有返回结果，要告诉用户知识库暂时没有相关信息，并建议转人工。
5. 如果工具返回了多个资料片段，要综合整理成自然语言回答。
6. 最终回答要简洁、清楚、适合客服场景。
7. 如果引用了知识库内容，最后要说明参考来源。
"""

        result = await self.react_with_tools(
            message=message,
            user_id=user_id,
            context=context,
            system_instruction=system_instruction,
            session_id=session_id,
            node_name="knowledge_agent_node",
            trace_id=trace_id,
            max_steps=4,
        )

        return self._ensure_sources_in_answer(
            answer=result.final_answer,
            steps=result.steps,
        )

    def _fallback_search(self, message: str) -> str:
        """
        没有配置 LLM 时的兜底逻辑。

        这样 pytest、本地开发、无 API Key 环境也可以跑通。
        """
        if hasattr(self.knowledge_tool, "build_rag_context"):
            rag_context = self.knowledge_tool.build_rag_context(
                query=message,
                top_k=3,
            )
            results = rag_context.chunks
            sources = rag_context.sources
        else:
            results = self.knowledge_tool.search_knowledge(
                query=message,
                top_k=3,
            )
            sources = self._dedupe_sources(
                getattr(item, "source", "")
                for item in results
            )

        if not results:
            return "抱歉，知识库中暂时没有找到相关信息，建议转人工客服处理。"

        answer_parts = ["根据知识库，我找到以下相关信息："]

        for index, item in enumerate(results, start=1):
            answer_parts.append(
                f"\n〖资料 {index}〗\n"
                f"来源：{item.source}\n"
                f"相关度：{item.score:.1f}\n"
                f"{item.content}"
            )

        answer_parts.append(
            self._format_sources(sources)
            + "\n以上内容来自本地知识库，仅供参考。"
        )
        return "\n".join(answer_parts)

    def _ensure_sources_in_answer(self, answer: str, steps: list) -> str:
        sources = []

        for step in steps:
            observation = getattr(step, "observation", None)

            if not isinstance(observation, dict):
                continue

            sources.extend(observation.get("sources") or [])
            sources.extend(
                item.get("source", "")
                for item in observation.get("items", [])
                if isinstance(item, dict)
            )

        sources = self._dedupe_sources(sources)

        if not sources:
            return answer

        if any(source and source in answer for source in sources):
            return answer

        return answer.rstrip() + self._format_sources(sources)

    def _format_sources(self, sources: list[str]) -> str:
        sources = self._dedupe_sources(sources)

        if not sources:
            return ""

        source_lines = "\n".join(
            f"- {source}"
            for source in sources
        )

        return f"\n\n参考来源：\n{source_lines}"

    def _dedupe_sources(self, sources) -> list[str]:
        return [
            source
            for source in dict.fromkeys(sources)
            if source
        ]

    def reload_knowledge_base(self) -> None:
        self.knowledge_tool.reload_knowledge_base()
