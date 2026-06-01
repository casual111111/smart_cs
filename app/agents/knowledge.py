from app.agents.llm_agent import LLMAgent
from app.tools.knowledge_tool import KnowledgeTool


class KnowledgeAgent:
    """
    知识问答 Agent。

    当前版本：
    - KnowledgeAgent 负责组织回答逻辑
    - KnowledgeTool 负责检索知识库
    - LLMAgent 负责生成自然语言回答
    """

    def __init__(self):
        self.knowledge_tool = KnowledgeTool()
        self.llm_agent = LLMAgent()

    async def run(self, message: str) -> str:
        results = self.knowledge_tool.search_knowledge(
            query=message,
            top_k=3,
        )

        if not results:
            return "抱歉，知识库中暂时没有找到相关信息，建议转人工客服处理。"

        context = self._build_context(results)

        if self.llm_agent.llm.enabled:
            answer = await self.llm_agent.generate_answer(
                question=message,
                context=context,
            )

            if answer:
                sources = self._build_sources(results)
                return f"{answer}\n\n参考来源：{sources}"

        return self._fallback_answer(results)

    def _build_context(self, results) -> str:
        parts = []

        for index, item in enumerate(results, start=1):
            parts.append(
                f"【资料 {index}】\n"
                f"来源：{item.source}\n"
                f"内容：{item.content}"
            )

        return "\n\n".join(parts)

    def _build_sources(self, results) -> str:
        sources = []

        for item in results:
            if item.source not in sources:
                sources.append(item.source)

        return "、".join(sources)

    def _fallback_answer(self, results) -> str:
        answer_parts = ["根据知识库，我找到以下相关信息："]

        for index, item in enumerate(results, start=1):
            answer_parts.append(
                f"\n【资料 {index}】\n"
                f"来源：{item.source}\n"
                f"相关度：{item.score:.1f}\n"
                f"{item.content}"
            )

        answer_parts.append("\n\n以上内容来自本地知识库，仅供参考。")

        return "\n".join(answer_parts)

    def reload_knowledge_base(self) -> None:
        self.knowledge_tool.reload_knowledge_base()